package controlplane

import (
	"fmt"
	"regexp"
	"sync"
)

// ArtifactKind distinguishes executable model bundles from tensor inputs.
type ArtifactKind string

const (
	ArtifactModel ArtifactKind = "model"
	ArtifactInput ArtifactKind = "input"
)

var signingKeyIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)

// ArtifactMetadata is the tenant-owned immutable metadata needed to stage an
// artifact without granting the scheduler ambient object-store access.
// Digest identifies the signed inner model manifest for model entries, while
// BundleArchiveDigest identifies the exact archive downloaded by the worker.
type ArtifactMetadata struct {
	Digest                 string       `json:"digest"`
	Kind                   ArtifactKind `json:"kind"`
	SizeBytes              uint64       `json:"size_bytes,omitempty"`
	BundleArchiveDigest    string       `json:"bundle_archive_digest,omitempty"`
	BundleArchiveSizeBytes uint64       `json:"bundle_archive_size_bytes,omitempty"`
	BundleSigningKeyID     string       `json:"bundle_signing_key_id,omitempty"`
}

func (m ArtifactMetadata) Validate() error {
	if !digestPattern.MatchString(m.Digest) {
		return fmt.Errorf("%w: catalog digest must be immutable", ErrInvalidRequest)
	}
	switch m.Kind {
	case ArtifactInput:
		if m.SizeBytes == 0 || m.BundleArchiveDigest != "" || m.BundleArchiveSizeBytes != 0 || m.BundleSigningKeyID != "" {
			return fmt.Errorf("%w: input catalog metadata is incomplete or contains model fields", ErrInvalidRequest)
		}
	case ArtifactModel:
		if m.SizeBytes != 0 || !digestPattern.MatchString(m.BundleArchiveDigest) || m.BundleArchiveSizeBytes == 0 || !signingKeyIDPattern.MatchString(m.BundleSigningKeyID) {
			return fmt.Errorf("%w: model catalog metadata is incomplete", ErrInvalidRequest)
		}
	default:
		return fmt.Errorf("%w: catalog artifact kind is invalid", ErrInvalidRequest)
	}
	return nil
}

// ArtifactCatalog is the object-level authorization boundary. Production
// implementations answer from a durable, tenant-scoped catalog.
type ArtifactCatalog interface {
	Owns(Scope, string, ArtifactKind) bool
	Resolve(Scope, string, ArtifactKind) (ArtifactMetadata, bool)
}

// MemoryArtifactCatalog is a concurrency-safe development/test catalog.
type MemoryArtifactCatalog struct {
	mu      sync.RWMutex
	entries map[string]ArtifactMetadata
}

func NewMemoryArtifactCatalog() *MemoryArtifactCatalog {
	return &MemoryArtifactCatalog{entries: make(map[string]ArtifactMetadata)}
}

func artifactKey(scope Scope, digest string, kind ArtifactKind) string {
	return scopeKey(scope, string(kind)+":"+digest)
}

// Register stores complete metadata suitable for artifact capability issuance.
func (c *MemoryArtifactCatalog) Register(scope Scope, metadata ArtifactMetadata) error {
	if err := scope.Validate(); err != nil {
		return err
	}
	if err := metadata.Validate(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	key := artifactKey(scope, metadata.Digest, metadata.Kind)
	if existing, ok := c.entries[key]; ok && existing != metadata {
		return fmt.Errorf("%w: catalog identity already has different metadata", ErrInvalidRequest)
	}
	c.entries[key] = metadata
	return nil
}

// Grant is retained for state-machine tests that need authorization but do not
// launch workers. Resolve deliberately reports these ownership-only records as
// incomplete so a production launcher always fails closed.
func (c *MemoryArtifactCatalog) Grant(scope Scope, digest string, kind ArtifactKind) error {
	if err := scope.Validate(); err != nil {
		return err
	}
	if !digestPattern.MatchString(digest) || (kind != ArtifactModel && kind != ArtifactInput) {
		return ErrInvalidRequest
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[artifactKey(scope, digest, kind)] = ArtifactMetadata{Digest: digest, Kind: kind}
	return nil
}

func (c *MemoryArtifactCatalog) Owns(scope Scope, digest string, kind ArtifactKind) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	_, ok := c.entries[artifactKey(scope, digest, kind)]
	return ok
}

func (c *MemoryArtifactCatalog) Resolve(scope Scope, digest string, kind ArtifactKind) (ArtifactMetadata, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	metadata, ok := c.entries[artifactKey(scope, digest, kind)]
	if !ok || metadata.Validate() != nil {
		return ArtifactMetadata{}, false
	}
	return metadata, true
}
