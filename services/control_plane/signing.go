package controlplane

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"os"
	"time"
)

const (
	artifactCapabilityDomain = "mindclade-artifact-capability-v2"
	artifactCapabilityTTL    = 15 * time.Minute
	minimumCapabilityTTL     = time.Minute
	maximumCapabilityTTL     = 7 * 24 * time.Hour
)

// Ed25519SigningKey is a rotation-addressable private key loaded from a
// deployment-owned secret. It never marshals its private bytes.
type Ed25519SigningKey struct {
	keyID      string
	privateKey ed25519.PrivateKey
}

func NewEd25519SigningKey(keyID string, privateKey ed25519.PrivateKey) (*Ed25519SigningKey, error) {
	if !signingKeyIDPattern.MatchString(keyID) || len(privateKey) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("%w: invalid Ed25519 signing key", ErrInvalidRequest)
	}
	return &Ed25519SigningKey{keyID: keyID, privateKey: privateKey}, nil
}

func LoadEd25519SigningKey(keyID, filename string) (*Ed25519SigningKey, error) {
	encoded, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("read Ed25519 private key: %w", err)
	}
	block, remainder := pem.Decode(encoded)
	if block == nil || len(remainder) != 0 || block.Type != "PRIVATE KEY" {
		return nil, errors.New("Ed25519 private key must be one PKCS#8 PEM block")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parse Ed25519 private key: %w", err)
	}
	privateKey, ok := parsed.(ed25519.PrivateKey)
	if !ok {
		return nil, errors.New("private key is not Ed25519")
	}
	return NewEd25519SigningKey(keyID, privateKey)
}

func (k *Ed25519SigningKey) KeyID() string { return k.keyID }

func (k *Ed25519SigningKey) Sign(message []byte) []byte {
	return ed25519.Sign(k.privateKey, message)
}

// artifactCapability matches services/artifact_proxy/src/capability.rs. The
// outer transport is URL-safe unpadded base64(JSON); the signature is also
// URL-safe unpadded base64.
type artifactCapability struct {
	Scope        Scope  `json:"scope"`
	Digest       string `json:"digest"`
	Operation    string `json:"operation"`
	ExpiresUnix  uint64 `json:"expires_unix"`
	MaxSizeBytes uint64 `json:"max_size_bytes"`
	SessionID    string `json:"session_id"`
	Nonce        string `json:"nonce"`
	KeyID        string `json:"key_id"`
	Signature    string `json:"signature"`
}

type ArtifactCapabilityIssuer struct {
	key     *Ed25519SigningKey
	clock   Clock
	entropy io.Reader
	ttl     time.Duration
}

func NewArtifactCapabilityIssuer(key *Ed25519SigningKey) (*ArtifactCapabilityIssuer, error) {
	return NewArtifactCapabilityIssuerWithTTL(key, artifactCapabilityTTL)
}

// NewArtifactCapabilityIssuerWithTTL creates an issuer with an explicit,
// bounded lifetime. Long-lived staging capabilities use this constructor so
// their expiry can be proven to cover queue, startup, and execution bounds;
// result-transfer capabilities retain the shorter default constructor.
func NewArtifactCapabilityIssuerWithTTL(key *Ed25519SigningKey, ttl time.Duration) (*ArtifactCapabilityIssuer, error) {
	if key == nil {
		return nil, fmt.Errorf("%w: artifact capability key is required", ErrInvalidRequest)
	}
	if ttl < minimumCapabilityTTL || ttl > maximumCapabilityTTL {
		return nil, fmt.Errorf("%w: artifact capability TTL must be within 1m..7d", ErrInvalidRequest)
	}
	return &ArtifactCapabilityIssuer{key: key, clock: systemClock{}, entropy: rand.Reader, ttl: ttl}, nil
}

func newArtifactCapabilityIssuerForTest(key *Ed25519SigningKey, clock Clock, entropy io.Reader, ttl time.Duration) *ArtifactCapabilityIssuer {
	return &ArtifactCapabilityIssuer{key: key, clock: clock, entropy: entropy, ttl: ttl}
}

func (i *ArtifactCapabilityIssuer) IssueDownload(scope Scope, digest string, sizeBytes uint64) (string, error) {
	capability, _, err := i.issue(scope, digest, sizeBytes, "download", "download-")
	return capability, err
}

// IssueUpload returns one exact Rust-v2 upload capability and the session ID
// cryptographically bound into it.
func (i *ArtifactCapabilityIssuer) IssueUpload(scope Scope, digest string, sizeBytes uint64) (string, string, error) {
	return i.issue(scope, digest, sizeBytes, "upload", "upload-")
}

func (i *ArtifactCapabilityIssuer) issue(scope Scope, digest string, sizeBytes uint64, operation, sessionPrefix string) (string, string, error) {
	if err := scope.Validate(); err != nil {
		return "", "", err
	}
	if i == nil || i.key == nil || i.clock == nil || i.entropy == nil || i.ttl <= 0 ||
		!digestPattern.MatchString(digest) || sizeBytes == 0 ||
		(operation != "download" && operation != "upload") {
		return "", "", fmt.Errorf("%w: invalid artifact capability request", ErrInvalidRequest)
	}
	session, err := randomIdentifier(i.entropy, sessionPrefix)
	if err != nil {
		return "", "", err
	}
	nonce, err := randomIdentifier(i.entropy, "nonce-")
	if err != nil {
		return "", "", err
	}
	now := i.clock.Now()
	expires := now.Add(i.ttl).Unix()
	if expires <= now.Unix() {
		return "", "", fmt.Errorf("%w: capability expiry overflow", ErrInvalidRequest)
	}
	capability := artifactCapability{
		Scope: scope, Digest: digest, Operation: operation, ExpiresUnix: uint64(expires),
		MaxSizeBytes: sizeBytes, SessionID: session, Nonce: nonce, KeyID: i.key.KeyID(),
	}
	capability.Signature = base64.RawURLEncoding.EncodeToString(i.key.Sign(capability.signingBytes()))
	payload, err := json.Marshal(capability)
	if err != nil {
		return "", "", fmt.Errorf("marshal artifact capability: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(payload), session, nil
}

func (c artifactCapability) signingBytes() []byte {
	result := []byte(artifactCapabilityDomain)
	if c.Operation == "upload" {
		result = append(result, byte(1))
	} else {
		result = append(result, byte(0))
	}
	for _, value := range []string{
		c.Scope.TenantID,
		c.Scope.ProjectID,
		c.Digest,
		c.SessionID,
		c.Nonce,
		c.KeyID,
	} {
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(value)))
		result = append(result, length[:]...)
		result = append(result, value...)
	}
	var integer [8]byte
	binary.BigEndian.PutUint64(integer[:], c.ExpiresUnix)
	result = append(result, integer[:]...)
	binary.BigEndian.PutUint64(integer[:], c.MaxSizeBytes)
	result = append(result, integer[:]...)
	return result
}

func randomIdentifier(source io.Reader, prefix string) (string, error) {
	bytes := make([]byte, 16)
	if _, err := io.ReadFull(source, bytes); err != nil {
		return "", fmt.Errorf("generate capability entropy: %w", err)
	}
	return prefix + base64.RawURLEncoding.EncodeToString(bytes), nil
}
