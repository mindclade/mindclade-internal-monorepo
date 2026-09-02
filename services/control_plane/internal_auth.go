package controlplane

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"
)

const (
	internalAssertionWindow = 60 * time.Second
	maximumNoncesPerTenant  = 1_000
	maximumNonceTenants     = 1_024
)

type tenantNonceCache struct {
	mu     sync.Mutex
	nonces map[string]time.Time
	users  int
}

// InternalIdentityVerifier authenticates gateway-derived identity and rejects replay.
type InternalIdentityVerifier struct {
	secret []byte
	now    func() time.Time
	mu     sync.Mutex
	nonces map[string]*tenantNonceCache
}

func NewInternalIdentityVerifier(secret []byte) (*InternalIdentityVerifier, error) {
	if len(secret) < 32 {
		return nil, errors.New("internal identity secret must contain at least 32 bytes")
	}
	return &InternalIdentityVerifier{
		secret: append([]byte(nil), secret...), now: time.Now,
		nonces: make(map[string]*tenantNonceCache),
	}, nil
}

func (v *InternalIdentityVerifier) Authenticate(request *http.Request) (Principal, error) {
	timestampText := request.Header.Get("X-Mindclade-Internal-Timestamp")
	nonce := request.Header.Get("X-Mindclade-Internal-Nonce")
	assertionText := request.Header.Get("X-Mindclade-Internal-Assertion")
	principal := Principal{
		Subject:  request.Header.Get("X-Mindclade-Principal"),
		TenantID: request.Header.Get("X-Mindclade-Tenant"),
		Projects: map[string]bool{request.Header.Get("X-Mindclade-Project"): true},
	}
	timestamp, err := strconv.ParseInt(timestampText, 10, 64)
	if err != nil {
		return Principal{}, ErrForbidden
	}
	nonceBytes, err := base64.RawURLEncoding.DecodeString(nonce)
	if err != nil || len(nonceBytes) != 16 {
		return Principal{}, ErrForbidden
	}
	assertion, err := base64.RawURLEncoding.DecodeString(assertionText)
	if err != nil || len(assertion) != sha256.Size {
		return Principal{}, ErrForbidden
	}
	now := v.now().UTC()
	signedAt := time.Unix(timestamp, 0).UTC()
	if signedAt.Before(now.Add(-internalAssertionWindow)) || signedAt.After(now.Add(internalAssertionWindow)) {
		return Principal{}, ErrForbidden
	}
	project := request.Header.Get("X-Mindclade-Project")
	bodyDigest, err := authenticatedBodyDigest(request)
	if err != nil || request.Header.Get("X-Mindclade-Content-SHA256") != bodyDigest {
		return Principal{}, ErrForbidden
	}
	mac := hmac.New(sha256.New, v.secret)
	for _, value := range []string{
		timestampText,
		nonce,
		request.Method,
		request.URL.EscapedPath(),
		request.URL.RawQuery,
		request.Header.Get("Content-Type"),
		request.Header.Get("Idempotency-Key"),
		bodyDigest,
		principal.Subject,
		principal.TenantID,
		project,
	} {
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(value)))
		_, _ = mac.Write(length[:])
		_, _ = mac.Write([]byte(value))
	}
	if !hmac.Equal(assertion, mac.Sum(nil)) {
		return Principal{}, ErrForbidden
	}
	if principal.Authorize(Scope{TenantID: principal.TenantID, ProjectID: project}) != nil {
		return Principal{}, ErrForbidden
	}
	tenantNonces, acquired := v.acquireTenantNonceCache(principal.TenantID, now)
	if !acquired {
		return Principal{}, ErrForbidden
	}
	defer v.releaseTenantNonceCache(tenantNonces)
	tenantNonces.mu.Lock()
	defer tenantNonces.mu.Unlock()
	for value, seenAt := range tenantNonces.nonces {
		if seenAt.Before(now.Add(-internalAssertionWindow)) {
			delete(tenantNonces.nonces, value)
		}
	}
	if _, replayed := tenantNonces.nonces[nonce]; replayed || len(tenantNonces.nonces) >= maximumNoncesPerTenant {
		return Principal{}, ErrForbidden
	}
	tenantNonces.nonces[nonce] = now
	return principal, nil
}

func (v *InternalIdentityVerifier) acquireTenantNonceCache(tenant string, now time.Time) (*tenantNonceCache, bool) {
	v.mu.Lock()
	defer v.mu.Unlock()
	cache := v.nonces[tenant]
	if cache == nil {
		if len(v.nonces) >= maximumNonceTenants {
			cutoff := now.Add(-internalAssertionWindow)
			for candidate, existing := range v.nonces {
				if existing.users != 0 {
					continue
				}
				existing.mu.Lock()
				for value, seenAt := range existing.nonces {
					if seenAt.Before(cutoff) {
						delete(existing.nonces, value)
					}
				}
				empty := len(existing.nonces) == 0
				existing.mu.Unlock()
				if empty {
					delete(v.nonces, candidate)
				}
			}
		}
		if len(v.nonces) >= maximumNonceTenants {
			return nil, false
		}
		cache = &tenantNonceCache{nonces: make(map[string]time.Time)}
		v.nonces[tenant] = cache
	}
	cache.users++
	return cache, true
}

func (v *InternalIdentityVerifier) releaseTenantNonceCache(cache *tenantNonceCache) {
	v.mu.Lock()
	cache.users--
	v.mu.Unlock()
}

func authenticatedBodyDigest(request *http.Request) (string, error) {
	if request.Body == nil {
		sum := sha256.Sum256(nil)
		return "sha256:" + hex.EncodeToString(sum[:]), nil
	}
	payload, err := io.ReadAll(io.LimitReader(request.Body, (1<<20)+1))
	if err != nil || len(payload) > 1<<20 {
		return "", ErrForbidden
	}
	request.Body = io.NopCloser(bytes.NewReader(payload))
	request.ContentLength = int64(len(payload))
	sum := sha256.Sum256(payload)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
