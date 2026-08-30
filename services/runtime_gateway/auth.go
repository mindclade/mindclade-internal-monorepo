// Package runtimegateway authenticates external requests and forwards a scoped identity.
package runtimegateway

import (
	"crypto"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"strings"
	"time"
)

var ErrUnauthenticated = errors.New("request authentication failed")

// Claims are the only identity fields forwarded to the control plane.
type Claims struct {
	Subject   string
	TenantID  string
	Projects  map[string]bool
	ExpiresAt time.Time
}

// Authenticator permits a test implementation without weakening the production verifier.
type Authenticator interface {
	Authenticate(token string) (Claims, error)
}

// OIDCVerifier validates an RS256 JWT against a configured issuer, audience, and key.
// Key rotation is handled by replacing the mounted public-key file and restarting the pod.
type OIDCVerifier struct {
	issuer   string
	audience string
	keyID    string
	key      *rsa.PublicKey
	now      func() time.Time
}

func NewOIDCVerifier(issuer, audience, keyID string, pemBytes []byte) (*OIDCVerifier, error) {
	if issuer == "" || audience == "" || keyID == "" {
		return nil, fmt.Errorf("%w: issuer, audience, and key ID are required", ErrUnauthenticated)
	}
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, fmt.Errorf("%w: public key is not PEM", ErrUnauthenticated)
	}
	var publicKey *rsa.PublicKey
	parsed, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err == nil {
		publicKey, _ = parsed.(*rsa.PublicKey)
	} else if certificate, certErr := x509.ParseCertificate(block.Bytes); certErr == nil {
		publicKey, _ = certificate.PublicKey.(*rsa.PublicKey)
	}
	if publicKey == nil || publicKey.N.BitLen() < 2048 {
		return nil, fmt.Errorf("%w: RSA public key must be at least 2048 bits", ErrUnauthenticated)
	}
	return &OIDCVerifier{issuer: issuer, audience: audience, keyID: keyID, key: publicKey, now: time.Now}, nil
}

type tokenHeader struct {
	Algorithm string `json:"alg"`
	Type      string `json:"typ"`
	KeyID     string `json:"kid"`
}

type tokenClaims struct {
	Issuer    string          `json:"iss"`
	Audience  json.RawMessage `json:"aud"`
	Subject   string          `json:"sub"`
	ExpiresAt int64           `json:"exp"`
	NotBefore int64           `json:"nbf"`
	IssuedAt  int64           `json:"iat"`
	TenantID  string          `json:"tenant_id"`
	Projects  []string        `json:"project_ids"`
}

func (v *OIDCVerifier) Authenticate(token string) (Claims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return Claims{}, ErrUnauthenticated
	}
	headerBytes, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return Claims{}, ErrUnauthenticated
	}
	var header tokenHeader
	if json.Unmarshal(headerBytes, &header) != nil || header.Algorithm != "RS256" || header.KeyID != v.keyID || (header.Type != "" && header.Type != "JWT") {
		return Claims{}, ErrUnauthenticated
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return Claims{}, ErrUnauthenticated
	}
	digest := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	if rsa.VerifyPKCS1v15(v.key, crypto.SHA256, digest[:], signature) != nil {
		return Claims{}, ErrUnauthenticated
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return Claims{}, ErrUnauthenticated
	}
	var raw tokenClaims
	if json.Unmarshal(payload, &raw) != nil || raw.Issuer != v.issuer || raw.Subject == "" || raw.TenantID == "" {
		return Claims{}, ErrUnauthenticated
	}
	now := v.now().Unix()
	const clockSkewSeconds int64 = 60
	const maximumTokenLifetimeSeconds int64 = 24 * 60 * 60
	if !audienceContains(raw.Audience, v.audience) ||
		raw.ExpiresAt <= now-clockSkewSeconds ||
		raw.NotBefore > now+clockSkewSeconds ||
		raw.IssuedAt <= 0 ||
		raw.IssuedAt > now+clockSkewSeconds ||
		raw.ExpiresAt-raw.IssuedAt > maximumTokenLifetimeSeconds {
		return Claims{}, ErrUnauthenticated
	}
	projects := make(map[string]bool, len(raw.Projects))
	for _, project := range raw.Projects {
		if project != "" {
			projects[project] = true
		}
	}
	if len(projects) == 0 {
		return Claims{}, ErrUnauthenticated
	}
	return Claims{raw.Subject, raw.TenantID, projects, time.Unix(raw.ExpiresAt, 0).UTC()}, nil
}

func audienceContains(raw json.RawMessage, expected string) bool {
	var single string
	if json.Unmarshal(raw, &single) == nil {
		return single == expected
	}
	var multiple []string
	if json.Unmarshal(raw, &multiple) != nil {
		return false
	}
	for _, audience := range multiple {
		if audience == expected {
			return true
		}
	}
	return false
}
