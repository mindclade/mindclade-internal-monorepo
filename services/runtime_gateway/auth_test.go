package runtimegateway

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"testing"
	"time"
)

func testVerifierAndToken(t *testing.T, mutate func(map[string]any)) (*OIDCVerifier, string) {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	public, err := x509.MarshalPKIXPublicKey(&key.PublicKey)
	if err != nil {
		t.Fatal(err)
	}
	verifier, err := NewOIDCVerifier("https://issuer.invalid", "mindclade-api", "test-key", pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: public}))
	if err != nil {
		t.Fatal(err)
	}
	verifier.now = func() time.Time { return time.Unix(1_700_000_000, 0) }
	header := map[string]any{"alg": "RS256", "typ": "JWT", "kid": "test-key"}
	claims := map[string]any{
		"iss": "https://issuer.invalid", "aud": "mindclade-api", "sub": "subject-1",
		"exp": int64(1_700_003_600), "nbf": int64(1_699_999_900), "iat": int64(1_699_999_900),
		"tenant_id": "tenant-a", "project_ids": []string{"project-a"},
	}
	if mutate != nil {
		mutate(claims)
	}
	encode := func(value any) string {
		data, _ := json.Marshal(value)
		return base64.RawURLEncoding.EncodeToString(data)
	}
	signingInput := encode(header) + "." + encode(claims)
	digest := sha256.Sum256([]byte(signingInput))
	signature, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, digest[:])
	if err != nil {
		t.Fatal(err)
	}
	return verifier, signingInput + "." + base64.RawURLEncoding.EncodeToString(signature)
}

func TestOIDCVerifierAcceptsValidRS256Token(t *testing.T) {
	verifier, token := testVerifierAndToken(t, nil)
	claims, err := verifier.Authenticate(token)
	if err != nil || claims.Subject != "subject-1" || !claims.Projects["project-a"] {
		t.Fatalf("claims = (%+v, %v)", claims, err)
	}
}

func TestOIDCVerifierRejectsExpiredAndTamperedTokens(t *testing.T) {
	verifier, expired := testVerifierAndToken(t, func(claims map[string]any) { claims["exp"] = int64(1) })
	if _, err := verifier.Authenticate(expired); err == nil {
		t.Fatal("expired token accepted")
	}
	verifier, token := testVerifierAndToken(t, nil)
	if _, err := verifier.Authenticate(token[:len(token)-2] + "xx"); err == nil {
		t.Fatal("tampered token accepted")
	}
}

func TestOIDCVerifierRejectsFutureAndOverlongTokens(t *testing.T) {
	verifier, future := testVerifierAndToken(t, func(claims map[string]any) {
		claims["nbf"] = int64(1_700_001_000)
	})
	if _, err := verifier.Authenticate(future); err == nil {
		t.Fatal("future not-before token accepted")
	}
	verifier, overlong := testVerifierAndToken(t, func(claims map[string]any) {
		claims["iat"] = int64(1_699_900_000)
	})
	if _, err := verifier.Authenticate(overlong); err == nil {
		t.Fatal("overlong token accepted")
	}
}
