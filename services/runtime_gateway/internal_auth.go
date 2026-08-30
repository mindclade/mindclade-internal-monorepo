package runtimegateway

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

const minimumInternalSecretBytes = 32

var ErrInternalRequestTooLarge = errors.New("internal request body exceeds the signed size limit")

// InternalIdentitySigner authenticates gateway-derived identity to the control plane.
type InternalIdentitySigner struct {
	secret []byte
	now    func() time.Time
}

func NewInternalIdentitySigner(secret []byte) (*InternalIdentitySigner, error) {
	if len(secret) < minimumInternalSecretBytes {
		return nil, errors.New("internal identity secret must contain at least 32 bytes")
	}
	return &InternalIdentitySigner{secret: append([]byte(nil), secret...), now: time.Now}, nil
}

func (s *InternalIdentitySigner) Sign(request *http.Request, claims Claims, tenant, project string) error {
	bodyDigest, err := signedBodyDigest(request)
	if err != nil {
		return err
	}
	nonceBytes := make([]byte, 16)
	if _, err := rand.Read(nonceBytes); err != nil {
		return err
	}
	timestamp := strconv.FormatInt(s.now().Unix(), 10)
	nonce := base64.RawURLEncoding.EncodeToString(nonceBytes)
	request.Header.Set("X-Mindclade-Principal", claims.Subject)
	request.Header.Set("X-Mindclade-Tenant", tenant)
	request.Header.Set("X-Mindclade-Project", project)
	request.Header.Set("X-Mindclade-Content-SHA256", bodyDigest)
	request.Header.Set("X-Mindclade-Internal-Timestamp", timestamp)
	request.Header.Set("X-Mindclade-Internal-Nonce", nonce)
	mac := hmac.New(sha256.New, s.secret)
	for _, value := range []string{
		timestamp,
		nonce,
		request.Method,
		request.URL.EscapedPath(),
		request.URL.RawQuery,
		request.Header.Get("Content-Type"),
		request.Header.Get("Idempotency-Key"),
		bodyDigest,
		claims.Subject,
		tenant,
		project,
	} {
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(value)))
		_, _ = mac.Write(length[:])
		_, _ = mac.Write([]byte(value))
	}
	request.Header.Set("X-Mindclade-Internal-Assertion", base64.RawURLEncoding.EncodeToString(mac.Sum(nil)))
	return nil
}

func signedBodyDigest(request *http.Request) (string, error) {
	if request.Body == nil {
		sum := sha256.Sum256(nil)
		return "sha256:" + hex.EncodeToString(sum[:]), nil
	}
	payload, err := io.ReadAll(io.LimitReader(request.Body, maximumRequestBytes+1))
	if err != nil {
		return "", fmt.Errorf("read request body for internal assertion: %w", err)
	}
	if int64(len(payload)) > maximumRequestBytes {
		return "", ErrInternalRequestTooLarge
	}
	request.Body = io.NopCloser(bytes.NewReader(payload))
	request.ContentLength = int64(len(payload))
	request.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(payload)), nil
	}
	sum := sha256.Sum256(payload)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
