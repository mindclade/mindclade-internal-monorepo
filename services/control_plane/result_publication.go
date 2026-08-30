package controlplane

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

var ErrResultPublication = errors.New("result artifact publication failed")

// ResultArtifactVerifier independently attests that an exact digest and size
// are committed in the authorized artifact store.
type ResultArtifactVerifier interface {
	VerifyCommitted(context.Context, Scope, string, uint64) error
}

// ResultPublication owns capability issuance and committed-object attestation.
type ResultPublication struct {
	issuer   *ArtifactCapabilityIssuer
	verifier ResultArtifactVerifier
}

func NewResultPublication(issuer *ArtifactCapabilityIssuer, verifier ResultArtifactVerifier) (*ResultPublication, error) {
	if issuer == nil || verifier == nil {
		return nil, fmt.Errorf("%w: issuer and verifier are required", ErrInvalidRequest)
	}
	return &ResultPublication{issuer: issuer, verifier: verifier}, nil
}

// HTTPResultArtifactVerifier uses a fresh download capability for an
// authenticated HEAD request. The artifact proxy re-hashes the committed bytes
// and returns exact digest and size headers.
type HTTPResultArtifactVerifier struct {
	baseURL string
	issuer  *ArtifactCapabilityIssuer
	client  *http.Client
}

func NewHTTPResultArtifactVerifier(baseURL string, issuer *ArtifactCapabilityIssuer, timeout time.Duration) (*HTTPResultArtifactVerifier, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, fmt.Errorf("%w: artifact proxy URL is invalid", ErrInvalidRequest)
	}
	if issuer == nil || timeout <= 0 || timeout > 30*time.Second {
		return nil, fmt.Errorf("%w: artifact verifier configuration is invalid", ErrInvalidRequest)
	}
	dialer := &net.Dialer{Timeout: timeout, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		DialContext:           dialer.DialContext,
		DisableCompression:    true,
		MaxIdleConns:          8,
		MaxIdleConnsPerHost:   8,
		IdleConnTimeout:       30 * time.Second,
		ResponseHeaderTimeout: timeout,
	}
	return &HTTPResultArtifactVerifier{
		baseURL: strings.TrimRight(baseURL, "/"), issuer: issuer,
		client: &http.Client{Transport: transport, Timeout: timeout},
	}, nil
}

func (v *HTTPResultArtifactVerifier) VerifyCommitted(ctx context.Context, scope Scope, digest string, sizeBytes uint64) error {
	capability, err := v.issuer.IssueDownload(scope, digest, sizeBytes)
	if err != nil {
		return fmt.Errorf("%w: issue metadata capability: %v", ErrResultPublication, err)
	}
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodHead,
		v.baseURL+"/v1alpha1/artifacts/"+url.PathEscape(digest),
		nil,
	)
	if err != nil {
		return fmt.Errorf("%w: construct metadata request", ErrResultPublication)
	}
	request.Header.Set("X-Mindclade-Capability", capability)
	response, err := v.client.Do(request)
	if err != nil {
		return fmt.Errorf("%w: artifact metadata request failed", ErrResultPublication)
	}
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 1024))
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("%w: artifact metadata status %d", ErrResultPublication, response.StatusCode)
	}
	committedSize, err := strconv.ParseUint(response.Header.Get("X-Mindclade-Artifact-Size"), 10, 64)
	if err != nil || committedSize != sizeBytes || response.Header.Get("X-Mindclade-Artifact-Digest") != digest {
		return fmt.Errorf("%w: committed artifact metadata mismatch", ErrResultPublication)
	}
	return nil
}

func resultArtifactURI(digest string) string {
	return "artifact://sha256/" + strings.TrimPrefix(digest, "sha256:")
}
