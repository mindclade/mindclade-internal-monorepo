package controlplane

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	runtimegateway "github.com/mindclade/mindclade-internal-monorepo/services/runtime_gateway"
)

var testInternalSecret = []byte("0123456789abcdef0123456789abcdef")

func TestPublicHTTPServerDeadlineCoversReadAndMaximumHandlerWork(t *testing.T) {
	launchTimeout := 10 * time.Second
	artifactVerifyTimeout := 30 * time.Second
	handlerTimeout := MinimumHandlerTimeout(launchTimeout, artifactVerifyTimeout)
	server := NewPublicHTTPServer("127.0.0.1:0", http.NotFoundHandler(), handlerTimeout)
	if handlerTimeout != 70*time.Second || server.ReadTimeout != 15*time.Second ||
		server.WriteTimeout != handlerTimeout || server.ReadHeaderTimeout != 5*time.Second ||
		server.IdleTimeout != 60*time.Second || server.MaxHeaderBytes != 32*1024 {
		t.Fatalf("public server policy = %+v, minimum = %s", server, handlerTimeout)
	}
	if server.WriteTimeout < server.ReadTimeout+5*launchTimeout+5*time.Second {
		t.Fatalf("write timeout %s does not cover read and launch rollback", server.WriteTimeout)
	}
}

func TestInternalJobActionRequiresCanonicalRestartUniqueJobID(t *testing.T) {
	id, action, matched := internalJobAction(
		"/internal/v1alpha1/jobs/" + testJobID + "/complete",
	)
	if !matched || id != testJobID || action != "complete" {
		t.Fatalf("canonical route = (%q, %q, %v)", id, action, matched)
	}
	if _, _, matched := internalJobAction(
		"/internal/v1alpha1/jobs/job-00000001/complete",
	); matched {
		t.Fatal("legacy sequential job ID matched the internal route")
	}
}

func TestHTTPSubmitDoesNotLogOrEchoInputBytes(t *testing.T) {
	service, _, _ := fixture()
	identity, err := NewInternalIdentityVerifier(testInternalSecret)
	if err != nil {
		t.Fatal(err)
	}
	handler := NewHTTPHandler(service, identity)
	body := `{"model_digest":"` + modelDigest + `","input_artifact":"` + inputDigest + `","seed":1,"diffusion_steps":8}`
	request := httptest.NewRequest(http.MethodPost, "/v1alpha1/tenants/tenant-a/projects/project-a/inference-jobs", strings.NewReader(body))
	request.Header.Set("Idempotency-Key", "request-http-1")
	signer, _ := runtimegateway.NewInternalIdentitySigner(testInternalSecret)
	if err := signer.Sign(request, runtimegateway.Claims{
		Subject: "user-1", TenantID: "tenant-a", Projects: map[string]bool{"project-a": true}, ExpiresAt: time.Now().Add(time.Hour),
	}, "tenant-a", "project-a"); err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), "little_endian_data") {
		t.Fatal("response exposed tensor bytes")
	}
}

func TestHTTPRejectsForgedAndReplayedInternalIdentity(t *testing.T) {
	service, _, _ := fixture()
	identity, _ := NewInternalIdentityVerifier(testInternalSecret)
	handler := NewHTTPHandler(service, identity)
	signer, _ := runtimegateway.NewInternalIdentitySigner(testInternalSecret)
	makeRequest := func() *http.Request {
		request := httptest.NewRequest(http.MethodGet, "/v1alpha1/tenants/tenant-a/projects/project-a/inference-jobs/"+testJobID, nil)
		_ = signer.Sign(request, runtimegateway.Claims{Subject: "user-1", TenantID: "tenant-a", Projects: map[string]bool{"project-a": true}}, "tenant-a", "project-a")
		return request
	}
	request := makeRequest()
	request.Header.Set("X-Mindclade-Tenant", "tenant-b")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusForbidden {
		t.Fatalf("forged identity status = %d", response.Code)
	}
	request = makeRequest()
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	replay := request.Clone(request.Context())
	replayResponse := httptest.NewRecorder()
	handler.ServeHTTP(replayResponse, replay)
	if replayResponse.Code != http.StatusForbidden {
		t.Fatalf("replayed assertion status = %d", replayResponse.Code)
	}
}

func TestInternalIdentityNonceCapacityIsTenantIsolated(t *testing.T) {
	identity, err := NewInternalIdentityVerifier(testInternalSecret)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	identity.now = func() time.Time { return now }
	identity.nonces["tenant-a"] = &tenantNonceCache{nonces: make(map[string]time.Time, maximumNoncesPerTenant)}
	for index := 0; index < maximumNoncesPerTenant; index++ {
		identity.nonces["tenant-a"].nonces[fmt.Sprintf("occupied-%05d", index)] = now
	}
	signer, err := runtimegateway.NewInternalIdentitySigner(testInternalSecret)
	if err != nil {
		t.Fatal(err)
	}
	requestFor := func(tenant string) *http.Request {
		request := httptest.NewRequest(
			http.MethodGet,
			"/v1alpha1/tenants/"+tenant+"/projects/project-a/inference-jobs/"+testJobID,
			nil,
		)
		if err := signer.Sign(request, runtimegateway.Claims{
			Subject: "user-1", TenantID: tenant, Projects: map[string]bool{"project-a": true},
		}, tenant, "project-a"); err != nil {
			t.Fatal(err)
		}
		return request
	}
	if _, err := identity.Authenticate(requestFor("tenant-a")); !errors.Is(err, ErrForbidden) {
		t.Fatalf("saturated tenant error = %v", err)
	}
	if principal, err := identity.Authenticate(requestFor("tenant-b")); err != nil || principal.TenantID != "tenant-b" {
		t.Fatalf("independent tenant authentication = (%+v, %v)", principal, err)
	}
	for nonce := range identity.nonces["tenant-a"].nonces {
		identity.nonces["tenant-a"].nonces[nonce] = now.Add(-internalAssertionWindow - time.Second)
	}
	if principal, err := identity.Authenticate(requestFor("tenant-a")); err != nil || principal.TenantID != "tenant-a" {
		t.Fatalf("expired capacity authentication = (%+v, %v)", principal, err)
	}
}

func TestHTTPRejectsBodyAndIdempotencyMutationAfterGatewaySignature(t *testing.T) {
	service, _, _ := fixture()
	identity, _ := NewInternalIdentityVerifier(testInternalSecret)
	handler := NewHTTPHandler(service, identity)
	signer, _ := runtimegateway.NewInternalIdentitySigner(testInternalSecret)
	path := "/v1alpha1/tenants/tenant-a/projects/project-a/inference-jobs"
	original := `{"model_digest":"` + modelDigest + `","input_artifact":"` + inputDigest + `","seed":1,"diffusion_steps":8}`

	bodyMutation := httptest.NewRequest(http.MethodPost, path, strings.NewReader(original))
	bodyMutation.Header.Set("Idempotency-Key", "request-signed-1")
	if err := signer.Sign(bodyMutation, runtimegateway.Claims{
		Subject: "user-1", TenantID: "tenant-a", Projects: map[string]bool{"project-a": true},
	}, "tenant-a", "project-a"); err != nil {
		t.Fatal(err)
	}
	bodyMutation.Body = io.NopCloser(strings.NewReader(strings.Replace(original, `"seed":1`, `"seed":2`, 1)))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, bodyMutation)
	if response.Code != http.StatusForbidden {
		t.Fatalf("body mutation status = %d", response.Code)
	}

	headerMutation := httptest.NewRequest(http.MethodPost, path, strings.NewReader(original))
	headerMutation.Header.Set("Idempotency-Key", "request-signed-2")
	if err := signer.Sign(headerMutation, runtimegateway.Claims{
		Subject: "user-1", TenantID: "tenant-a", Projects: map[string]bool{"project-a": true},
	}, "tenant-a", "project-a"); err != nil {
		t.Fatal(err)
	}
	headerMutation.Header.Set("Idempotency-Key", "request-mutated-2")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, headerMutation)
	if response.Code != http.StatusForbidden {
		t.Fatalf("idempotency mutation status = %d", response.Code)
	}
}
