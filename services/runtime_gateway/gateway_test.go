package runtimegateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

type staticAuthenticator struct{ claims Claims }

func (a staticAuthenticator) Authenticate(string) (Claims, error) { return a.claims, nil }

func TestGatewayEnforcesScopeAndOverwritesIdentityHeaders(t *testing.T) {
	var received http.Header
	upstream := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		received = request.Header.Clone()
		response.WriteHeader(http.StatusAccepted)
		_, _ = io.WriteString(response, `{"state":"queued"}`)
	}))
	defer upstream.Close()
	upstreamURL, _ := url.Parse(upstream.URL)
	auth := staticAuthenticator{Claims{"subject-1", "tenant-a", map[string]bool{"project-a": true}, time.Now().Add(time.Hour)}}
	signer, err := NewInternalIdentitySigner([]byte("0123456789abcdef0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	gateway := New(auth, upstreamURL, upstream.Client(), nil, signer)
	request := httptest.NewRequest(http.MethodPost, "/v1alpha1/tenants/tenant-a/projects/project-a/inference-jobs", strings.NewReader("{}"))
	request.Header.Set("Authorization", "Bearer signed-token")
	request.Header.Set("X-Mindclade-Principal", "attacker")
	response := httptest.NewRecorder()
	gateway.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("status = %d", response.Code)
	}
	if received.Get("X-Mindclade-Principal") != "subject-1" || received.Get("Authorization") != "" {
		t.Fatalf("forwarded headers = %v", received)
	}
	if received.Get("X-Mindclade-Internal-Assertion") == "" || received.Get("X-Mindclade-Internal-Nonce") == "" {
		t.Fatal("gateway did not authenticate its internal identity projection")
	}
	if received.Get("X-Mindclade-Content-SHA256") != "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a" {
		t.Fatalf("signed content digest = %q", received.Get("X-Mindclade-Content-SHA256"))
	}

	request = httptest.NewRequest(http.MethodGet, "/v1alpha1/tenants/tenant-b/projects/project-a/inference-jobs/job-11111111111111111111111111111111-00000001", nil)
	request.Header.Set("Authorization", "Bearer signed-token")
	response = httptest.NewRecorder()
	gateway.ServeHTTP(response, request)
	if response.Code != http.StatusForbidden {
		t.Fatalf("cross-tenant status = %d", response.Code)
	}
}

func TestRouteTemplateRemovesResourceIdentifiers(t *testing.T) {
	got := routeTemplate("/v1alpha1/tenants/tenant-secret/projects/project-secret/inference-jobs/job-secret/events")
	want := "/v1alpha1/tenants/{tenant}/projects/{project}/inference-jobs/{job}/events"
	if got != want {
		t.Fatalf("route template = %q, want %q", got, want)
	}
}
