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

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

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

func TestGatewayDefaultClientCoversMaximumControlPlaneWork(t *testing.T) {
	upstream, _ := url.Parse("http://control-plane.invalid")
	gateway := New(staticAuthenticator{}, upstream, nil, nil, nil)
	if gateway.client.Timeout != defaultControlPlaneTimeout || gateway.client.Timeout < 155*time.Second {
		t.Fatalf("default upstream timeout = %s", gateway.client.Timeout)
	}
}

func TestGatewayReportsUpstreamDeadlineAsGatewayTimeout(t *testing.T) {
	upstream, _ := url.Parse("http://control-plane.invalid")
	client := &http.Client{
		Timeout: 10 * time.Millisecond,
		Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			<-request.Context().Done()
			return nil, request.Context().Err()
		}),
	}
	signer, err := NewInternalIdentitySigner([]byte("0123456789abcdef0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	auth := staticAuthenticator{Claims{"subject-1", "tenant-a", map[string]bool{"project-a": true}, time.Now().Add(time.Hour)}}
	gateway := New(auth, upstream, client, nil, signer)
	request := httptest.NewRequest(http.MethodPost, "/v1alpha1/tenants/tenant-a/projects/project-a/inference-jobs", strings.NewReader("{}"))
	request.Header.Set("Authorization", "Bearer signed-token")
	response := httptest.NewRecorder()
	gateway.ServeHTTP(response, request)
	if response.Code != http.StatusGatewayTimeout {
		t.Fatalf("upstream timeout status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestPublicHTTPServerHasBoundedReadAndWritePolicy(t *testing.T) {
	server := newPublicHTTPServer("127.0.0.1:0", http.NotFoundHandler(), 60*time.Second, 25*time.Millisecond)
	upstreamTimeout := ControlPlaneClientTimeout(60 * time.Second)
	if server.ReadHeaderTimeout != 5*time.Second || server.ReadTimeout != 25*time.Millisecond ||
		server.WriteTimeout != 25*time.Millisecond+upstreamTimeout+5*time.Second || server.IdleTimeout != 60*time.Second ||
		server.MaxHeaderBytes != 32*1024 {
		t.Fatalf("public server policy = %+v", server)
	}
	if server.WriteTimeout <= server.ReadTimeout+upstreamTimeout {
		t.Fatalf("write timeout %s does not cover read %s and upstream %s", server.WriteTimeout, server.ReadTimeout, upstreamTimeout)
	}
}
