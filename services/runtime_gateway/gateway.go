package runtimegateway

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const maximumRequestBytes int64 = 1 << 20

const defaultControlPlaneTimeout = 3 * time.Minute

const (
	publicReadTimeout      = 15 * time.Second
	responseDeadlineMargin = 5 * time.Second
)

// Gateway authenticates, authorizes path scope, and relays requests without tensor logging.
type Gateway struct {
	authenticator  Authenticator
	upstream       *url.URL
	client         *http.Client
	logger         *slog.Logger
	identitySigner *InternalIdentitySigner
}

func New(authenticator Authenticator, upstream *url.URL, client *http.Client, logger *slog.Logger, identitySigner *InternalIdentitySigner) *Gateway {
	if client == nil {
		client = &http.Client{Timeout: defaultControlPlaneTimeout}
	}
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	return &Gateway{authenticator, upstream, client, logger, identitySigner}
}

func (g *Gateway) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if request.URL.Path == "/healthz" {
		response.WriteHeader(http.StatusNoContent)
		return
	}
	if request.Method == http.MethodGet && request.URL.Path == "/metrics" {
		response.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = io.WriteString(response, "# TYPE mindclade_runtime_gateway_up gauge\nmindclade_runtime_gateway_up 1\n")
		return
	}
	tenant, project, ok := pathScope(request.URL.Path)
	if !ok {
		http.NotFound(response, request)
		return
	}
	token, ok := bearerToken(request.Header.Get("Authorization"))
	if !ok {
		writeGatewayError(response, http.StatusUnauthorized, "missing bearer token")
		return
	}
	claims, err := g.authenticator.Authenticate(token)
	if err != nil {
		writeGatewayError(response, http.StatusUnauthorized, "invalid bearer token")
		return
	}
	if claims.TenantID != tenant || !claims.Projects[project] {
		writeGatewayError(response, http.StatusForbidden, "request scope is not authorized")
		return
	}
	if request.Body != nil {
		request.Body = http.MaxBytesReader(response, request.Body, maximumRequestBytes)
	}
	upstreamRequest := request.Clone(request.Context())
	upstreamRequest.URL.Scheme = g.upstream.Scheme
	upstreamRequest.URL.Host = g.upstream.Host
	upstreamRequest.Host = g.upstream.Host
	upstreamRequest.RequestURI = ""
	upstreamRequest.Header = request.Header.Clone()
	upstreamRequest.Header.Del("Authorization")
	for name := range upstreamRequest.Header {
		if strings.HasPrefix(strings.ToLower(name), "x-mindclade-") {
			upstreamRequest.Header.Del(name)
		}
	}
	if g.identitySigner == nil {
		writeGatewayError(response, http.StatusServiceUnavailable, "internal identity unavailable")
		return
	}
	if err := g.identitySigner.Sign(upstreamRequest, claims, tenant, project); err != nil {
		if errors.Is(err, ErrInternalRequestTooLarge) {
			writeGatewayError(response, http.StatusRequestEntityTooLarge, "request body is too large")
		} else {
			writeGatewayError(response, http.StatusServiceUnavailable, "internal identity unavailable")
		}
		return
	}

	started := time.Now()
	upstreamResponse, err := g.client.Do(upstreamRequest)
	if err != nil {
		if errors.Is(err, http.ErrHandlerTimeout) || errors.Is(err, context.DeadlineExceeded) {
			writeGatewayError(response, http.StatusGatewayTimeout, "upstream deadline exceeded")
		} else {
			writeGatewayError(response, http.StatusBadGateway, "control plane unavailable")
		}
		return
	}
	defer upstreamResponse.Body.Close()
	copyResponseHeaders(response.Header(), upstreamResponse.Header)
	response.WriteHeader(upstreamResponse.StatusCode)
	_, copyErr := io.Copy(response, upstreamResponse.Body)
	g.logger.Info("request completed",
		"method", request.Method,
		"route", routeTemplate(request.URL.Path),
		"status", upstreamResponse.StatusCode,
		"duration_ms", time.Since(started).Milliseconds(),
		"subject", claims.Subject,
		"tenant_id", tenant,
		"project_id", project,
		"response_copy_error", copyErr != nil,
	)
}

// NewPublicHTTPServer applies the bounded public HTTP policy around a gateway.
func NewPublicHTTPServer(address string, handler http.Handler, controlPlaneBudget time.Duration) *http.Server {
	return newPublicHTTPServer(address, handler, controlPlaneBudget, publicReadTimeout)
}

// ControlPlaneClientTimeout leaves room beyond the control plane's complete
// server-side deadline for the response to traverse the gateway connection.
func ControlPlaneClientTimeout(controlPlaneBudget time.Duration) time.Duration {
	return controlPlaneBudget + responseDeadlineMargin
}

func newPublicHTTPServer(
	address string,
	handler http.Handler,
	controlPlaneBudget time.Duration,
	readTimeout time.Duration,
) *http.Server {
	upstreamTimeout := ControlPlaneClientTimeout(controlPlaneBudget)
	return &http.Server{
		Addr: address, Handler: handler,
		ReadHeaderTimeout: 5 * time.Second, ReadTimeout: readTimeout,
		WriteTimeout: readTimeout + upstreamTimeout + responseDeadlineMargin, IdleTimeout: 60 * time.Second,
		MaxHeaderBytes: 32 * 1024,
	}
}

func bearerToken(value string) (string, bool) {
	parts := strings.Fields(value)
	returnValue := ""
	if len(parts) == 2 && strings.EqualFold(parts[0], "Bearer") {
		returnValue = parts[1]
	}
	return returnValue, returnValue != ""
}

func pathScope(path string) (string, string, bool) {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) < 6 || parts[0] != "v1alpha1" || parts[1] != "tenants" || parts[3] != "projects" || parts[5] != "inference-jobs" {
		return "", "", false
	}
	return parts[2], parts[4], parts[2] != "" && parts[4] != ""
}

func routeTemplate(path string) string {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) >= 6 {
		parts[2], parts[4] = "{tenant}", "{project}"
		if len(parts) >= 7 {
			parts[6] = "{job}"
		}
	}
	return "/" + strings.Join(parts, "/")
}

func copyResponseHeaders(destination, source http.Header) {
	for name, values := range source {
		lower := strings.ToLower(name)
		if lower == "connection" || lower == "transfer-encoding" || lower == "set-cookie" {
			continue
		}
		for _, value := range values {
			destination.Add(name, value)
		}
	}
}

func writeGatewayError(response http.ResponseWriter, status int, message string) {
	response.Header().Set("Content-Type", "application/json")
	response.Header().Set("Cache-Control", "no-store")
	response.WriteHeader(status)
	_, _ = io.WriteString(response, `{"error":"`+message+`"}`+"\n")
}
