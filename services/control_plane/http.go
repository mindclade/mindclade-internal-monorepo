package controlplane

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strconv"
	"strings"
)

// HTTPHandler exposes the JSON and SSE transport after identity verification at the gateway.
type HTTPHandler struct {
	service    *Service
	identity   *InternalIdentityVerifier
	dispatcher *Dispatcher
}

func NewHTTPHandler(service *Service, identity *InternalIdentityVerifier) http.Handler {
	return &HTTPHandler{service: service, identity: identity}
}

// NewHTTPHandlerWithDispatcher wires executable submit-to-launch behavior while
// preserving NewHTTPHandler for queue-only development and store tests.
func NewHTTPHandlerWithDispatcher(service *Service, identity *InternalIdentityVerifier, dispatcher *Dispatcher) http.Handler {
	return &HTTPHandler{service: service, identity: identity, dispatcher: dispatcher}
}

func (h *HTTPHandler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if request.URL.Path == "/healthz" {
		response.WriteHeader(http.StatusNoContent)
		return
	}
	if request.Method == http.MethodGet && request.URL.Path == "/metrics" {
		response.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = response.Write([]byte("# TYPE mindclade_control_plane_up gauge\nmindclade_control_plane_up 1\n"))
		return
	}
	if id, action, matched := internalJobAction(request.URL.Path); matched {
		if request.Method != http.MethodPost {
			response.Header().Set("Allow", http.MethodPost)
			response.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		switch action {
		case "complete":
			h.complete(response, request, id)
		case "result-upload-capability":
			h.authorizeResultUpload(response, request, id)
		default:
			http.NotFound(response, request)
		}
		return
	}
	if strings.HasPrefix(request.URL.Path, "/internal/") {
		http.NotFound(response, request)
		return
	}
	if h.identity == nil {
		writeError(response, ErrForbidden)
		return
	}
	principal, err := h.identity.Authenticate(request)
	if err != nil {
		writeError(response, err)
		return
	}
	parts := strings.Split(strings.Trim(request.URL.Path, "/"), "/")
	if len(parts) < 6 || parts[0] != "v1alpha1" || parts[1] != "tenants" || parts[3] != "projects" || parts[5] != "inference-jobs" {
		http.NotFound(response, request)
		return
	}
	scope := Scope{TenantID: parts[2], ProjectID: parts[4]}
	if len(parts) == 6 && request.Method == http.MethodPost {
		h.submit(response, request, principal, scope)
		return
	}
	if len(parts) >= 7 {
		id := parts[6]
		switch {
		case len(parts) == 7 && request.Method == http.MethodGet:
			h.get(response, principal, scope, id)
		case len(parts) == 7 && request.Method == http.MethodDelete:
			h.cancel(response, request, principal, scope, id)
		case len(parts) == 8 && parts[7] == "events" && request.Method == http.MethodGet:
			h.events(response, request, principal, scope, id)
		default:
			http.NotFound(response, request)
		}
		return
	}
	http.NotFound(response, request)
}

func (h *HTTPHandler) submit(response http.ResponseWriter, request *http.Request, principal Principal, scope Scope) {
	request.Body = http.MaxBytesReader(response, request.Body, 1<<20)
	var body struct {
		ModelDigest    string `json:"model_digest"`
		InputArtifact  string `json:"input_artifact"`
		Seed           uint64 `json:"seed"`
		DiffusionSteps uint32 `json:"diffusion_steps"`
	}
	if err := decodeStrictJSON(request.Body, &body); err != nil {
		writeError(response, fmt.Errorf("%w: malformed JSON", ErrInvalidRequest))
		return
	}
	job, replayed, err := h.service.Submit(principal, SubmitRequest{
		Scope: scope, IdempotencyKey: request.Header.Get("Idempotency-Key"),
		ModelDigest: body.ModelDigest, InputArtifact: body.InputArtifact,
		Seed: body.Seed, DiffusionSteps: body.DiffusionSteps,
	})
	if err != nil {
		writeError(response, err)
		return
	}
	if !replayed && h.dispatcher != nil {
		job, err = h.dispatcher.Dispatch(request.Context(), job)
		if err != nil {
			writeError(response, err)
			return
		}
	}
	response.Header().Set("Content-Type", "application/json")
	response.Header().Set("X-Idempotent-Replay", strconv.FormatBool(replayed))
	response.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(response).Encode(job)
}

func (h *HTTPHandler) complete(response http.ResponseWriter, request *http.Request, pathJobID string) {
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		writeError(response, fmt.Errorf("%w: Content-Type must be application/json", ErrInvalidRequest))
		return
	}
	request.Body = http.MaxBytesReader(response, request.Body, 1<<20)
	payload, err := io.ReadAll(request.Body)
	if err != nil {
		writeError(response, fmt.Errorf("%w: malformed JSON", ErrInvalidRequest))
		return
	}
	var receipt ResultReceipt
	if err := decodeStrictJSON(bytes.NewReader(payload), &receipt); err != nil {
		writeError(response, fmt.Errorf("%w: malformed JSON", ErrInvalidRequest))
		return
	}
	if receipt.JobID != pathJobID {
		writeError(response, fmt.Errorf("%w: path and receipt job IDs differ", ErrInvalidRequest))
		return
	}
	job, err := h.service.CompleteSignedAttempt(
		request.Context(),
		receipt,
		request.Header.Get("X-Mindclade-Completion-Capability"),
		payload,
		request.Header.Get("X-Mindclade-Completion-Signature"),
	)
	if err != nil {
		writeError(response, err)
		return
	}
	response.Header().Set("Content-Type", "application/json")
	response.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(response).Encode(job)
}

func (h *HTTPHandler) authorizeResultUpload(response http.ResponseWriter, request *http.Request, pathJobID string) {
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		writeError(response, fmt.Errorf("%w: Content-Type must be application/json", ErrInvalidRequest))
		return
	}
	request.Body = http.MaxBytesReader(response, request.Body, 64<<10)
	payload, err := io.ReadAll(request.Body)
	if err != nil {
		writeError(response, fmt.Errorf("%w: malformed JSON", ErrInvalidRequest))
		return
	}
	var uploadRequest ResultUploadRequest
	if err := decodeStrictJSON(bytes.NewReader(payload), &uploadRequest); err != nil {
		writeError(response, fmt.Errorf("%w: malformed JSON", ErrInvalidRequest))
		return
	}
	if uploadRequest.JobID != pathJobID {
		writeError(response, fmt.Errorf("%w: path and upload job IDs differ", ErrInvalidRequest))
		return
	}
	authorization, err := h.service.AuthorizeResultUpload(
		uploadRequest,
		request.Header.Get("X-Mindclade-Completion-Capability"),
		payload,
		request.Header.Get("X-Mindclade-Completion-Signature"),
	)
	if err != nil {
		writeError(response, err)
		return
	}
	response.Header().Set("Content-Type", "application/json")
	response.Header().Set("Cache-Control", "no-store")
	response.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(response).Encode(authorization)
}

func (h *HTTPHandler) get(response http.ResponseWriter, principal Principal, scope Scope, id string) {
	job, err := h.service.Get(principal, scope, id)
	if err != nil {
		writeError(response, err)
		return
	}
	response.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(response).Encode(job)
}

func (h *HTTPHandler) cancel(response http.ResponseWriter, request *http.Request, principal Principal, scope Scope, id string) {
	var job Job
	var err error
	if h.dispatcher != nil {
		job, err = h.dispatcher.Cancel(request.Context(), principal, scope, id)
	} else {
		job, err = h.service.Cancel(principal, scope, id)
	}
	if err != nil {
		writeError(response, err)
		return
	}
	response.Header().Set("Content-Type", "application/json")
	response.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(response).Encode(job)
}

func (h *HTTPHandler) events(response http.ResponseWriter, request *http.Request, principal Principal, scope Scope, id string) {
	after, _ := strconv.ParseInt(request.Header.Get("Last-Event-ID"), 10, 64)
	events, err := h.service.Events(principal, scope, id, after)
	if err != nil {
		writeError(response, err)
		return
	}
	response.Header().Set("Content-Type", "text/event-stream")
	response.Header().Set("Cache-Control", "no-store")
	for _, event := range events {
		payload, _ := json.Marshal(event)
		_, _ = fmt.Fprintf(response, "id: %d\nevent: job-state\ndata: %s\n\n", event.Sequence, payload)
	}
}

func writeError(response http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, ErrForbidden), errors.Is(err, ErrInvalidCapability):
		status = http.StatusForbidden
	case errors.Is(err, ErrArtifactForbidden):
		status = http.StatusForbidden
	case errors.Is(err, ErrInvalidRequest):
		status = http.StatusBadRequest
	case errors.Is(err, ErrNotFound):
		status = http.StatusNotFound
	case errors.Is(err, ErrIdempotencyConflict), errors.Is(err, ErrInvalidTransition), errors.Is(err, ErrStaleFence):
		status = http.StatusConflict
	case errors.Is(err, ErrBudgetExceeded):
		status = http.StatusTooManyRequests
	case errors.Is(err, ErrCapacityExceeded):
		status = http.StatusServiceUnavailable
	case errors.Is(err, ErrAttemptLaunch), errors.Is(err, ErrAttemptCancellation):
		status = http.StatusServiceUnavailable
	case errors.Is(err, ErrResultPublication):
		status = http.StatusServiceUnavailable
	}
	response.Header().Set("Content-Type", "application/json")
	response.WriteHeader(status)
	_ = json.NewEncoder(response).Encode(map[string]string{"error": err.Error()})
}

func internalJobAction(requestPath string) (string, string, bool) {
	if !strings.HasPrefix(requestPath, "/") {
		return "", "", false
	}
	parts := strings.Split(strings.TrimPrefix(requestPath, "/"), "/")
	if len(parts) != 5 || parts[0] != "internal" || parts[1] != "v1alpha1" || parts[2] != "jobs" ||
		(parts[4] != "complete" && parts[4] != "result-upload-capability") {
		return "", "", false
	}
	if !jobIDPattern.MatchString(parts[3]) {
		return "", "", false
	}
	return parts[3], parts[4], true
}

func decodeStrictJSON(body io.Reader, destination any) error {
	decoder := json.NewDecoder(body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}
