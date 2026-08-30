package controlplane

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	runtimegateway "github.com/mindclade/mindclade-internal-monorepo/services/runtime_gateway"
)

const (
	servingRevisionDigest = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
	requestFingerprint    = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
	samplerDigest         = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
)

func testAttemptProvenance() AttemptProvenance {
	return AttemptProvenance{
		ServingRevisionDigest: servingRevisionDigest,
		ExecutionMode:         "eager",
		SamplerDigest:         samplerDigest,
	}
}

func resultReceipt(attempt AttemptLease) ResultReceipt {
	fence := attempt.Job.FencingToken
	return ResultReceipt{
		JobID:                 attempt.Job.ID,
		TenantID:              attempt.Job.Scope.TenantID,
		ProjectID:             attempt.Job.Scope.ProjectID,
		ModelDigest:           attempt.Job.ModelDigest,
		InputDigest:           attempt.Job.InputArtifact,
		ServingRevisionDigest: attempt.Provenance.ServingRevisionDigest,
		ResultDigest:          resultDigest,
		ResultSizeBytes:       4096,
		ResultManifestPath:    fmt.Sprintf("/var/run/mindclade-results/output/result.fence-%d.receipt.json", fence),
		FencingToken:          fence,
		RequestFingerprint:    requestFingerprint,
		SelectedCandidateID:   "candidate-0000",
		ExecutionMode:         attempt.Provenance.ExecutionMode,
		SamplerDigest:         attempt.Provenance.SamplerDigest,
		SchemaVersion:         "v1alpha1",
	}
}

func leaseFixture(t *testing.T) (*Service, Principal, SubmitRequest, AttemptLease) {
	t.Helper()
	service, principal, request := fixture()
	job, _, err := service.Submit(principal, request)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Admit(request.Scope, job.ID); err != nil {
		t.Fatal(err)
	}
	attempt, err := service.LeaseAttempt(request.Scope, job.ID, testAttemptProvenance())
	if err != nil {
		t.Fatal(err)
	}
	return service, principal, request, attempt
}

func TestCompletionCapabilityRejectsWrongProvenanceAndReplay(t *testing.T) {
	service, principal, request, attempt := leaseFixture(t)
	receipt := resultReceipt(attempt)

	wrongCapability := attempt.CompletionCapability[:len(attempt.CompletionCapability)-1] + "A"
	if wrongCapability == attempt.CompletionCapability {
		wrongCapability = attempt.CompletionCapability[:len(attempt.CompletionCapability)-1] + "B"
	}
	if _, err := completeSigned(t, service, attempt, receipt, wrongCapability); !errors.Is(err, ErrInvalidCapability) {
		t.Fatalf("wrong capability error = %v", err)
	}

	wrongProvenance := receipt
	wrongProvenance.ServingRevisionDigest = modelDigest
	if _, err := completeSigned(t, service, attempt, wrongProvenance, attempt.CompletionCapability); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("wrong provenance error = %v", err)
	}

	completed, err := completeSigned(t, service, attempt, receipt, attempt.CompletionCapability)
	if err != nil || completed.State != StateSucceeded || completed.ResultArtifact != resultDigest ||
		completed.ResultArtifactURI != resultArtifactURI(resultDigest) {
		t.Fatalf("complete = (%+v, %v)", completed, err)
	}
	if _, err := completeSigned(t, service, attempt, receipt, attempt.CompletionCapability); !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("replay error = %v", err)
	}
	stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
	if err != nil || stored.State != StateSucceeded {
		t.Fatalf("stored job = (%+v, %v)", stored, err)
	}
}

func TestCompletionCapabilityIsFenceAndScopeBoundAndNotSerialized(t *testing.T) {
	service, principal, request, attempt := leaseFixture(t)
	receipt := resultReceipt(attempt)
	receipt.FencingToken++
	receipt.ResultManifestPath = fmt.Sprintf("/var/run/mindclade-results/output/result.fence-%d.receipt.json", receipt.FencingToken)
	if _, err := completeSigned(t, service, attempt, receipt, attempt.CompletionCapability); !errors.Is(err, ErrStaleFence) {
		t.Fatalf("stale fence error = %v", err)
	}

	encoded, err := json.Marshal(attempt)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), attempt.CompletionCapability) ||
		strings.Contains(string(encoded), attempt.CompletionSigningPrivateKey) ||
		strings.Contains(string(encoded), "completion_capability") || strings.Contains(string(encoded), "completion_signing_private_key") {
		t.Fatalf("attempt JSON exposed completion authority: %s", encoded)
	}

	secondRequest := request
	secondRequest.IdempotencyKey = "request-second-job"
	second, _, err := service.Submit(principal, secondRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Admit(secondRequest.Scope, second.ID); err != nil {
		t.Fatal(err)
	}
	secondAttempt, err := service.LeaseAttempt(secondRequest.Scope, second.ID, testAttemptProvenance())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := completeSigned(t, service, secondAttempt, resultReceipt(secondAttempt), attempt.CompletionCapability); !errors.Is(err, ErrInvalidCapability) {
		t.Fatalf("cross-job capability error = %v", err)
	}
	crossScope := resultReceipt(attempt)
	crossScope.TenantID = "tenant-b"
	if _, err := completeSigned(t, service, attempt, crossScope, attempt.CompletionCapability); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-scope capability error = %v", err)
	}
}

func TestBoundedEventWindowRetainsMonotonicSequences(t *testing.T) {
	_, principal, request := fixture()
	store := deterministicMemoryStore(1, 1)
	catalog := NewMemoryArtifactCatalog()
	if err := catalog.Grant(request.Scope, modelDigest, ArtifactModel); err != nil {
		t.Fatal(err)
	}
	if err := catalog.Grant(request.Scope, inputDigest, ArtifactInput); err != nil {
		t.Fatal(err)
	}
	service := NewServiceWithClockAndResultPublication(
		store,
		BudgetPolicy{MaxGPUMillisecondsPerJob: 10_000},
		catalog,
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
		testResultPublication(),
	)
	job, _, err := service.Submit(principal, request)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Admit(request.Scope, job.ID); err != nil {
		t.Fatal(err)
	}
	attempt, err := service.LeaseAttempt(request.Scope, job.ID, testAttemptProvenance())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := completeSigned(t, service, attempt, resultReceipt(attempt), attempt.CompletionCapability); err != nil {
		t.Fatal(err)
	}
	events, err := service.Events(principal, request.Scope, job.ID, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Sequence != 4 || events[0].State != StateSucceeded {
		t.Fatalf("bounded events = %+v", events)
	}
}

type captureLauncher struct {
	attempts  []AttemptLease
	err       error
	cancelled []Job
	cancelErr error
}

type rejectingResultVerifier struct{}

func (rejectingResultVerifier) VerifyCommitted(context.Context, Scope, string, uint64) error {
	return ErrResultPublication
}

func TestCompletionWaitsForIndependentArtifactVerification(t *testing.T) {
	service, principal, request, attempt := leaseFixture(t)
	service.results.verifier = rejectingResultVerifier{}
	receipt := resultReceipt(attempt)
	if _, err := completeSigned(t, service, attempt, receipt, attempt.CompletionCapability); !errors.Is(err, ErrResultPublication) {
		t.Fatalf("unverified completion error = %v", err)
	}
	stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
	if err != nil || stored.State != StateRunning {
		t.Fatalf("unverified result changed state = (%+v, %v)", stored, err)
	}
	service.results.verifier = acceptingResultVerifier{}
	completed, err := completeSigned(t, service, attempt, receipt, attempt.CompletionCapability)
	if err != nil || completed.State != StateSucceeded || completed.ResultArtifactURI != resultArtifactURI(receipt.ResultDigest) {
		t.Fatalf("verified completion = (%+v, %v)", completed, err)
	}
}

func (l *captureLauncher) Cancel(_ context.Context, job Job) error {
	l.cancelled = append(l.cancelled, job)
	return l.cancelErr
}

func (l *captureLauncher) Launch(_ context.Context, attempt AttemptLease) error {
	l.attempts = append(l.attempts, attempt)
	return l.err
}

func TestDispatcherLaunchFailureIsTerminal(t *testing.T) {
	service, principal, request := fixture()
	job, _, err := service.Submit(principal, request)
	if err != nil {
		t.Fatal(err)
	}
	launcher := &captureLauncher{err: errors.New("scheduler unavailable")}
	dispatcher, err := NewDispatcher(service, launcher, testAttemptProvenance())
	if err != nil {
		t.Fatal(err)
	}
	failed, err := dispatcher.Dispatch(context.Background(), job)
	if !errors.Is(err, ErrAttemptLaunch) {
		t.Fatalf("dispatch error = %v", err)
	}
	if failed.State != StateFailed || failed.FailureCode != "attempt_launch_failed" ||
		len(launcher.attempts) != 1 || len(launcher.cancelled) != 1 {
		t.Fatalf("failed launch = (%+v, attempts=%d, cancellations=%d)", failed, len(launcher.attempts), len(launcher.cancelled))
	}
	stored, getErr := service.Get(principal, request.Scope, job.ID)
	if getErr != nil || stored.State != StateFailed {
		t.Fatalf("stored failed job = (%+v, %v)", stored, getErr)
	}
}

func TestDispatcherLaunchFailureKeepsBudgetUntilDeletionAcknowledged(t *testing.T) {
	service, principal, request := fixture()
	job, _, err := service.Submit(principal, request)
	if err != nil {
		t.Fatal(err)
	}
	launcher := &captureLauncher{
		err:       errors.New("scheduler unavailable"),
		cancelErr: errors.New("JobSet absence not observed"),
	}
	dispatcher, err := NewDispatcher(service, launcher, testAttemptProvenance())
	if err != nil {
		t.Fatal(err)
	}
	running, err := dispatcher.Dispatch(context.Background(), job)
	if !errors.Is(err, ErrAttemptLaunch) || !errors.Is(err, ErrAttemptCancellation) || running.State != StateRunning {
		t.Fatalf("unacknowledged launch rollback = (%+v, %v)", running, err)
	}
	stored, getErr := service.Get(principal, request.Scope, job.ID)
	if getErr != nil || stored.State != StateRunning {
		t.Fatalf("unacknowledged rollback released budget = (%+v, %v)", stored, getErr)
	}
}

func TestDispatcherAcknowledgesSchedulerDeletionBeforeCancellation(t *testing.T) {
	service, principal, request := fixture()
	job, _, err := service.Submit(principal, request)
	if err != nil {
		t.Fatal(err)
	}
	launcher := &captureLauncher{}
	dispatcher, err := NewDispatcher(service, launcher, testAttemptProvenance())
	if err != nil {
		t.Fatal(err)
	}
	running, err := dispatcher.Dispatch(context.Background(), job)
	if err != nil {
		t.Fatal(err)
	}
	launcher.cancelErr = errors.New("deletion not observed")
	if _, err := dispatcher.Cancel(context.Background(), principal, request.Scope, running.ID); !errors.Is(err, ErrAttemptCancellation) {
		t.Fatalf("unacknowledged cancel error = %v", err)
	}
	stored, err := service.Get(principal, request.Scope, running.ID)
	if err != nil || stored.State != StateRunning {
		t.Fatalf("unacknowledged cancel changed state = (%+v, %v)", stored, err)
	}
	launcher.cancelErr = nil
	cancelled, err := dispatcher.Cancel(context.Background(), principal, request.Scope, running.ID)
	if err != nil || cancelled.State != StateCancelled || len(launcher.cancelled) != 2 {
		t.Fatalf("acknowledged cancel = (%+v, %v), calls=%d", cancelled, err, len(launcher.cancelled))
	}
}

func TestHTTPSubmitLaunchCompleteAndIdempotentReplay(t *testing.T) {
	service, _, _ := fixture()
	identity, err := NewInternalIdentityVerifier(testInternalSecret)
	if err != nil {
		t.Fatal(err)
	}
	launcher := &captureLauncher{}
	dispatcher, err := NewDispatcher(service, launcher, testAttemptProvenance())
	if err != nil {
		t.Fatal(err)
	}
	handler := NewHTTPHandlerWithDispatcher(service, identity, dispatcher)
	requestBody := `{"model_digest":"` + modelDigest + `","input_artifact":"` + inputDigest + `","seed":1,"diffusion_steps":8}`

	submit := signedSubmitRequest(t, requestBody)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, submit)
	if response.Code != http.StatusAccepted || len(launcher.attempts) != 1 {
		t.Fatalf("submit status = %d, launches = %d, body = %s", response.Code, len(launcher.attempts), response.Body.String())
	}
	var running Job
	if err := json.Unmarshal(response.Body.Bytes(), &running); err != nil {
		t.Fatal(err)
	}
	if running.State != StateRunning || running.FencingToken != 1 {
		t.Fatalf("running job = %+v", running)
	}

	attempt := launcher.attempts[0]
	uploadResponse := sendUploadAuthorization(t, handler, attempt, resultReceipt(attempt))
	if uploadResponse.Code != http.StatusCreated {
		t.Fatalf("upload authorization status = %d, body = %s", uploadResponse.Code, uploadResponse.Body.String())
	}
	receiptBody, err := json.Marshal(resultReceipt(attempt))
	if err != nil {
		t.Fatal(err)
	}
	completion := httptest.NewRequest(
		http.MethodPost,
		"/internal/v1alpha1/jobs/"+attempt.Job.ID+"/complete",
		strings.NewReader(string(receiptBody)),
	)
	completion.Header.Set("Content-Type", "application/json")
	completion.Header.Set("X-Mindclade-Completion-Capability", attempt.CompletionCapability)
	completion.Header.Set("X-Mindclade-Completion-Signature", signCompletionPayload(t, attempt, receiptBody))
	completionResponse := httptest.NewRecorder()
	handler.ServeHTTP(completionResponse, completion)
	if completionResponse.Code != http.StatusOK {
		t.Fatalf("completion status = %d, body = %s", completionResponse.Code, completionResponse.Body.String())
	}

	replay := signedSubmitRequest(t, requestBody)
	replayResponse := httptest.NewRecorder()
	handler.ServeHTTP(replayResponse, replay)
	if replayResponse.Code != http.StatusAccepted || replayResponse.Header().Get("X-Idempotent-Replay") != "true" {
		t.Fatalf("replay status = %d, header = %q, body = %s", replayResponse.Code, replayResponse.Header().Get("X-Idempotent-Replay"), replayResponse.Body.String())
	}
	if len(launcher.attempts) != 1 {
		t.Fatalf("idempotent replay launched %d attempts", len(launcher.attempts))
	}
}

func TestHTTPCompletionStrictlyRejectsUnknownFieldsWithoutGatewayIdentity(t *testing.T) {
	service, _, _, attempt := leaseFixture(t)
	handler := NewHTTPHandler(service, nil)
	receiptBody, err := json.Marshal(resultReceipt(attempt))
	if err != nil {
		t.Fatal(err)
	}
	malformed := strings.TrimSuffix(string(receiptBody), "}") + `,"untrusted":true}`
	request := httptest.NewRequest(
		http.MethodPost,
		"/internal/v1alpha1/jobs/"+attempt.Job.ID+"/complete",
		strings.NewReader(malformed),
	)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Mindclade-Completion-Capability", attempt.CompletionCapability)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("unknown field status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestHTTPCompletionRejectsWrongCapabilityStaleFenceAndReplay(t *testing.T) {
	service, _, _, attempt := leaseFixture(t)
	handler := NewHTTPHandler(service, nil)
	receipt := resultReceipt(attempt)
	payload, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	tamperedRequest := httptest.NewRequest(
		http.MethodPost,
		"/internal/v1alpha1/jobs/"+attempt.Job.ID+"/complete",
		bytes.NewReader(append(payload, ' ')),
	)
	tamperedRequest.Header.Set("Content-Type", "application/json")
	tamperedRequest.Header.Set("X-Mindclade-Completion-Capability", attempt.CompletionCapability)
	tamperedRequest.Header.Set("X-Mindclade-Completion-Signature", signCompletionPayload(t, attempt, payload))
	tamperedResponse := httptest.NewRecorder()
	handler.ServeHTTP(tamperedResponse, tamperedRequest)
	if tamperedResponse.Code != http.StatusForbidden {
		t.Fatalf("tampered payload status = %d, body = %s", tamperedResponse.Code, tamperedResponse.Body.String())
	}

	wrongCapability := attempt.CompletionCapability[:len(attempt.CompletionCapability)-1] + "A"
	if wrongCapability == attempt.CompletionCapability {
		wrongCapability = attempt.CompletionCapability[:len(attempt.CompletionCapability)-1] + "B"
	}
	if response := sendCompletion(t, handler, receipt, attempt, wrongCapability); response.Code != http.StatusForbidden {
		t.Fatalf("wrong capability status = %d, body = %s", response.Code, response.Body.String())
	}
	stale := receipt
	stale.FencingToken++
	stale.ResultManifestPath = fmt.Sprintf("/var/run/mindclade-results/output/result.fence-%d.receipt.json", stale.FencingToken)
	if response := sendCompletion(t, handler, stale, attempt, attempt.CompletionCapability); response.Code != http.StatusConflict {
		t.Fatalf("stale fence status = %d, body = %s", response.Code, response.Body.String())
	}
	uploadResponse := sendUploadAuthorization(t, handler, attempt, receipt)
	if uploadResponse.Code != http.StatusCreated {
		t.Fatalf("upload authorization status = %d, body = %s", uploadResponse.Code, uploadResponse.Body.String())
	}
	var firstAuthorization ResultUploadAuthorization
	if err := json.Unmarshal(uploadResponse.Body.Bytes(), &firstAuthorization); err != nil {
		t.Fatal(err)
	}
	retriedUpload := sendUploadAuthorization(t, handler, attempt, receipt)
	if retriedUpload.Code != http.StatusCreated {
		t.Fatalf("upload authorization retry status = %d, body = %s", retriedUpload.Code, retriedUpload.Body.String())
	}
	var retryAuthorization ResultUploadAuthorization
	if err := json.Unmarshal(retriedUpload.Body.Bytes(), &retryAuthorization); err != nil {
		t.Fatal(err)
	}
	if retryAuthorization.SessionID == firstAuthorization.SessionID ||
		retryAuthorization.UploadCapability == firstAuthorization.UploadCapability {
		t.Fatal("exact upload retry did not receive fresh bounded authority")
	}
	differentResult := receipt
	differentResult.ResultDigest = "sha256:" + strings.Repeat("9", 64)
	if response := sendUploadAuthorization(t, handler, attempt, differentResult); response.Code != http.StatusConflict {
		t.Fatalf("changed upload identity status = %d, body = %s", response.Code, response.Body.String())
	}
	if response := sendCompletion(t, handler, receipt, attempt, attempt.CompletionCapability); response.Code != http.StatusOK {
		t.Fatalf("completion status = %d, body = %s", response.Code, response.Body.String())
	}
	if response := sendCompletion(t, handler, receipt, attempt, attempt.CompletionCapability); response.Code != http.StatusConflict {
		t.Fatalf("replay status = %d, body = %s", response.Code, response.Body.String())
	}
}

func sendCompletion(t *testing.T, handler http.Handler, receipt ResultReceipt, attempt AttemptLease, capability string) *httptest.ResponseRecorder {
	t.Helper()
	payload, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/internal/v1alpha1/jobs/"+receipt.JobID+"/complete",
		strings.NewReader(string(payload)),
	)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Mindclade-Completion-Capability", capability)
	request.Header.Set("X-Mindclade-Completion-Signature", signCompletionPayload(t, attempt, payload))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func sendUploadAuthorization(t *testing.T, handler http.Handler, attempt AttemptLease, receipt ResultReceipt) *httptest.ResponseRecorder {
	t.Helper()
	uploadRequest := ResultUploadRequest{
		JobID: receipt.JobID, TenantID: receipt.TenantID, ProjectID: receipt.ProjectID,
		ResultDigest: receipt.ResultDigest, ResultSizeBytes: receipt.ResultSizeBytes,
		FencingToken: receipt.FencingToken, SchemaVersion: "v1alpha1",
	}
	payload, err := json.Marshal(uploadRequest)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/internal/v1alpha1/jobs/"+receipt.JobID+"/result-upload-capability",
		bytes.NewReader(payload),
	)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Mindclade-Completion-Capability", attempt.CompletionCapability)
	request.Header.Set("X-Mindclade-Completion-Signature", signCompletionPayload(t, attempt, payload))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func signCompletionPayload(t *testing.T, attempt AttemptLease, payload []byte) string {
	t.Helper()
	seed, err := base64.StdEncoding.DecodeString(attempt.CompletionSigningPrivateKey)
	if err != nil || len(seed) != ed25519.SeedSize {
		t.Fatalf("completion signing key is invalid: %v", err)
	}
	return base64.StdEncoding.EncodeToString(ed25519.Sign(ed25519.NewKeyFromSeed(seed), payload))
}

func completeSigned(t *testing.T, service *Service, attempt AttemptLease, receipt ResultReceipt, capability string) (Job, error) {
	t.Helper()
	uploadRequest := ResultUploadRequest{
		JobID: receipt.JobID, TenantID: receipt.TenantID, ProjectID: receipt.ProjectID,
		ResultDigest: receipt.ResultDigest, ResultSizeBytes: receipt.ResultSizeBytes,
		FencingToken: receipt.FencingToken, SchemaVersion: "v1alpha1",
	}
	uploadPayload, err := json.Marshal(uploadRequest)
	if err != nil {
		t.Fatal(err)
	}
	_, authorizationErr := service.AuthorizeResultUpload(
		uploadRequest,
		capability,
		uploadPayload,
		signCompletionPayload(t, attempt, uploadPayload),
	)
	if authorizationErr != nil && !errors.Is(authorizationErr, ErrInvalidTransition) {
		return Job{}, authorizationErr
	}
	payload, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	return service.CompleteSignedAttempt(
		context.Background(), receipt, capability, payload, signCompletionPayload(t, attempt, payload),
	)
}

func signedSubmitRequest(t *testing.T, body string) *http.Request {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, "/v1alpha1/tenants/tenant-a/projects/project-a/inference-jobs", strings.NewReader(body))
	request.Header.Set("Idempotency-Key", "request-http-dispatch-1")
	signer, err := runtimegateway.NewInternalIdentitySigner(testInternalSecret)
	if err != nil {
		t.Fatal(err)
	}
	if err := signer.Sign(request, runtimegateway.Claims{
		Subject: "user-1", TenantID: "tenant-a", Projects: map[string]bool{"project-a": true}, ExpiresAt: time.Now().Add(time.Hour),
	}, "tenant-a", "project-a"); err != nil {
		t.Fatal(err)
	}
	return request
}
