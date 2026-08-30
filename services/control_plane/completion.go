package controlplane

import (
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"path"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	completionCapabilityBytes = 32
	maxResultSizeBytes        = int64(8 << 30)
)

var completionCapabilityPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{43}$`)

// AttemptProvenance pins execution facts chosen before an attempt is launched.
// A worker receipt must repeat these values exactly before its result is accepted.
type AttemptProvenance struct {
	ServingRevisionDigest string `json:"serving_revision_digest"`
	ExecutionMode         string `json:"execution_mode"`
	SamplerDigest         string `json:"sampler_digest"`
}

func (p AttemptProvenance) Validate() error {
	if !digestPattern.MatchString(p.ServingRevisionDigest) || !digestPattern.MatchString(p.SamplerDigest) {
		return fmt.Errorf("%w: attempt provenance must use immutable sha256 digests", ErrInvalidRequest)
	}
	if p.ExecutionMode != "eager" {
		return fmt.Errorf("%w: only the qualified eager execution mode is supported", ErrInvalidRequest)
	}
	return nil
}

// AttemptLease carries the one-time capability to a trusted launcher. The
// capability is deliberately excluded from JSON so it cannot enter job APIs.
type AttemptLease struct {
	Job                         Job               `json:"job"`
	CompletionCapability        string            `json:"-"`
	CompletionSigningPrivateKey string            `json:"-"`
	Provenance                  AttemptProvenance `json:"provenance"`
}

// ResultUploadRequest asks for one upload capability after result bytes and
// their immutable identity are known. It carries no tensor values.
type ResultUploadRequest struct {
	JobID           string `json:"job_id"`
	TenantID        string `json:"tenant_id"`
	ProjectID       string `json:"project_id"`
	ResultDigest    string `json:"result_digest"`
	ResultSizeBytes int64  `json:"result_size_bytes"`
	FencingToken    int64  `json:"fencing_token"`
	SchemaVersion   string `json:"schema_version"`
}

func (r ResultUploadRequest) Scope() Scope {
	return Scope{TenantID: r.TenantID, ProjectID: r.ProjectID}
}

func (r ResultUploadRequest) Validate() error {
	if err := r.Scope().Validate(); err != nil {
		return err
	}
	if !jobIDPattern.MatchString(r.JobID) || !digestPattern.MatchString(r.ResultDigest) ||
		r.ResultSizeBytes < 1 || r.ResultSizeBytes > maxResultSizeBytes || r.FencingToken < 1 ||
		r.SchemaVersion != "v1alpha1" {
		return fmt.Errorf("%w: invalid result upload request", ErrInvalidRequest)
	}
	return nil
}

// ResultUploadAuthorization is a one-use bearer response. Callers must never
// log or persist it beyond the active transfer.
type ResultUploadAuthorization struct {
	UploadCapability string `json:"upload_capability"`
	SessionID        string `json:"session_id"`
}

// ResultReceipt mirrors the inference worker's completion contract. It contains
// identities and provenance only; result tensor bytes never cross this boundary.
type ResultReceipt struct {
	JobID                 string `json:"job_id"`
	TenantID              string `json:"tenant_id"`
	ProjectID             string `json:"project_id"`
	ModelDigest           string `json:"model_digest"`
	InputDigest           string `json:"input_digest"`
	ServingRevisionDigest string `json:"serving_revision_digest"`
	ResultDigest          string `json:"result_digest"`
	ResultSizeBytes       int64  `json:"result_size_bytes"`
	ResultManifestPath    string `json:"result_manifest_path"`
	FencingToken          int64  `json:"fencing_token"`
	RequestFingerprint    string `json:"request_fingerprint"`
	SelectedCandidateID   string `json:"selected_candidate_id"`
	ExecutionMode         string `json:"execution_mode"`
	SamplerDigest         string `json:"sampler_digest"`
	SchemaVersion         string `json:"schema_version"`
}

func (r ResultReceipt) Scope() Scope {
	return Scope{TenantID: r.TenantID, ProjectID: r.ProjectID}
}

func (r ResultReceipt) Validate() error {
	if err := r.Scope().Validate(); err != nil {
		return err
	}
	if !jobIDPattern.MatchString(r.JobID) || !identifierPattern.MatchString(r.SelectedCandidateID) {
		return fmt.Errorf("%w: job or candidate identity is invalid", ErrInvalidRequest)
	}
	for _, digest := range []string{
		r.ModelDigest,
		r.InputDigest,
		r.ServingRevisionDigest,
		r.ResultDigest,
		r.RequestFingerprint,
		r.SamplerDigest,
	} {
		if !digestPattern.MatchString(digest) {
			return fmt.Errorf("%w: receipt identities must use immutable sha256 digests", ErrInvalidRequest)
		}
	}
	if r.ResultSizeBytes < 1 || r.ResultSizeBytes > maxResultSizeBytes {
		return fmt.Errorf("%w: result_size_bytes must be within 1..%d", ErrInvalidRequest, maxResultSizeBytes)
	}
	if r.FencingToken < 1 {
		return fmt.Errorf("%w: fencing_token must be positive", ErrInvalidRequest)
	}
	if r.SchemaVersion != "v1alpha1" || r.ExecutionMode != "eager" {
		return fmt.Errorf("%w: unsupported receipt schema or execution mode", ErrInvalidRequest)
	}
	if !validResultManifestPath(r.ResultManifestPath, r.FencingToken) {
		return fmt.Errorf("%w: result_manifest_path is not fence-bound", ErrInvalidRequest)
	}
	return nil
}

func validResultManifestPath(value string, fencingToken int64) bool {
	if value == "" || len(value) > 1024 || !utf8.ValidString(value) || strings.Contains(value, "\\") {
		return false
	}
	for _, character := range value {
		if character < 0x20 || character == 0x7f {
			return false
		}
	}
	cleaned := path.Clean(value)
	if cleaned != value || cleaned == "." {
		return false
	}
	for _, segment := range strings.Split(cleaned, "/") {
		if segment == ".." {
			return false
		}
	}
	expected := fmt.Sprintf(
		"/var/run/mindclade-results/output/result.fence-%d.receipt.json",
		fencingToken,
	)
	return cleaned == expected
}

// CompleteAttempt performs capability verification and the terminal transition
// under one lock. This is the replay-prevention boundary for worker receipts.
func (s *MemoryStore) AuthorizeResultUpload(
	request ResultUploadRequest,
	capability string,
	payload []byte,
	signature string,
	sessionID string,
) error {
	if err := request.Validate(); err != nil {
		return err
	}
	if !signingKeyIDPattern.MatchString(sessionID) {
		return ErrInvalidRequest
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	key := scopeKey(request.Scope(), request.JobID)
	job, ok := s.jobs[key]
	if !ok {
		return ErrNotFound
	}
	if err := validateAttemptAuthority(
		job,
		request.Scope(),
		request.JobID,
		request.FencingToken,
		capability,
		payload,
		signature,
	); err != nil {
		return err
	}
	if job.resultUploadDigest != "" &&
		(job.resultUploadDigest != request.ResultDigest ||
			job.resultUploadSizeBytes != uint64(request.ResultSizeBytes)) {
		return fmt.Errorf("%w: result upload identity differs from the fenced attempt", ErrInvalidTransition)
	}
	// An exact retry is intentionally allowed. The prior artifact-proxy session
	// may have been lost with a response or worker restart, so bind the new
	// short-lived session while keeping the immutable digest and size fixed.
	job.resultUploadDigest = request.ResultDigest
	job.resultUploadSizeBytes = uint64(request.ResultSizeBytes)
	job.resultUploadSessionID = sessionID
	s.jobs[key] = job
	return nil
}

func (s *MemoryStore) ValidateSignedCompletion(receipt ResultReceipt, capability string, payload []byte, signature string) error {
	if err := receipt.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[scopeKey(receipt.Scope(), receipt.JobID)]
	if !ok {
		return ErrNotFound
	}
	return validateCompletion(job, receipt, capability, payload, signature)
}

func (s *MemoryStore) CompleteSignedAttempt(receipt ResultReceipt, capability string, payload []byte, signature string, now time.Time) (Job, error) {
	if err := receipt.Validate(); err != nil {
		return Job{}, err
	}
	if !completionCapabilityPattern.MatchString(capability) {
		return Job{}, ErrInvalidCapability
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	scope := receipt.Scope()
	key := scopeKey(scope, receipt.JobID)
	job, ok := s.jobs[key]
	if !ok {
		return Job{}, ErrNotFound
	}
	if err := validateCompletion(job, receipt, capability, payload, signature); err != nil {
		return Job{}, err
	}

	job.State = StateSucceeded
	job.ResultArtifact = receipt.ResultDigest
	job.ResultArtifactURI = resultArtifactURI(receipt.ResultDigest)
	job.UpdatedAt = now
	job.completionBinding = [sha256.Size]byte{}
	job.attemptProvenance = AttemptProvenance{}
	job.completionVerificationKey = [ed25519.PublicKeySize]byte{}
	job.resultUploadDigest = ""
	job.resultUploadSizeBytes = 0
	job.resultUploadSessionID = ""
	s.jobs[key] = job
	s.appendEvent(job, now)
	return cloneJob(job), nil
}

func validateCompletion(job Job, receipt ResultReceipt, capability string, payload []byte, signature string) error {
	if err := validateAttemptAuthority(
		job,
		receipt.Scope(),
		receipt.JobID,
		receipt.FencingToken,
		capability,
		payload,
		signature,
	); err != nil {
		return err
	}
	if receipt.ModelDigest != job.ModelDigest || receipt.InputDigest != job.InputArtifact {
		return fmt.Errorf("%w: receipt artifacts do not match the admitted job", ErrInvalidRequest)
	}
	if receipt.ServingRevisionDigest != job.attemptProvenance.ServingRevisionDigest ||
		receipt.ExecutionMode != job.attemptProvenance.ExecutionMode ||
		receipt.SamplerDigest != job.attemptProvenance.SamplerDigest {
		return fmt.Errorf("%w: receipt provenance does not match the leased attempt", ErrInvalidRequest)
	}
	if job.resultUploadDigest == "" || receipt.ResultDigest != job.resultUploadDigest ||
		uint64(receipt.ResultSizeBytes) != job.resultUploadSizeBytes || job.resultUploadSessionID == "" {
		return fmt.Errorf("%w: result upload was not authorized for this receipt", ErrInvalidTransition)
	}
	return nil
}

func validateAttemptAuthority(
	job Job,
	scope Scope,
	jobID string,
	fencingToken int64,
	capability string,
	payload []byte,
	signature string,
) error {
	if job.State != StateRunning {
		return ErrInvalidTransition
	}
	if fencingToken != job.FencingToken {
		return ErrStaleFence
	}
	if !completionCapabilityPattern.MatchString(capability) {
		return ErrInvalidCapability
	}
	actualBinding := bindCompletionCapability(scope, jobID, fencingToken, capability)
	if subtle.ConstantTimeCompare(job.completionBinding[:], actualBinding[:]) != 1 {
		return ErrInvalidCapability
	}
	signatureBytes, err := base64.StdEncoding.DecodeString(signature)
	if err != nil || len(signatureBytes) != ed25519.SignatureSize ||
		base64.StdEncoding.EncodeToString(signatureBytes) != signature ||
		!ed25519.Verify(ed25519.PublicKey(job.completionVerificationKey[:]), payload, signatureBytes) {
		return ErrInvalidCapability
	}
	return nil
}
