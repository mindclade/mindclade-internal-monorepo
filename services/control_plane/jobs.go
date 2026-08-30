// Package controlplane owns admission and the development inference-job state
// machine behind a durable-store boundary. Its executable MemoryStore is not
// restart durable.
package controlplane

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"sync"
	"time"
)

const (
	jobIDIncarnationBytes = 16
	maximumJobSequence    = 99_999_999
)

var digestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
var identifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)
var idempotencyPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$`)
var jobIDPattern = regexp.MustCompile(`^job-[0-9a-f]{32}-[0-9]{8}$`)

func validateJobID(id string) error {
	if !jobIDPattern.MatchString(id) {
		return fmt.Errorf(
			"%w: job_id must use job-<32 lowercase hexadecimal characters>-<8 decimal digits>",
			ErrInvalidRequest,
		)
	}
	return nil
}

var (
	ErrForbidden           = errors.New("principal is outside the requested tenant or project")
	ErrInvalidRequest      = errors.New("invalid inference request")
	ErrIdempotencyConflict = errors.New("idempotency key reused with different content")
	ErrBudgetExceeded      = errors.New("tenant compute budget exceeded")
	ErrCapacityExceeded    = errors.New("control-plane capacity exceeded")
	ErrArtifactForbidden   = errors.New("artifact is not authorized for the requested scope")
	ErrNotFound            = errors.New("job not found")
	ErrInvalidTransition   = errors.New("invalid job state transition")
	ErrStaleFence          = errors.New("stale attempt fencing token")
	ErrInvalidCapability   = errors.New("invalid completion capability")
)

// Scope is the mandatory isolation boundary for every repository operation.
type Scope struct {
	TenantID  string `json:"tenant_id"`
	ProjectID string `json:"project_id"`
}

func (s Scope) Validate() error {
	if !identifierPattern.MatchString(s.TenantID) || !identifierPattern.MatchString(s.ProjectID) {
		return fmt.Errorf("%w: tenant_id and project_id must use the shared identifier grammar", ErrInvalidRequest)
	}
	return nil
}

// Principal contains identity claims already verified by the edge gateway.
type Principal struct {
	Subject  string
	TenantID string
	Projects map[string]bool
}

func (p Principal) Authorize(scope Scope) error {
	if !identifierPattern.MatchString(p.Subject) || p.TenantID != scope.TenantID || !p.Projects[scope.ProjectID] {
		return ErrForbidden
	}
	return nil
}

// State is the externally visible job lifecycle.
type State string

const (
	StateQueued    State = "queued"
	StateAdmitted  State = "admitted"
	StateRunning   State = "running"
	StateSucceeded State = "succeeded"
	StateFailed    State = "failed"
	StateCancelled State = "cancelled"
)

func (s State) terminal() bool {
	return s == StateSucceeded || s == StateFailed || s == StateCancelled
}

// SubmitRequest is the canonical request recorded for idempotent submission.
type SubmitRequest struct {
	Scope          Scope  `json:"scope"`
	IdempotencyKey string `json:"idempotency_key"`
	ModelDigest    string `json:"model_digest"`
	InputArtifact  string `json:"input_artifact"`
	Seed           uint64 `json:"seed"`
	DiffusionSteps uint32 `json:"diffusion_steps"`
}

func (r SubmitRequest) Validate() error {
	if err := r.Scope.Validate(); err != nil {
		return err
	}
	if !idempotencyPattern.MatchString(r.IdempotencyKey) {
		return fmt.Errorf("%w: idempotency_key must use 8..128 safe characters", ErrInvalidRequest)
	}
	if !digestPattern.MatchString(r.ModelDigest) || !digestPattern.MatchString(r.InputArtifact) {
		return fmt.Errorf("%w: model and input must use immutable sha256 digests", ErrInvalidRequest)
	}
	if r.Seed >= 1<<63 {
		return fmt.Errorf("%w: seed must be within 0..2^63-1", ErrInvalidRequest)
	}
	if r.DiffusionSteps < 2 || r.DiffusionSteps > 128 {
		return fmt.Errorf("%w: diffusion_steps must be within 2..128", ErrInvalidRequest)
	}
	return nil
}

func (r SubmitRequest) fingerprint() string {
	canonical, _ := json.Marshal(struct {
		ModelDigest    string `json:"model_digest"`
		InputArtifact  string `json:"input_artifact"`
		Seed           uint64 `json:"seed"`
		DiffusionSteps uint32 `json:"diffusion_steps"`
	}{r.ModelDigest, r.InputArtifact, r.Seed, r.DiffusionSteps})
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])
}

// Job is an immutable projection returned from the service boundary.
type Job struct {
	ID                        string    `json:"id"`
	Scope                     Scope     `json:"scope"`
	State                     State     `json:"state"`
	ModelDigest               string    `json:"model_digest"`
	InputArtifact             string    `json:"input_artifact"`
	Seed                      uint64    `json:"seed"`
	DiffusionSteps            uint32    `json:"diffusion_steps"`
	ResultArtifact            string    `json:"result_artifact,omitempty"`
	ResultArtifactURI         string    `json:"result_artifact_uri,omitempty"`
	FailureCode               string    `json:"failure_code,omitempty"`
	EstimatedGPUMilliseconds  int64     `json:"estimated_gpu_milliseconds"`
	FencingToken              int64     `json:"fencing_token"`
	CreatedAt                 time.Time `json:"created_at"`
	UpdatedAt                 time.Time `json:"updated_at"`
	fingerprint               string
	completionBinding         [sha256.Size]byte
	attemptProvenance         AttemptProvenance
	completionVerificationKey [ed25519.PublicKeySize]byte
	resultUploadDigest        string
	resultUploadSizeBytes     uint64
	resultUploadSessionID     string
}

// Event is an outbox record; payloads deliberately contain no tensor values.
type Event struct {
	Sequence int64     `json:"sequence"`
	JobID    string    `json:"job_id"`
	Scope    Scope     `json:"scope"`
	State    State     `json:"state"`
	At       time.Time `json:"at"`
}

// Clock keeps state-machine tests deterministic.
type Clock interface{ Now() time.Time }

type systemClock struct{}

func (systemClock) Now() time.Time { return time.Now().UTC() }

// BudgetPolicy gates submissions before they enter the queue.
type BudgetPolicy struct {
	MaxGPUMillisecondsPerJob      int64
	MaxOutstandingGPUMilliseconds int64
	MaxActiveJobsPerTenant        int
}

func (p BudgetPolicy) Estimate(request SubmitRequest) (int64, error) {
	cost := int64(request.DiffusionSteps) * 100
	if p.MaxGPUMillisecondsPerJob > 0 && cost > p.MaxGPUMillisecondsPerJob {
		return 0, ErrBudgetExceeded
	}
	return cost, nil
}

// Store defines the transaction boundary implemented by Postgres in production.
type Store interface {
	CreateOrReplay(SubmitRequest, int64, BudgetPolicy, time.Time) (Job, bool, error)
	Get(Scope, string) (Job, error)
	Transition(Scope, string, State, string, int64, time.Time) (Job, error)
	Lease(Scope, string, AttemptProvenance, string, [ed25519.PublicKeySize]byte, time.Time) (Job, error)
	AuthorizeResultUpload(ResultUploadRequest, string, []byte, string, string) error
	ValidateSignedCompletion(ResultReceipt, string, []byte, string) error
	CompleteSignedAttempt(ResultReceipt, string, []byte, string, time.Time) (Job, error)
	NonterminalJobs() ([]Job, error)
	Events(Scope, string, int64) ([]Event, error)
}

// MemoryStore is a concurrency-safe reference store for local operation and tests.
type MemoryStore struct {
	mu          sync.Mutex
	jobs        map[string]Job
	idempotency map[string]string
	events      []Event
	nextID      int64
	nextEvent   int64
	maxJobs     int
	maxEvents   int
	incarnation string
}

// NewMemoryStore creates a bounded development store whose public job IDs use
// a fresh cryptographic process incarnation. The state remains process-local,
// but a restart cannot deterministically reissue an earlier public identity.
func NewMemoryStore() (*MemoryStore, error) {
	return NewBoundedMemoryStore(10_000, 80_000)
}

// NewBoundedMemoryStore is an executable development store with hard memory caps.
func NewBoundedMemoryStore(maxJobs, maxEvents int) (*MemoryStore, error) {
	return newBoundedMemoryStore(maxJobs, maxEvents, rand.Reader)
}

func newBoundedMemoryStore(maxJobs, maxEvents int, entropy io.Reader) (*MemoryStore, error) {
	if maxJobs < 1 || maxJobs > maximumJobSequence || maxEvents < maxJobs {
		return nil, fmt.Errorf(
			"%w: memory store limits must fit the job ID sequence and allow one event per job",
			ErrInvalidRequest,
		)
	}
	if entropy == nil {
		return nil, fmt.Errorf("%w: job ID entropy source is required", ErrInvalidRequest)
	}
	rawIncarnation := make([]byte, jobIDIncarnationBytes)
	if _, err := io.ReadFull(entropy, rawIncarnation); err != nil {
		return nil, fmt.Errorf("generate job ID incarnation: %w", err)
	}
	return &MemoryStore{
		jobs: map[string]Job{}, idempotency: map[string]string{},
		maxJobs: maxJobs, maxEvents: maxEvents,
		incarnation: hex.EncodeToString(rawIncarnation),
	}, nil
}

func scopeKey(scope Scope, value string) string {
	return fmt.Sprintf("%d:%s%d:%s%d:%s", len(scope.TenantID), scope.TenantID, len(scope.ProjectID), scope.ProjectID, len(value), value)
}

func cloneJob(job Job) Job {
	job.completionBinding = [sha256.Size]byte{}
	job.attemptProvenance = AttemptProvenance{}
	job.completionVerificationKey = [ed25519.PublicKeySize]byte{}
	job.resultUploadDigest = ""
	job.resultUploadSizeBytes = 0
	job.resultUploadSessionID = ""
	return job
}

func (s *MemoryStore) CreateOrReplay(request SubmitRequest, estimate int64, policy BudgetPolicy, now time.Time) (Job, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := scopeKey(request.Scope, request.IdempotencyKey)
	if id, exists := s.idempotency[key]; exists {
		job := s.jobs[scopeKey(request.Scope, id)]
		if job.fingerprint != request.fingerprint() {
			return Job{}, false, ErrIdempotencyConflict
		}
		return cloneJob(job), true, nil
	}
	if len(s.jobs) >= s.maxJobs {
		return Job{}, false, ErrCapacityExceeded
	}
	active := 0
	var outstanding int64
	for _, existing := range s.jobs {
		if existing.Scope.TenantID == request.Scope.TenantID && !existing.State.terminal() {
			active++
			outstanding += existing.EstimatedGPUMilliseconds
		}
	}
	if policy.MaxActiveJobsPerTenant > 0 && active >= policy.MaxActiveJobsPerTenant {
		return Job{}, false, ErrBudgetExceeded
	}
	if policy.MaxOutstandingGPUMilliseconds > 0 && outstanding+estimate > policy.MaxOutstandingGPUMilliseconds {
		return Job{}, false, ErrBudgetExceeded
	}
	s.nextID++
	id := fmt.Sprintf("job-%s-%08d", s.incarnation, s.nextID)
	job := Job{
		ID: id, Scope: request.Scope, State: StateQueued, ModelDigest: request.ModelDigest,
		InputArtifact: request.InputArtifact, Seed: request.Seed, DiffusionSteps: request.DiffusionSteps,
		EstimatedGPUMilliseconds: estimate,
		CreatedAt:                now, UpdatedAt: now, fingerprint: request.fingerprint(),
	}
	s.jobs[scopeKey(request.Scope, id)] = job
	s.idempotency[key] = id
	s.appendEvent(job, now)
	return cloneJob(job), false, nil
}

func (s *MemoryStore) Get(scope Scope, id string) (Job, error) {
	if err := validateJobID(id); err != nil {
		return Job{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[scopeKey(scope, id)]
	if !ok {
		return Job{}, ErrNotFound
	}
	return cloneJob(job), nil
}

func (s *MemoryStore) NonterminalJobs() ([]Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	jobs := make([]Job, 0, len(s.jobs))
	for _, job := range s.jobs {
		if !job.State.terminal() {
			jobs = append(jobs, cloneJob(job))
		}
	}
	sort.Slice(jobs, func(left, right int) bool {
		if jobs[left].Scope.TenantID != jobs[right].Scope.TenantID {
			return jobs[left].Scope.TenantID < jobs[right].Scope.TenantID
		}
		if jobs[left].Scope.ProjectID != jobs[right].Scope.ProjectID {
			return jobs[left].Scope.ProjectID < jobs[right].Scope.ProjectID
		}
		return jobs[left].ID < jobs[right].ID
	})
	return jobs, nil
}

var transitions = map[State]map[State]bool{
	StateQueued:   {StateAdmitted: true, StateCancelled: true},
	StateAdmitted: {StateRunning: true, StateCancelled: true},
	StateRunning:  {StateSucceeded: true, StateFailed: true, StateCancelled: true},
}

func (s *MemoryStore) Transition(scope Scope, id string, next State, result string, fence int64, now time.Time) (Job, error) {
	if err := validateJobID(id); err != nil {
		return Job{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	key := scopeKey(scope, id)
	job, ok := s.jobs[key]
	if !ok {
		return Job{}, ErrNotFound
	}
	if !transitions[job.State][next] {
		return Job{}, ErrInvalidTransition
	}
	if job.State == StateRunning && fence != job.FencingToken {
		return Job{}, ErrStaleFence
	}
	if next == StateSucceeded {
		if !digestPattern.MatchString(result) {
			return Job{}, fmt.Errorf("%w: result must be an immutable sha256 digest", ErrInvalidRequest)
		}
		job.ResultArtifact = result
	}
	if next == StateFailed {
		if !identifierPattern.MatchString(result) {
			return Job{}, fmt.Errorf("%w: failure code must use the shared identifier grammar", ErrInvalidRequest)
		}
		job.FailureCode = result
	}
	job.State = next
	job.UpdatedAt = now
	if next.terminal() {
		job.completionBinding = [sha256.Size]byte{}
		job.attemptProvenance = AttemptProvenance{}
		job.completionVerificationKey = [ed25519.PublicKeySize]byte{}
		job.resultUploadDigest = ""
		job.resultUploadSizeBytes = 0
		job.resultUploadSessionID = ""
	}
	s.jobs[key] = job
	s.appendEvent(job, now)
	return cloneJob(job), nil
}

func (s *MemoryStore) Lease(scope Scope, id string, provenance AttemptProvenance, capability string, verificationKey [ed25519.PublicKeySize]byte, now time.Time) (Job, error) {
	if err := validateJobID(id); err != nil {
		return Job{}, err
	}
	if err := provenance.Validate(); err != nil {
		return Job{}, err
	}
	if !completionCapabilityPattern.MatchString(capability) {
		return Job{}, ErrInvalidCapability
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	key := scopeKey(scope, id)
	job, ok := s.jobs[key]
	if !ok {
		return Job{}, ErrNotFound
	}
	if job.State != StateAdmitted {
		return Job{}, ErrInvalidTransition
	}
	job.FencingToken++
	job.completionBinding = bindCompletionCapability(scope, id, job.FencingToken, capability)
	job.completionVerificationKey = verificationKey
	job.attemptProvenance = provenance
	job.State = StateRunning
	job.UpdatedAt = now
	s.jobs[key] = job
	s.appendEvent(job, now)
	return cloneJob(job), nil
}

func (s *MemoryStore) appendEvent(job Job, now time.Time) {
	if len(s.events) >= s.maxEvents {
		// State histories are small; retaining the newest bounded window keeps
		// the development process safe without pretending to be an audit store.
		copy(s.events, s.events[1:])
		s.events = s.events[:len(s.events)-1]
	}
	s.nextEvent++
	s.events = append(s.events, Event{s.nextEvent, job.ID, job.Scope, job.State, now})
}

func (s *MemoryStore) Events(scope Scope, id string, after int64) ([]Event, error) {
	if err := validateJobID(id); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.jobs[scopeKey(scope, id)]; !ok {
		return nil, ErrNotFound
	}
	var result []Event
	for _, event := range s.events {
		if event.Sequence > after && event.JobID == id && event.Scope == scope {
			result = append(result, event)
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Sequence < result[j].Sequence })
	return result, nil
}

// Service applies authorization, admission, and clock policy around storage.
type Service struct {
	store   Store
	budget  BudgetPolicy
	clock   Clock
	catalog ArtifactCatalog
	results *ResultPublication
}

func NewService(store Store, budget BudgetPolicy, catalog ArtifactCatalog) *Service {
	return &Service{store: store, budget: budget, clock: systemClock{}, catalog: catalog}
}

func NewServiceWithClock(store Store, budget BudgetPolicy, catalog ArtifactCatalog, clock Clock) *Service {
	return &Service{store: store, budget: budget, clock: clock, catalog: catalog}
}

func NewServiceWithResultPublication(store Store, budget BudgetPolicy, catalog ArtifactCatalog, results *ResultPublication) *Service {
	return &Service{store: store, budget: budget, clock: systemClock{}, catalog: catalog, results: results}
}

func NewServiceWithClockAndResultPublication(
	store Store,
	budget BudgetPolicy,
	catalog ArtifactCatalog,
	clock Clock,
	results *ResultPublication,
) *Service {
	return &Service{store: store, budget: budget, clock: clock, catalog: catalog, results: results}
}

func (s *Service) Submit(principal Principal, request SubmitRequest) (Job, bool, error) {
	if err := principal.Authorize(request.Scope); err != nil {
		return Job{}, false, err
	}
	if err := request.Validate(); err != nil {
		return Job{}, false, err
	}
	estimate, err := s.budget.Estimate(request)
	if err != nil {
		return Job{}, false, err
	}
	if s.catalog == nil || !s.catalog.Owns(request.Scope, request.ModelDigest, ArtifactModel) || !s.catalog.Owns(request.Scope, request.InputArtifact, ArtifactInput) {
		return Job{}, false, ErrArtifactForbidden
	}
	return s.store.CreateOrReplay(request, estimate, s.budget, s.clock.Now())
}

func (s *Service) Get(principal Principal, scope Scope, id string) (Job, error) {
	if err := principal.Authorize(scope); err != nil {
		return Job{}, err
	}
	if err := validateJobID(id); err != nil {
		return Job{}, err
	}
	return s.store.Get(scope, id)
}

// nonterminalJobs is an internal reconciliation view. It deliberately exposes
// no capabilities or verification keys because stores return cloned jobs.
func (s *Service) nonterminalJobs() ([]Job, error) {
	if s == nil || s.store == nil {
		return nil, ErrInvalidRequest
	}
	return s.store.NonterminalJobs()
}

func (s *Service) Cancel(principal Principal, scope Scope, id string) (Job, error) {
	if err := principal.Authorize(scope); err != nil {
		return Job{}, err
	}
	if err := validateJobID(id); err != nil {
		return Job{}, err
	}
	job, err := s.store.Get(scope, id)
	if err != nil {
		return Job{}, err
	}
	if job.State.terminal() {
		return Job{}, ErrInvalidTransition
	}
	return s.store.Transition(scope, id, StateCancelled, "", job.FencingToken, s.clock.Now())
}

func (s *Service) Admit(scope Scope, id string) (Job, error) {
	if err := scope.Validate(); err != nil {
		return Job{}, ErrInvalidRequest
	}
	if err := validateJobID(id); err != nil {
		return Job{}, err
	}
	return s.store.Transition(scope, id, StateAdmitted, "", 0, s.clock.Now())
}

func (s *Service) LeaseAttempt(scope Scope, id string, provenance AttemptProvenance) (AttemptLease, error) {
	if err := scope.Validate(); err != nil {
		return AttemptLease{}, ErrInvalidRequest
	}
	if err := validateJobID(id); err != nil {
		return AttemptLease{}, err
	}
	if err := provenance.Validate(); err != nil {
		return AttemptLease{}, err
	}
	secret := make([]byte, completionCapabilityBytes)
	if _, err := rand.Read(secret); err != nil {
		return AttemptLease{}, fmt.Errorf("generate completion capability: %w", err)
	}
	capability := base64.RawURLEncoding.EncodeToString(secret)
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return AttemptLease{}, fmt.Errorf("generate completion signing key: %w", err)
	}
	var verificationKey [ed25519.PublicKeySize]byte
	copy(verificationKey[:], publicKey)
	job, err := s.store.Lease(scope, id, provenance, capability, verificationKey, s.clock.Now())
	if err != nil {
		return AttemptLease{}, err
	}
	return AttemptLease{
		Job: job, CompletionCapability: capability,
		CompletionSigningPrivateKey: base64.StdEncoding.EncodeToString(privateKey.Seed()),
		Provenance:                  provenance,
	}, nil
}

func (s *Service) AuthorizeResultUpload(request ResultUploadRequest, capability string, payload []byte, signature string) (ResultUploadAuthorization, error) {
	if err := request.Validate(); err != nil {
		return ResultUploadAuthorization{}, err
	}
	if s.results == nil {
		return ResultUploadAuthorization{}, ErrResultPublication
	}
	uploadCapability, sessionID, err := s.results.issuer.IssueUpload(
		request.Scope(), request.ResultDigest, uint64(request.ResultSizeBytes),
	)
	if err != nil {
		return ResultUploadAuthorization{}, err
	}
	if err := s.store.AuthorizeResultUpload(
		request, capability, payload, signature, sessionID,
	); err != nil {
		return ResultUploadAuthorization{}, err
	}
	return ResultUploadAuthorization{UploadCapability: uploadCapability, SessionID: sessionID}, nil
}

func (s *Service) CompleteSignedAttempt(ctx context.Context, receipt ResultReceipt, capability string, payload []byte, signature string) (Job, error) {
	if err := receipt.Validate(); err != nil {
		return Job{}, err
	}
	if s.results == nil {
		return Job{}, ErrResultPublication
	}
	if err := s.store.ValidateSignedCompletion(receipt, capability, payload, signature); err != nil {
		return Job{}, err
	}
	if err := s.results.verifier.VerifyCommitted(
		ctx, receipt.Scope(), receipt.ResultDigest, uint64(receipt.ResultSizeBytes),
	); err != nil {
		return Job{}, err
	}
	return s.store.CompleteSignedAttempt(receipt, capability, payload, signature, s.clock.Now())
}

func (s *Service) FailAttempt(scope Scope, id string, fencingToken int64, failureCode string) (Job, error) {
	if err := scope.Validate(); err != nil || !identifierPattern.MatchString(failureCode) {
		return Job{}, ErrInvalidRequest
	}
	if err := validateJobID(id); err != nil {
		return Job{}, err
	}
	return s.store.Transition(scope, id, StateFailed, failureCode, fencingToken, s.clock.Now())
}

func (s *Service) Events(principal Principal, scope Scope, id string, after int64) ([]Event, error) {
	if err := principal.Authorize(scope); err != nil {
		return nil, err
	}
	if err := validateJobID(id); err != nil {
		return nil, err
	}
	return s.store.Events(scope, id, after)
}

func bindCompletionCapability(scope Scope, id string, fencingToken int64, capability string) [sha256.Size]byte {
	hash := sha256.New()
	for _, value := range []string{
		"mindclade-completion-capability-v1",
		scope.TenantID,
		scope.ProjectID,
		id,
		fmt.Sprintf("%d", fencingToken),
		capability,
	} {
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(value)))
		_, _ = hash.Write(length[:])
		_, _ = hash.Write([]byte(value))
	}
	var binding [sha256.Size]byte
	copy(binding[:], hash.Sum(nil))
	return binding
}
