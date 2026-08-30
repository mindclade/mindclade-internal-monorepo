package controlplane

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"errors"
	"testing"
	"time"
)

const (
	modelDigest        = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	inputDigest        = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	resultDigest       = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	testJobIncarnation = "11111111111111111111111111111111"
	testJobID          = "job-" + testJobIncarnation + "-00000001"
)

type fixedClock struct{ value time.Time }

func (c fixedClock) Now() time.Time { return c.value }

type acceptingResultVerifier struct{}

func (acceptingResultVerifier) VerifyCommitted(context.Context, Scope, string, uint64) error {
	return nil
}

func testResultPublication() *ResultPublication {
	key, err := NewEd25519SigningKey(
		"artifact-test-key",
		ed25519.NewKeyFromSeed(bytes.Repeat([]byte{5}, ed25519.SeedSize)),
	)
	if err != nil {
		panic(err)
	}
	entropy := make([]byte, 4096)
	for index := range entropy {
		entropy[index] = byte(index)
	}
	issuer := newArtifactCapabilityIssuerForTest(
		key,
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
		bytes.NewReader(entropy),
		15*time.Minute,
	)
	publication, err := NewResultPublication(issuer, acceptingResultVerifier{})
	if err != nil {
		panic(err)
	}
	return publication
}

func deterministicMemoryStore(maxJobs, maxEvents int) *MemoryStore {
	store, err := newBoundedMemoryStore(
		maxJobs,
		maxEvents,
		bytes.NewReader(bytes.Repeat([]byte{0x11}, jobIDIncarnationBytes)),
	)
	if err != nil {
		panic(err)
	}
	return store
}

func testMemoryStore() *MemoryStore {
	return deterministicMemoryStore(10_000, 80_000)
}

func fixture() (*Service, Principal, SubmitRequest) {
	scope := Scope{TenantID: "tenant-a", ProjectID: "project-a"}
	principal := Principal{Subject: "user-1", TenantID: scope.TenantID, Projects: map[string]bool{scope.ProjectID: true}}
	request := SubmitRequest{
		Scope: scope, IdempotencyKey: "request-0001", ModelDigest: modelDigest,
		InputArtifact: inputDigest, Seed: 7, DiffusionSteps: 16,
	}
	catalog := NewMemoryArtifactCatalog()
	if err := catalog.Grant(scope, modelDigest, ArtifactModel); err != nil {
		panic(err)
	}
	if err := catalog.Grant(scope, inputDigest, ArtifactInput); err != nil {
		panic(err)
	}
	service := NewServiceWithClockAndResultPublication(
		testMemoryStore(),
		BudgetPolicy{
			MaxGPUMillisecondsPerJob:      10_000,
			MaxOutstandingGPUMilliseconds: 20_000,
			MaxActiveJobsPerTenant:        4,
		},
		catalog,
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
		testResultPublication(),
	)
	return service, principal, request
}

func TestArtifactOwnershipIsRequired(t *testing.T) {
	service, principal, request := fixture()
	request.InputArtifact = resultDigest
	if _, _, err := service.Submit(principal, request); !errors.Is(err, ErrArtifactForbidden) {
		t.Fatalf("ownership error = %v", err)
	}
}

func TestTenantAggregateBudgetAndActiveJobCapAreAtomic(t *testing.T) {
	scope := Scope{TenantID: "tenant-a", ProjectID: "project-a"}
	principal := Principal{Subject: "user-1", TenantID: scope.TenantID, Projects: map[string]bool{scope.ProjectID: true}}
	catalog := NewMemoryArtifactCatalog()
	_ = catalog.Grant(scope, modelDigest, ArtifactModel)
	_ = catalog.Grant(scope, inputDigest, ArtifactInput)
	service := NewServiceWithClock(
		testMemoryStore(),
		BudgetPolicy{MaxGPUMillisecondsPerJob: 2_000, MaxOutstandingGPUMilliseconds: 1_999, MaxActiveJobsPerTenant: 2},
		catalog,
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
	)
	for i, key := range []string{"request-aggregate-1", "request-aggregate-2"} {
		_, _, err := service.Submit(principal, SubmitRequest{
			Scope: scope, IdempotencyKey: key, ModelDigest: modelDigest,
			InputArtifact: inputDigest, DiffusionSteps: 10,
		})
		if i == 0 && err != nil {
			t.Fatal(err)
		}
		if i == 1 && !errors.Is(err, ErrBudgetExceeded) {
			t.Fatalf("aggregate budget error = %v", err)
		}
	}
}

func TestIdentifierGrammarRejectsControlCharacters(t *testing.T) {
	_, principal, request := fixture()
	request.Scope.TenantID = "tenant\x00other"
	if _, _, err := NewService(testMemoryStore(), BudgetPolicy{}, NewMemoryArtifactCatalog()).Submit(principal, request); !errors.Is(err, ErrForbidden) && !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("identifier error = %v", err)
	}
}

func TestMemoryStoreJobIDsAreDeterministicWithinProcessAndUniqueAcrossRestarts(t *testing.T) {
	request := SubmitRequest{
		Scope:          Scope{TenantID: "tenant-a", ProjectID: "project-a"},
		IdempotencyKey: "request-restart-identity", ModelDigest: modelDigest,
		InputArtifact: inputDigest, Seed: 7, DiffusionSteps: 16,
	}
	newStore := func(entropyByte byte) *MemoryStore {
		store, err := newBoundedMemoryStore(
			10,
			80,
			bytes.NewReader(bytes.Repeat([]byte{entropyByte}, jobIDIncarnationBytes)),
		)
		if err != nil {
			t.Fatal(err)
		}
		return store
	}
	create := func(store *MemoryStore) Job {
		job, replayed, err := store.CreateOrReplay(request, 1600, BudgetPolicy{}, time.Unix(0, 0))
		if err != nil || replayed {
			t.Fatalf("create job = (%+v, %v, %v)", job, replayed, err)
		}
		return job
	}

	first := create(newStore(0x11))
	repeatedProcess := create(newStore(0x11))
	restarted := create(newStore(0x22))
	if first.ID != testJobID || repeatedProcess.ID != first.ID {
		t.Fatalf("deterministic job IDs = %q, %q", first.ID, repeatedProcess.ID)
	}
	if restarted.ID == first.ID || !jobIDPattern.MatchString(restarted.ID) {
		t.Fatalf("restart job IDs = %q, %q", first.ID, restarted.ID)
	}
}

func TestMemoryStoreRejectsMissingEntropyAndUnrepresentableLimits(t *testing.T) {
	if _, err := newBoundedMemoryStore(1, 1, nil); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("nil job ID entropy error = %v", err)
	}
	if _, err := newBoundedMemoryStore(1, 1, bytes.NewReader(nil)); err == nil {
		t.Fatal("short job ID entropy was accepted")
	}
	if _, err := newBoundedMemoryStore(maximumJobSequence+1, maximumJobSequence+1, bytes.NewReader(bytes.Repeat([]byte{1}, jobIDIncarnationBytes))); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("unrepresentable job sequence error = %v", err)
	}
}

func TestJobIDIngressRejectsLegacySequentialIdentity(t *testing.T) {
	const legacyJobID = "job-00000001"
	scope := Scope{TenantID: "tenant-a", ProjectID: "project-a"}
	principal := Principal{
		Subject: "user-1", TenantID: scope.TenantID,
		Projects: map[string]bool{scope.ProjectID: true},
	}
	store := testMemoryStore()
	service := NewServiceWithClock(
		store,
		BudgetPolicy{},
		NewMemoryArtifactCatalog(),
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
	)
	checks := []struct {
		name string
		call func() error
	}{
		{"store get", func() error { _, err := store.Get(scope, legacyJobID); return err }},
		{"store transition", func() error {
			_, err := store.Transition(scope, legacyJobID, StateAdmitted, "", 0, time.Unix(0, 0))
			return err
		}},
		{"store lease", func() error {
			_, err := store.Lease(
				scope, legacyJobID, AttemptProvenance{}, "", [ed25519.PublicKeySize]byte{}, time.Unix(0, 0),
			)
			return err
		}},
		{"store events", func() error { _, err := store.Events(scope, legacyJobID, 0); return err }},
		{"service get", func() error { _, err := service.Get(principal, scope, legacyJobID); return err }},
		{"service cancel", func() error { _, err := service.Cancel(principal, scope, legacyJobID); return err }},
		{"service admit", func() error { _, err := service.Admit(scope, legacyJobID); return err }},
		{"service lease", func() error {
			_, err := service.LeaseAttempt(scope, legacyJobID, testAttemptProvenance())
			return err
		}},
		{"service fail", func() error {
			_, err := service.FailAttempt(scope, legacyJobID, 1, "attempt_failed")
			return err
		}},
		{"service events", func() error {
			_, err := service.Events(principal, scope, legacyJobID, 0)
			return err
		}},
		{"result upload request", func() error {
			return (ResultUploadRequest{
				JobID: legacyJobID, TenantID: scope.TenantID, ProjectID: scope.ProjectID,
				ResultDigest: resultDigest, ResultSizeBytes: 1, FencingToken: 1,
				SchemaVersion: "v1alpha1",
			}).Validate()
		}},
		{"result receipt", func() error {
			return (ResultReceipt{
				JobID: legacyJobID, TenantID: scope.TenantID, ProjectID: scope.ProjectID,
				SelectedCandidateID: "candidate-0000",
			}).Validate()
		}},
	}
	for _, check := range checks {
		t.Run(check.name, func(t *testing.T) {
			if err := check.call(); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("legacy job ID error = %v", err)
			}
		})
	}
}

func TestSubmitIsIdempotentWithinScope(t *testing.T) {
	service, principal, request := fixture()
	first, replayed, err := service.Submit(principal, request)
	if err != nil || replayed {
		t.Fatalf("first submit = (%v, %v)", replayed, err)
	}
	second, replayed, err := service.Submit(principal, request)
	if err != nil || !replayed || second.ID != first.ID {
		t.Fatalf("replay = (%q, %v, %v), want %q", second.ID, replayed, err, first.ID)
	}
	request.Seed++
	if _, _, err := service.Submit(principal, request); !errors.Is(err, ErrIdempotencyConflict) {
		t.Fatalf("changed replay error = %v", err)
	}
}

func TestTenantIsolationDoesNotRevealJob(t *testing.T) {
	service, principal, request := fixture()
	job, _, _ := service.Submit(principal, request)
	attacker := Principal{Subject: "user-2", TenantID: "tenant-b", Projects: map[string]bool{"project-a": true}}
	if _, err := service.Get(attacker, request.Scope, job.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-tenant get error = %v", err)
	}
}

func TestFencingRejectsLateAttempt(t *testing.T) {
	service, principal, request := fixture()
	job, _, _ := service.Submit(principal, request)
	if _, err := service.Admit(request.Scope, job.ID); err != nil {
		t.Fatal(err)
	}
	leased, err := service.LeaseAttempt(request.Scope, job.ID, testAttemptProvenance())
	if err != nil {
		t.Fatal(err)
	}
	stale := resultReceipt(leased)
	stale.FencingToken++
	stale.ResultManifestPath = "/var/run/mindclade-results/output/result.fence-2.receipt.json"
	if _, err := completeSigned(t, service, leased, stale, leased.CompletionCapability); !errors.Is(err, ErrStaleFence) {
		t.Fatalf("stale completion error = %v", err)
	}
	completed, err := completeSigned(t, service, leased, resultReceipt(leased), leased.CompletionCapability)
	if err != nil || completed.State != StateSucceeded {
		t.Fatalf("completion = (%v, %v)", completed.State, err)
	}
}

func TestBudgetAndCancellation(t *testing.T) {
	service, principal, request := fixture()
	request.DiffusionSteps = 101
	if _, _, err := service.Submit(principal, request); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("budget error = %v", err)
	}
	request.DiffusionSteps = 16
	job, _, _ := service.Submit(principal, request)
	cancelled, err := service.Cancel(principal, request.Scope, job.ID)
	if err != nil || cancelled.State != StateCancelled {
		t.Fatalf("cancel = (%v, %v)", cancelled.State, err)
	}
}
