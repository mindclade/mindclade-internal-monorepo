package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"testing"
	"time"
)

type reconciliationKubernetesClient struct {
	items               []kubernetesJobSetObservation
	objects             map[string]bool
	acknowledgeDeletion bool
	calls               []kubernetesCall
}

func (c *reconciliationKubernetesClient) Create(_ context.Context, _ string, _ any) (kubernetesIdentity, error) {
	panic("unexpected create")
}

func (c *reconciliationKubernetesClient) Patch(_ context.Context, _ string, _ any) error {
	panic("unexpected patch")
}

func (c *reconciliationKubernetesClient) Delete(_ context.Context, path string) error {
	c.calls = append(c.calls, kubernetesCall{method: http.MethodDelete, path: path})
	if c.acknowledgeDeletion {
		c.objects[path] = false
	}
	return nil
}

func (c *reconciliationKubernetesClient) Exists(_ context.Context, path string) (bool, error) {
	c.calls = append(c.calls, kubernetesCall{method: http.MethodGet, path: path})
	return c.objects[path], nil
}

func (c *reconciliationKubernetesClient) Read(_ context.Context, path string) ([]byte, error) {
	c.calls = append(c.calls, kubernetesCall{method: http.MethodGet, path: path})
	return json.Marshal(kubernetesJobSetList{
		APIVersion: "jobset.x-k8s.io/v1alpha2",
		Kind:       "JobSetList",
		Items:      c.items,
	})
}

func reconciliationConfig() KubernetesAttemptLauncherConfig {
	return KubernetesAttemptLauncherConfig{
		Namespace: "mindclade-model-runtime", ResourceIncarnation: testResourceIncarnation,
		QueueName:            "inference",
		WorkerServiceAccount: "inference-worker", WorkerTrustSecret: "worker-trusted-public-keys",
		WorkerImage:      "registry.example/mindclade/inference-worker@sha256:4444444444444444444444444444444444444444444444444444444444444444",
		ArtifactProxyURL: "http://artifact-proxy:8082", ControlPlaneURL: "http://control-plane:8081",
		ResultStorageClass: "premium-rwo", ResultStorageRequest: "20Gi", ArtifactScratchLimit: "48Gi",
		QueueDeadlineSeconds: 60, StartupDeadlineSeconds: 60, ActiveDeadlineSeconds: 60,
		TTLSecondsAfterFinish: 300, LaunchTimeout: 5 * time.Millisecond,
		ReconcileInterval: time.Second,
	}
}

func observedJobSet(
	t *testing.T,
	job Job,
	createdAt time.Time,
	suspended bool,
	terminalState string,
	active int64,
	failed int64,
) kubernetesJobSetObservation {
	t.Helper()
	document := map[string]any{
		"apiVersion": "jobset.x-k8s.io/v1alpha2",
		"kind":       "JobSet",
		"metadata": map[string]any{
			"name":              attemptResourceBase(testResourceIncarnation, job.ID, job.FencingToken),
			"creationTimestamp": createdAt.UTC().Format(time.RFC3339),
			"labels": map[string]string{
				"app.kubernetes.io/name":   "inference-attempt",
				"mindclade.dev/managed-by": "control-plane",
			},
			"annotations": map[string]string{
				"mindclade.dev/job-id":               job.ID,
				"mindclade.dev/resource-incarnation": testResourceIncarnation,
				"mindclade.dev/tenant-id":            job.Scope.TenantID,
				"mindclade.dev/project-id":           job.Scope.ProjectID,
				"mindclade.dev/fencing-token":        "1",
				"mindclade.dev/queue-deadline":       "60",
				"mindclade.dev/startup-deadline":     "60",
				"mindclade.dev/active-deadline":      "60",
			},
		},
		"spec": map[string]any{"suspend": suspended},
		"status": map[string]any{
			"terminalState": terminalState,
			"replicatedJobsStatus": []any{map[string]any{
				"name": "worker", "active": active, "failed": failed,
				"ready": int64(0), "succeeded": int64(0), "suspended": int64(0),
			}},
		},
	}
	document["metadata"].(map[string]any)["annotations"].(map[string]string)["mindclade.dev/fencing-token"] =
		strconv.FormatInt(job.FencingToken, 10)
	encoded, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	var observation kubernetesJobSetObservation
	if err := json.Unmarshal(encoded, &observation); err != nil {
		t.Fatal(err)
	}
	return observation
}

func reconciliationClientFor(job Job, observation kubernetesJobSetObservation, acknowledge bool) *reconciliationKubernetesClient {
	path := "/apis/jobset.x-k8s.io/v1alpha2/namespaces/mindclade-model-runtime/jobsets/" +
		observation.Metadata.Name
	return &reconciliationKubernetesClient{
		items: []kubernetesJobSetObservation{observation}, objects: map[string]bool{path: true},
		acknowledgeDeletion: acknowledge,
	}
}

func TestJobSetReconcilerTerminatesBoundedFailureStatesBeforeFailingJob(t *testing.T) {
	tests := []struct {
		name          string
		age           time.Duration
		suspended     bool
		terminalState string
		active        int64
		failed        int64
		resumedAge    time.Duration
		failureCode   string
	}{
		{name: "queue deadline", age: 61 * time.Second, suspended: true, failureCode: "attempt_queue_deadline"},
		{name: "startup deadline", age: 70 * time.Second, resumedAge: 61 * time.Second, failureCode: "attempt_startup_deadline"},
		{name: "active deadline", age: 181 * time.Second, active: 1, failureCode: "attempt_active_deadline"},
		{name: "failed child", age: 10 * time.Second, active: 1, failed: 1, failureCode: "attempt_jobset_failed"},
		{name: "terminal without receipt", age: 10 * time.Second, terminalState: "Completed", failureCode: "attempt_completion_missing"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service, principal, request, attempt := leaseFixture(t)
			now := time.Unix(1_700_000_500, 0).UTC()
			observation := observedJobSet(
				t, attempt.Job, now.Add(-test.age), test.suspended,
				test.terminalState, test.active, test.failed,
			)
			if test.resumedAge > 0 {
				observation.Status.Conditions = []kubernetesConditionObservation{{
					Type: "Suspended", Status: "False",
					LastTransitionTime: now.Add(-test.resumedAge).Format(time.RFC3339),
				}}
			}
			client := reconciliationClientFor(attempt.Job, observation, true)
			reconciler, err := newJobSetReconciler(
				client, service, reconciliationConfig(), fixedClock{now},
			)
			if err != nil {
				t.Fatal(err)
			}
			if err := reconciler.ReconcileOnce(context.Background()); err != nil {
				t.Fatal(err)
			}
			stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
			if err != nil || stored.State != StateFailed || stored.FailureCode != test.failureCode {
				t.Fatalf("reconciled job = (%+v, %v)", stored, err)
			}
			if len(client.calls) != 3 || client.calls[1].method != http.MethodDelete || client.calls[2].method != http.MethodGet {
				t.Fatalf("Kubernetes calls = %+v", client.calls)
			}
		})
	}
}

func TestJobSetReconcilerKeepsHealthyExactFenceRunning(t *testing.T) {
	service, principal, request, attempt := leaseFixture(t)
	now := time.Unix(1_700_000_500, 0).UTC()
	observation := observedJobSet(t, attempt.Job, now.Add(-30*time.Second), false, "", 1, 0)
	client := reconciliationClientFor(attempt.Job, observation, true)
	reconciler, err := newJobSetReconciler(client, service, reconciliationConfig(), fixedClock{now})
	if err != nil {
		t.Fatal(err)
	}
	if err := reconciler.ReconcileOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
	if err != nil || stored.State != StateRunning || len(client.calls) != 1 {
		t.Fatalf("healthy reconciliation = (%+v, %v), calls=%+v", stored, err, client.calls)
	}
}

func TestJobSetReconcilerRejectsMismatchedFenceMetadata(t *testing.T) {
	service, principal, request, attempt := leaseFixture(t)
	now := time.Unix(1_700_000_500, 0).UTC()
	observation := observedJobSet(t, attempt.Job, now.Add(-30*time.Second), false, "", 1, 0)
	observation.Metadata.Annotations["mindclade.dev/fencing-token"] = "2"
	client := reconciliationClientFor(attempt.Job, observation, true)
	reconciler, err := newJobSetReconciler(client, service, reconciliationConfig(), fixedClock{now})
	if err != nil {
		t.Fatal(err)
	}
	if err := reconciler.ReconcileOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
	if err != nil || stored.State != StateFailed || stored.FailureCode != "attempt_status_invalid" {
		t.Fatalf("mismatched fence reconciliation = (%+v, %v)", stored, err)
	}
}

func TestJobSetReconcilerRejectsMismatchedResourceIncarnation(t *testing.T) {
	service, principal, request, attempt := leaseFixture(t)
	now := time.Unix(1_700_000_500, 0).UTC()
	observation := observedJobSet(t, attempt.Job, now.Add(-30*time.Second), false, "", 1, 0)
	observation.Metadata.Annotations["mindclade.dev/resource-incarnation"] = "ffeeddccbbaa99887766554433221100"
	client := reconciliationClientFor(attempt.Job, observation, true)
	reconciler, err := newJobSetReconciler(client, service, reconciliationConfig(), fixedClock{now})
	if err != nil {
		t.Fatal(err)
	}
	if err := reconciler.ReconcileOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
	if err != nil || stored.State != StateFailed || stored.FailureCode != "attempt_status_invalid" {
		t.Fatalf("mismatched incarnation reconciliation = (%+v, %v)", stored, err)
	}
}

func TestJobSetReconcilerDoesNotReleaseBudgetBeforeDeletionAcknowledgment(t *testing.T) {
	service, principal, request, attempt := leaseFixture(t)
	now := time.Unix(1_700_000_500, 0).UTC()
	observation := observedJobSet(t, attempt.Job, now.Add(-10*time.Second), false, "Failed", 0, 1)
	client := reconciliationClientFor(attempt.Job, observation, false)
	reconciler, err := newJobSetReconciler(client, service, reconciliationConfig(), fixedClock{now})
	if err != nil {
		t.Fatal(err)
	}
	if err := reconciler.ReconcileOnce(context.Background()); !errors.Is(err, ErrAttemptReconciliation) {
		t.Fatalf("unacknowledged deletion error = %v", err)
	}
	stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
	if err != nil || stored.State != StateRunning {
		t.Fatalf("unacknowledged deletion changed state = (%+v, %v)", stored, err)
	}
}

func TestJobSetReconcilerStartupRecoveryTerminatesOrphansAndMissingAttempts(t *testing.T) {
	t.Run("orphan from lost in-memory state", func(t *testing.T) {
		service, _, _ := fixture()
		job := Job{ID: "job-11111111111111111111111111111111-00000009", FencingToken: 1}
		now := time.Unix(1_700_000_500, 0).UTC()
		observation := observedJobSet(t, job, now.Add(-time.Minute), true, "", 0, 0)
		const previousIncarnation = "ffeeddccbbaa99887766554433221100"
		observation.Metadata.Name = attemptResourceBase(previousIncarnation, job.ID, job.FencingToken)
		observation.Metadata.Annotations["mindclade.dev/resource-incarnation"] = previousIncarnation
		client := reconciliationClientFor(job, observation, true)
		reconciler, err := newJobSetReconciler(client, service, reconciliationConfig(), fixedClock{now})
		if err != nil {
			t.Fatal(err)
		}
		if err := reconciler.Recover(context.Background()); err != nil {
			t.Fatal(err)
		}
		if len(client.calls) != 3 || client.calls[1].method != http.MethodDelete {
			t.Fatalf("orphan recovery calls = %+v", client.calls)
		}
	})

	t.Run("running row without exact JobSet", func(t *testing.T) {
		service, principal, request, attempt := leaseFixture(t)
		client := &reconciliationKubernetesClient{
			objects: map[string]bool{}, acknowledgeDeletion: true,
		}
		now := time.Unix(1_700_000_500, 0).UTC()
		reconciler, err := newJobSetReconciler(client, service, reconciliationConfig(), fixedClock{now})
		if err != nil {
			t.Fatal(err)
		}
		if err := reconciler.Recover(context.Background()); err != nil {
			t.Fatal(err)
		}
		stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
		if err != nil || stored.State != StateFailed || stored.FailureCode != "attempt_jobset_missing" {
			t.Fatalf("missing JobSet recovery = (%+v, %v)", stored, err)
		}
	})

	t.Run("launch handoff grace", func(t *testing.T) {
		service, principal, request, attempt := leaseFixture(t)
		client := &reconciliationKubernetesClient{
			objects: map[string]bool{}, acknowledgeDeletion: true,
		}
		now := attempt.Job.UpdatedAt.Add(time.Second)
		reconciler, err := newJobSetReconciler(client, service, reconciliationConfig(), fixedClock{now})
		if err != nil {
			t.Fatal(err)
		}
		if err := reconciler.ReconcileOnce(context.Background()); err != nil {
			t.Fatal(err)
		}
		stored, err := service.Get(principal, request.Scope, attempt.Job.ID)
		if err != nil || stored.State != StateRunning || len(client.calls) != 1 {
			t.Fatalf("launch grace reconciliation = (%+v, %v), calls=%+v", stored, err, client.calls)
		}
	})
}
