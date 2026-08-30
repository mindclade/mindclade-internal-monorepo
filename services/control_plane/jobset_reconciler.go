package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"
)

var ErrAttemptReconciliation = errors.New("attempt reconciliation failed")

const managedJobSetSelector = "app.kubernetes.io/name=inference-attempt"

type kubernetesConditionObservation struct {
	Type               string `json:"type"`
	Status             string `json:"status"`
	LastTransitionTime string `json:"lastTransitionTime"`
}

type kubernetesJobSetList struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Metadata   struct {
		Continue string `json:"continue"`
	} `json:"metadata"`
	Items []kubernetesJobSetObservation `json:"items"`
}

type kubernetesJobSetObservation struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Metadata   struct {
		Name              string            `json:"name"`
		CreationTimestamp string            `json:"creationTimestamp"`
		Labels            map[string]string `json:"labels"`
		Annotations       map[string]string `json:"annotations"`
	} `json:"metadata"`
	Spec struct {
		Suspend bool `json:"suspend"`
	} `json:"spec"`
	Status struct {
		TerminalState  string                           `json:"terminalState"`
		Conditions     []kubernetesConditionObservation `json:"conditions"`
		ReplicatedJobs []struct {
			Name      string `json:"name"`
			Ready     int64  `json:"ready"`
			Succeeded int64  `json:"succeeded"`
			Failed    int64  `json:"failed"`
			Active    int64  `json:"active"`
			Suspended int64  `json:"suspended"`
		} `json:"replicatedJobsStatus"`
	} `json:"status"`
}

// JobSetReconciler is the reusable polling boundary between Kubernetes status
// and the fenced job state machine. It lists only launcher-managed JobSets,
// deletes one exact deterministic attempt name, observes absence, and only then
// releases budget through a failed terminal transition.
type JobSetReconciler struct {
	client  kubernetesResourceClient
	service *Service
	config  KubernetesAttemptLauncherConfig
	clock   Clock
}

func NewJobSetReconciler(
	client kubernetesResourceClient,
	service *Service,
	config KubernetesAttemptLauncherConfig,
) (*JobSetReconciler, error) {
	return newJobSetReconciler(client, service, config, systemClock{})
}

func newJobSetReconciler(
	client kubernetesResourceClient,
	service *Service,
	config KubernetesAttemptLauncherConfig,
	clock Clock,
) (*JobSetReconciler, error) {
	if client == nil || service == nil || clock == nil {
		return nil, fmt.Errorf("%w: reconciler dependencies are required", ErrInvalidRequest)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &JobSetReconciler{client: client, service: service, config: config, clock: clock}, nil
}

// Recover reconciles the current store against exact Kubernetes objects before
// the HTTP server starts. With the development in-memory store, a process restart
// has no job rows to recover, so this safely terminates orphan compute without
// claiming restoration of the lost API state.
func (r *JobSetReconciler) Recover(ctx context.Context) error {
	return r.ReconcileOnce(ctx)
}

// Run checks status at a bounded interval and returns on the first observation
// error. The composition root treats that as fatal instead of serving while the
// lifecycle safety loop is unavailable.
func (r *JobSetReconciler) Run(ctx context.Context) error {
	ticker := time.NewTicker(r.config.ReconcileInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := r.ReconcileOnce(ctx); err != nil {
				return err
			}
		}
	}
}

func (r *JobSetReconciler) ReconcileOnce(ctx context.Context) error {
	observations, err := r.list(ctx)
	if err != nil {
		return err
	}
	jobs, err := r.service.nonterminalJobs()
	if err != nil {
		return fmt.Errorf("%w: list nonterminal jobs", ErrAttemptReconciliation)
	}
	running := make(map[string]Job, len(jobs))
	for _, job := range jobs {
		if job.State == StateRunning && job.FencingToken > 0 {
			running[attemptResourceBase(r.config.ResourceIncarnation, job.ID, job.FencingToken)] = job
		}
	}

	seen := make(map[string]bool, len(observations))
	now := r.clock.Now()
	for _, observation := range observations {
		name := observation.Metadata.Name
		if !dnsLabelPattern.MatchString(name) || !strings.HasPrefix(name, "mc-") {
			return fmt.Errorf("%w: managed JobSet name is invalid", ErrAttemptReconciliation)
		}
		if seen[name] {
			return fmt.Errorf("%w: Kubernetes returned duplicate JobSet identity", ErrAttemptReconciliation)
		}
		seen[name] = true
		job, known := running[name]
		if !known {
			if err := r.terminate(ctx, name); err != nil {
				return fmt.Errorf("%w: terminate orphan JobSet: %v", ErrAttemptReconciliation, err)
			}
			continue
		}
		if err := r.validateIdentity(observation, job); err != nil {
			if err := r.terminateAndFail(ctx, job, "attempt_status_invalid"); err != nil {
				return err
			}
			continue
		}
		failureCode, terminal := r.failure(observation, now)
		if terminal {
			if err := r.terminateAndFail(ctx, job, failureCode); err != nil {
				return err
			}
		}
	}

	missingGrace := r.config.LaunchTimeout + 2*r.config.ReconcileInterval
	for name, job := range running {
		if seen[name] || now.Sub(job.UpdatedAt) < missingGrace {
			continue
		}
		if err := r.terminateAndFail(ctx, job, "attempt_jobset_missing"); err != nil {
			return err
		}
	}
	return nil
}

func (r *JobSetReconciler) list(ctx context.Context) ([]kubernetesJobSetObservation, error) {
	collectionPath := fmt.Sprintf(
		"/apis/jobset.x-k8s.io/v1alpha2/namespaces/%s/jobsets?labelSelector=%s",
		r.config.Namespace,
		url.QueryEscape(managedJobSetSelector),
	)
	payload, err := r.client.Read(ctx, collectionPath)
	if err != nil {
		return nil, fmt.Errorf("%w: list managed JobSets: %v", ErrAttemptReconciliation, err)
	}
	var list kubernetesJobSetList
	if err := json.Unmarshal(payload, &list); err != nil ||
		list.APIVersion != "jobset.x-k8s.io/v1alpha2" || list.Kind != "JobSetList" ||
		list.Metadata.Continue != "" {
		return nil, fmt.Errorf("%w: invalid or paginated JobSet list", ErrAttemptReconciliation)
	}
	return list.Items, nil
}

func (r *JobSetReconciler) validateIdentity(observation kubernetesJobSetObservation, job Job) error {
	annotations := observation.Metadata.Annotations
	labels := observation.Metadata.Labels
	if observation.APIVersion != "jobset.x-k8s.io/v1alpha2" || observation.Kind != "JobSet" ||
		labels["app.kubernetes.io/name"] != "inference-attempt" ||
		labels["mindclade.dev/managed-by"] != "control-plane" ||
		annotations["mindclade.dev/job-id"] != job.ID ||
		annotations["mindclade.dev/resource-incarnation"] != r.config.ResourceIncarnation ||
		annotations["mindclade.dev/tenant-id"] != job.Scope.TenantID ||
		annotations["mindclade.dev/project-id"] != job.Scope.ProjectID ||
		annotations["mindclade.dev/fencing-token"] != strconv.FormatInt(job.FencingToken, 10) ||
		annotations["mindclade.dev/queue-deadline"] != strconv.FormatInt(r.config.QueueDeadlineSeconds, 10) ||
		annotations["mindclade.dev/startup-deadline"] != strconv.FormatInt(r.config.StartupDeadlineSeconds, 10) ||
		annotations["mindclade.dev/active-deadline"] != strconv.FormatInt(r.config.ActiveDeadlineSeconds, 10) {
		return errors.New("JobSet identity or lifecycle policy does not match the fenced job")
	}
	return nil
}

func (r *JobSetReconciler) failure(observation kubernetesJobSetObservation, now time.Time) (string, bool) {
	switch observation.Status.TerminalState {
	case "Completed":
		return "attempt_completion_missing", true
	case "Failed":
		return "attempt_jobset_failed", true
	case "":
	default:
		return "attempt_status_invalid", true
	}
	var resumedAt time.Time
	for _, condition := range observation.Status.Conditions {
		if condition.Type == "Suspended" && condition.Status == "False" {
			transition, err := time.Parse(time.RFC3339, condition.LastTransitionTime)
			if err != nil || transition.After(now) {
				return "attempt_status_invalid", true
			}
			resumedAt = transition
		}
		if condition.Status != "True" {
			continue
		}
		switch condition.Type {
		case "Completed", "Complete":
			return "attempt_completion_missing", true
		case "Failed":
			return "attempt_jobset_failed", true
		}
	}
	started := false
	for _, status := range observation.Status.ReplicatedJobs {
		if status.Ready < 0 || status.Succeeded < 0 || status.Failed < 0 || status.Active < 0 || status.Suspended < 0 {
			return "attempt_status_invalid", true
		}
		if status.Failed > 0 {
			return "attempt_jobset_failed", true
		}
		if status.Succeeded > 0 {
			return "attempt_completion_missing", true
		}
		started = started || status.Ready > 0 || status.Active > 0 || status.Succeeded > 0
	}
	createdAt, err := time.Parse(time.RFC3339, observation.Metadata.CreationTimestamp)
	if err != nil || createdAt.After(now) {
		return "attempt_status_invalid", true
	}
	age := now.Sub(createdAt)
	queueDeadline := time.Duration(r.config.QueueDeadlineSeconds) * time.Second
	startupDeadline := time.Duration(r.config.StartupDeadlineSeconds) * time.Second
	activeDeadline := time.Duration(r.config.ActiveDeadlineSeconds) * time.Second
	if observation.Spec.Suspend && age >= queueDeadline {
		return "attempt_queue_deadline", true
	}
	if !observation.Spec.Suspend && !started && !resumedAt.IsZero() && now.Sub(resumedAt) >= startupDeadline {
		return "attempt_startup_deadline", true
	}
	if !observation.Spec.Suspend && !started && age >= queueDeadline+startupDeadline {
		return "attempt_startup_deadline", true
	}
	if age >= queueDeadline+startupDeadline+activeDeadline {
		return "attempt_active_deadline", true
	}
	return "", false
}

func (r *JobSetReconciler) terminateAndFail(ctx context.Context, job Job, failureCode string) error {
	current, err := r.service.store.Get(job.Scope, job.ID)
	if errors.Is(err, ErrNotFound) || (err == nil && (current.State != StateRunning || current.FencingToken != job.FencingToken)) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("%w: re-read fenced job: %v", ErrAttemptReconciliation, err)
	}
	if err := r.terminate(ctx, attemptResourceBase(r.config.ResourceIncarnation, job.ID, job.FencingToken)); err != nil {
		return fmt.Errorf("%w: terminate fenced JobSet: %v", ErrAttemptReconciliation, err)
	}
	_, err = r.service.FailAttempt(job.Scope, job.ID, job.FencingToken, failureCode)
	if errors.Is(err, ErrInvalidTransition) || errors.Is(err, ErrStaleFence) || errors.Is(err, ErrNotFound) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("%w: fail fenced job: %v", ErrAttemptReconciliation, err)
	}
	return nil
}

func (r *JobSetReconciler) terminate(ctx context.Context, name string) error {
	resourcePath := fmt.Sprintf(
		"/apis/jobset.x-k8s.io/v1alpha2/namespaces/%s/jobsets/%s",
		r.config.Namespace,
		name,
	)
	return deleteAndObserveKubernetesResource(
		ctx,
		r.client,
		resourcePath,
		r.config.LaunchTimeout,
	)
}
