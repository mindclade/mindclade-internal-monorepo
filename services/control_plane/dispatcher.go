package controlplane

import (
	"context"
	"errors"
	"fmt"
)

var (
	ErrAttemptLaunch       = errors.New("attempt launcher failed")
	ErrAttemptCancellation = errors.New("attempt cancellation failed")
)

// AttemptLauncher is the scheduler adapter. Implementations receive the exact
// fenced lease and must embed its completion capability in the signed worker
// manifest without logging it. The concrete launcher persists it only in a
// mode-0400 immutable, owner-bound Kubernetes Secret.
type AttemptLauncher interface {
	Launch(context.Context, AttemptLease) error
}

// AttemptCanceller is implemented by launchers that can acknowledge deletion
// of one exact fenced scheduler object.
type AttemptCanceller interface {
	Cancel(context.Context, Job) error
}

// AttemptLauncherFunc adapts a function into an AttemptLauncher.
type AttemptLauncherFunc func(context.Context, AttemptLease) error

func (f AttemptLauncherFunc) Launch(ctx context.Context, attempt AttemptLease) error {
	return f(ctx, attempt)
}

// Dispatcher advances a newly created job through admission and leasing before
// handing its attempt to the scheduler adapter.
type Dispatcher struct {
	service    *Service
	launcher   AttemptLauncher
	provenance AttemptProvenance
}

func NewDispatcher(service *Service, launcher AttemptLauncher, provenance AttemptProvenance) (*Dispatcher, error) {
	if service == nil || launcher == nil {
		return nil, ErrInvalidRequest
	}
	if err := provenance.Validate(); err != nil {
		return nil, err
	}
	return &Dispatcher{service: service, launcher: launcher, provenance: provenance}, nil
}

// Dispatch is intentionally synchronous through the launch handoff: a submit
// response cannot claim an attempt is running until the launcher accepts it.
func (d *Dispatcher) Dispatch(ctx context.Context, job Job) (Job, error) {
	if _, err := d.service.Admit(job.Scope, job.ID); err != nil {
		return Job{}, err
	}
	attempt, err := d.service.LeaseAttempt(job.Scope, job.ID, d.provenance)
	if err != nil {
		return Job{}, err
	}
	if err := d.launcher.Launch(ctx, attempt); err != nil {
		canceller, ok := d.launcher.(AttemptCanceller)
		if !ok {
			// Keep the job nonterminal, and therefore keep its budget reserved,
			// when scheduler absence cannot be proven.
			return attempt.Job, ErrAttemptLaunch
		}
		if cancellationErr := canceller.Cancel(ctx, attempt.Job); cancellationErr != nil {
			return attempt.Job, errors.Join(
				ErrAttemptLaunch,
				fmt.Errorf("%w: %v", ErrAttemptCancellation, cancellationErr),
			)
		}
		failed, transitionErr := d.service.FailAttempt(
			job.Scope,
			job.ID,
			attempt.Job.FencingToken,
			"attempt_launch_failed",
		)
		if transitionErr != nil {
			return Job{}, errors.Join(ErrAttemptLaunch, transitionErr)
		}
		return failed, ErrAttemptLaunch
	}
	return attempt.Job, nil
}

// Cancel terminates a running scheduler object before releasing the job's
// outstanding budget. Queue-only states do not have a scheduler object.
func (d *Dispatcher) Cancel(ctx context.Context, principal Principal, scope Scope, id string) (Job, error) {
	job, err := d.service.Get(principal, scope, id)
	if err != nil {
		return Job{}, err
	}
	if job.State == StateRunning {
		canceller, ok := d.launcher.(AttemptCanceller)
		if !ok {
			return Job{}, ErrAttemptCancellation
		}
		if err := canceller.Cancel(ctx, job); err != nil {
			return Job{}, fmt.Errorf("%w: %v", ErrAttemptCancellation, err)
		}
	}
	return d.service.Cancel(principal, scope, id)
}
