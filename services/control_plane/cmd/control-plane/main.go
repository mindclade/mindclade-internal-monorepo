package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	controlplane "github.com/mindclade/mindclade-internal-monorepo/services/control_plane"
)

func main() {
	address := required("MINDCLADE_CONTROL_PLANE_ADDRESS")
	catalogDocument, err := os.ReadFile(required("MINDCLADE_ARTIFACT_CATALOG_FILE"))
	if err != nil {
		log.Fatal(err)
	}
	var entries []struct {
		TenantID            string                    `json:"tenant_id"`
		ProjectID           string                    `json:"project_id"`
		Digest              string                    `json:"digest"`
		Kind                controlplane.ArtifactKind `json:"kind"`
		SizeBytes           uint64                    `json:"size_bytes,omitempty"`
		BundleArchiveDigest string                    `json:"bundle_archive_digest,omitempty"`
		BundleArchiveSize   uint64                    `json:"bundle_archive_size_bytes,omitempty"`
		BundleSigningKeyID  string                    `json:"bundle_signing_key_id,omitempty"`
	}
	decoder := json.NewDecoder(bytes.NewReader(catalogDocument))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&entries); err != nil {
		log.Fatal(err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		log.Fatal("artifact catalog must contain one JSON value")
	}
	if len(entries) == 0 {
		log.Fatal("artifact catalog must not be empty")
	}
	catalog := controlplane.NewMemoryArtifactCatalog()
	for _, entry := range entries {
		metadata := controlplane.ArtifactMetadata{
			Digest: entry.Digest, Kind: entry.Kind, SizeBytes: entry.SizeBytes,
			BundleArchiveDigest:    entry.BundleArchiveDigest,
			BundleArchiveSizeBytes: entry.BundleArchiveSize,
			BundleSigningKeyID:     entry.BundleSigningKeyID,
		}
		if err := catalog.Register(controlplane.Scope{TenantID: entry.TenantID, ProjectID: entry.ProjectID}, metadata); err != nil {
			log.Fatal(err)
		}
	}
	internalSecret, err := os.ReadFile(required("MINDCLADE_INTERNAL_IDENTITY_SECRET_FILE"))
	if err != nil {
		log.Fatal(err)
	}
	identity, err := controlplane.NewInternalIdentityVerifier(internalSecret)
	if err != nil {
		log.Fatal(err)
	}
	schedulerKey, err := controlplane.LoadEd25519SigningKey(
		required("MINDCLADE_SCHEDULER_SIGNING_KEY_ID"),
		required("MINDCLADE_SCHEDULER_PRIVATE_KEY_FILE"),
	)
	if err != nil {
		log.Fatal(err)
	}
	artifactKey, err := controlplane.LoadEd25519SigningKey(
		required("MINDCLADE_ARTIFACT_CAPABILITY_SIGNING_KEY_ID"),
		required("MINDCLADE_ARTIFACT_CAPABILITY_PRIVATE_KEY_FILE"),
	)
	if err != nil {
		log.Fatal(err)
	}
	resultCapabilityIssuer, err := controlplane.NewArtifactCapabilityIssuer(artifactKey)
	if err != nil {
		log.Fatal(err)
	}
	stagingCapabilityIssuer, err := controlplane.NewArtifactCapabilityIssuerWithTTL(
		artifactKey,
		requiredSeconds("MINDCLADE_STAGING_CAPABILITY_TTL_SECONDS", 60, 604_800),
	)
	if err != nil {
		log.Fatal(err)
	}
	artifactProxyURL := required("MINDCLADE_ARTIFACT_PROXY_URL")
	artifactVerifier, err := controlplane.NewHTTPResultArtifactVerifier(
		artifactProxyURL,
		resultCapabilityIssuer,
		requiredSeconds("MINDCLADE_ARTIFACT_VERIFY_TIMEOUT_SECONDS", 1, 30),
	)
	if err != nil {
		log.Fatal(err)
	}
	resultPublication, err := controlplane.NewResultPublication(resultCapabilityIssuer, artifactVerifier)
	if err != nil {
		log.Fatal(err)
	}
	memoryStore, err := controlplane.NewMemoryStore()
	if err != nil {
		log.Fatal(err)
	}
	service := controlplane.NewServiceWithResultPublication(
		memoryStore,
		controlplane.BudgetPolicy{
			MaxGPUMillisecondsPerJob:      25_600,
			MaxOutstandingGPUMilliseconds: 102_400,
			MaxActiveJobsPerTenant:        8,
		},
		catalog,
		resultPublication,
	)
	launchTimeout := requiredSeconds("MINDCLADE_KUBERNETES_LAUNCH_TIMEOUT_SECONDS", 1, 30)
	kubernetesClient, err := controlplane.NewKubernetesRESTClient(
		required("MINDCLADE_KUBERNETES_API_SERVER"),
		required("MINDCLADE_KUBERNETES_TOKEN_FILE"),
		required("MINDCLADE_KUBERNETES_CA_FILE"),
		launchTimeout,
	)
	if err != nil {
		log.Fatal(err)
	}
	queueDeadline := requiredSeconds("MINDCLADE_QUEUE_DEADLINE_SECONDS", 60, 86_400)
	startupDeadline := requiredSeconds("MINDCLADE_ATTEMPT_STARTUP_DEADLINE_SECONDS", 60, 3_600)
	activeDeadline := requiredSeconds("MINDCLADE_ATTEMPT_ACTIVE_DEADLINE_SECONDS", 60, 86_400)
	reconcileInterval := requiredSeconds("MINDCLADE_JOBSET_RECONCILE_INTERVAL_SECONDS", 1, 300)
	resourceIncarnation, err := controlplane.GenerateResourceIncarnation()
	if err != nil {
		log.Fatal(err)
	}
	launcherConfig := controlplane.KubernetesAttemptLauncherConfig{
		Namespace:              required("MINDCLADE_RUNTIME_NAMESPACE"),
		ResourceIncarnation:    resourceIncarnation,
		QueueName:              required("MINDCLADE_KUEUE_LOCAL_QUEUE"),
		WorkerServiceAccount:   required("MINDCLADE_WORKER_SERVICE_ACCOUNT"),
		WorkerTrustSecret:      required("MINDCLADE_WORKER_TRUST_SECRET"),
		WorkerImage:            required("MINDCLADE_WORKER_IMAGE"),
		ArtifactProxyURL:       artifactProxyURL,
		ControlPlaneURL:        required("MINDCLADE_CONTROL_PLANE_INTERNAL_URL"),
		ResultStorageClass:     required("MINDCLADE_RESULT_STORAGE_CLASS"),
		ResultStorageRequest:   required("MINDCLADE_RESULT_STORAGE_REQUEST"),
		ArtifactScratchLimit:   required("MINDCLADE_ARTIFACT_SCRATCH_LIMIT"),
		QueueDeadlineSeconds:   int64(queueDeadline / time.Second),
		StartupDeadlineSeconds: int64(startupDeadline / time.Second),
		ActiveDeadlineSeconds:  int64(activeDeadline / time.Second),
		TTLSecondsAfterFinish:  int64(requiredSeconds("MINDCLADE_JOBSET_TTL_SECONDS", 300, 604_800) / time.Second),
		LaunchTimeout:          launchTimeout,
		ReconcileInterval:      reconcileInterval,
	}
	launcher, err := controlplane.NewKubernetesAttemptLauncher(
		kubernetesClient,
		catalog,
		stagingCapabilityIssuer,
		schedulerKey,
		launcherConfig,
	)
	if err != nil {
		log.Fatal(err)
	}
	reconciler, err := controlplane.NewJobSetReconciler(kubernetesClient, service, launcherConfig)
	if err != nil {
		log.Fatal(err)
	}
	if err := reconciler.Recover(context.Background()); err != nil {
		log.Fatal(err)
	}
	dispatcher, err := controlplane.NewDispatcher(service, launcher, controlplane.AttemptProvenance{
		ServingRevisionDigest: required("MINDCLADE_SERVING_REVISION_DIGEST"),
		ExecutionMode:         "eager",
		SamplerDigest:         required("MINDCLADE_SAMPLER_DIGEST"),
	})
	if err != nil {
		log.Fatal(err)
	}
	server := &http.Server{
		Addr: address, Handler: controlplane.NewHTTPHandlerWithDispatcher(service, identity, dispatcher),
		ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second,
		WriteTimeout: launchTimeout + 15*time.Second, IdleTimeout: 60 * time.Second,
		MaxHeaderBytes: 32 * 1024,
	}
	log.Printf("control plane listening on %s", address)
	runtimeContext, cancelRuntime := context.WithCancel(context.Background())
	runtimeErrors := make(chan error, 2)
	go func() { runtimeErrors <- server.ListenAndServe() }()
	go func() { runtimeErrors <- reconciler.Run(runtimeContext) }()
	runtimeError := <-runtimeErrors
	cancelRuntime()
	_ = server.Close()
	if errors.Is(runtimeError, http.ErrServerClosed) || errors.Is(runtimeError, context.Canceled) {
		return
	}
	log.Fatal(runtimeError)
}

func required(name string) string {
	value := os.Getenv(name)
	if value == "" {
		log.Fatalf("%s is required", name)
	}
	return value
}

func requiredSeconds(name string, minimum, maximum int64) time.Duration {
	value := required(name)
	seconds, err := strconv.ParseInt(value, 10, 64)
	if err != nil || seconds < minimum || seconds > maximum {
		log.Fatalf("%s must be within %d..%d seconds", name, minimum, maximum)
	}
	return time.Duration(seconds) * time.Second
}
