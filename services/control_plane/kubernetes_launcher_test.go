package controlplane

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

const (
	bundleArchiveDigest     = "sha256:9999999999999999999999999999999999999999999999999999999999999999"
	testResourceIncarnation = "00112233445566778899aabbccddeeff"
)

type kubernetesCall struct {
	method   string
	path     string
	document any
}

type recordingKubernetesClient struct {
	calls        []kubernetesCall
	failCreateAt int
	createCount  int
}

func (c *recordingKubernetesClient) Create(_ context.Context, path string, document any) (kubernetesIdentity, error) {
	c.createCount++
	c.calls = append(c.calls, kubernetesCall{method: http.MethodPost, path: path, document: document})
	if c.failCreateAt == c.createCount {
		return kubernetesIdentity{}, errors.New("injected create rejection")
	}
	metadata := document.(map[string]any)["metadata"].(map[string]any)
	return kubernetesIdentity{Name: metadata["name"].(string), UID: "uid-attempt-001"}, nil
}

func (c *recordingKubernetesClient) Patch(_ context.Context, path string, document any) error {
	c.calls = append(c.calls, kubernetesCall{method: http.MethodPatch, path: path, document: document})
	return nil
}

func (c *recordingKubernetesClient) Delete(_ context.Context, path string) error {
	c.calls = append(c.calls, kubernetesCall{method: http.MethodDelete, path: path})
	return nil
}

func (c *recordingKubernetesClient) Exists(_ context.Context, path string) (bool, error) {
	c.calls = append(c.calls, kubernetesCall{method: http.MethodGet, path: path})
	return false, nil
}

func (c *recordingKubernetesClient) Read(_ context.Context, path string) ([]byte, error) {
	c.calls = append(c.calls, kubernetesCall{method: http.MethodGet, path: path})
	return []byte(`{"apiVersion":"jobset.x-k8s.io/v1alpha2","kind":"JobSetList","items":[]}`), nil
}

func testSigningKey(t *testing.T, id string, value byte) *Ed25519SigningKey {
	t.Helper()
	key, err := NewEd25519SigningKey(id, ed25519.NewKeyFromSeed(bytes.Repeat([]byte{value}, ed25519.SeedSize)))
	if err != nil {
		t.Fatal(err)
	}
	return key
}

func testLauncher(t *testing.T, client kubernetesResourceClient) (*KubernetesAttemptLauncher, AttemptLease, ed25519.PublicKey) {
	t.Helper()
	scope := Scope{TenantID: "tenant-a", ProjectID: "project-a"}
	catalog := NewMemoryArtifactCatalog()
	if err := catalog.Register(scope, ArtifactMetadata{
		Digest: modelDigest, Kind: ArtifactModel,
		BundleArchiveDigest: bundleArchiveDigest, BundleArchiveSizeBytes: 4096,
		BundleSigningKeyID: "bundle-2026-08",
	}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.Register(scope, ArtifactMetadata{
		Digest: inputDigest, Kind: ArtifactInput, SizeBytes: 2048,
	}); err != nil {
		t.Fatal(err)
	}
	artifactKey := testSigningKey(t, "artifact-2026-08", 7)
	issuer := newArtifactCapabilityIssuerForTest(
		artifactKey,
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
		bytes.NewReader(bytes.Repeat([]byte{4}, 128)),
		2*time.Hour,
	)
	schedulerKey := testSigningKey(t, "scheduler-2026-08", 9)
	launcher, err := NewKubernetesAttemptLauncher(client, catalog, issuer, schedulerKey, KubernetesAttemptLauncherConfig{
		Namespace: "mindclade-model-runtime", ResourceIncarnation: testResourceIncarnation,
		QueueName:            "inference",
		WorkerServiceAccount: "inference-worker", WorkerTrustSecret: "worker-trusted-public-keys",
		WorkerImage:      "registry.example/mindclade/inference-worker@sha256:4444444444444444444444444444444444444444444444444444444444444444",
		ArtifactProxyURL: "http://artifact-proxy:8082", ControlPlaneURL: "http://control-plane:8081",
		ResultStorageClass: "premium-rwo", ResultStorageRequest: "20Gi", ArtifactScratchLimit: "48Gi",
		QueueDeadlineSeconds: 600, StartupDeadlineSeconds: 300,
		ActiveDeadlineSeconds: 3600, TTLSecondsAfterFinish: 86400,
		LaunchTimeout: 5 * time.Second, ReconcileInterval: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	attempt := AttemptLease{
		Job: Job{
			ID: testJobID, Scope: scope, State: StateRunning,
			ModelDigest: modelDigest, InputArtifact: inputDigest, Seed: 7, DiffusionSteps: 16,
			FencingToken: 3,
		},
		CompletionCapability:        strings.Repeat("C", 43),
		CompletionSigningPrivateKey: base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{6}, ed25519.SeedSize)),
		Provenance:                  testAttemptProvenance(),
	}
	return launcher, attempt, schedulerKey.privateKey.Public().(ed25519.PublicKey)
}

func TestArtifactDownloadCapabilityMatchesRustV2Contract(t *testing.T) {
	key := testSigningKey(t, "artifact-2026-08", 7)
	issuer := newArtifactCapabilityIssuerForTest(
		key,
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
		bytes.NewReader(bytes.Repeat([]byte{4}, 32)),
		15*time.Minute,
	)
	encoded, err := issuer.IssueDownload(
		Scope{TenantID: "tenant-a", ProjectID: "project-a"}, bundleArchiveDigest, 4096,
	)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatal(err)
	}
	var capability artifactCapability
	if err := json.Unmarshal(payload, &capability); err != nil {
		t.Fatal(err)
	}
	if capability.Operation != "download" || capability.ExpiresUnix != 1_700_000_900 || capability.MaxSizeBytes != 4096 {
		t.Fatalf("capability fields = %+v", capability)
	}
	signature, err := base64.RawURLEncoding.DecodeString(capability.Signature)
	if err != nil || !ed25519.Verify(key.privateKey.Public().(ed25519.PublicKey), capability.signingBytes(), signature) {
		t.Fatal("Rust-compatible capability signature did not verify")
	}
	digest := sha256.Sum256(capability.signingBytes())
	if actual := base64.RawStdEncoding.EncodeToString(digest[:]); actual != "ZUGc/Pkv3sVcE4110b+iFaNnNaKcerpUHoGXTt4X0Y4" {
		t.Fatalf("capability signing bytes changed: %s", actual)
	}
}

func TestArtifactUploadCapabilityMatchesRustV2Contract(t *testing.T) {
	key := testSigningKey(t, "artifact-2026-08", 7)
	issuer := newArtifactCapabilityIssuerForTest(
		key,
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
		bytes.NewReader(bytes.Repeat([]byte{6}, 32)),
		15*time.Minute,
	)
	encoded, sessionID, err := issuer.IssueUpload(
		Scope{TenantID: "tenant-a", ProjectID: "project-a"}, resultDigest, 4096,
	)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatal(err)
	}
	var capability artifactCapability
	if err := json.Unmarshal(payload, &capability); err != nil {
		t.Fatal(err)
	}
	if capability.Operation != "upload" || capability.SessionID != sessionID || capability.MaxSizeBytes != 4096 {
		t.Fatalf("upload capability fields = %+v", capability)
	}
	signature, err := base64.RawURLEncoding.DecodeString(capability.Signature)
	if err != nil || !ed25519.Verify(key.privateKey.Public().(ed25519.PublicKey), capability.signingBytes(), signature) {
		t.Fatal("Rust-compatible upload capability signature did not verify")
	}
	digest := sha256.Sum256(capability.signingBytes())
	if actual := base64.RawStdEncoding.EncodeToString(digest[:]); actual != "fGgOU51CLaBT/lsMD56xQOYfwqowejyuD1YoDm13dNg" {
		t.Fatalf("upload capability signing bytes changed: %s", actual)
	}
}

func TestHTTPResultArtifactVerifierRequiresExactAuthenticatedMetadata(t *testing.T) {
	key := testSigningKey(t, "artifact-2026-08", 7)
	issuer := newArtifactCapabilityIssuerForTest(
		key,
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
		bytes.NewReader(bytes.Repeat([]byte{3}, 128)),
		15*time.Minute,
	)
	wrongSize := false
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodHead || !strings.HasSuffix(request.URL.Path, resultDigest) {
			t.Errorf("metadata request = %s %s", request.Method, request.URL.Path)
		}
		encoded := request.Header.Get("X-Mindclade-Capability")
		payload, err := base64.RawURLEncoding.DecodeString(encoded)
		if err != nil {
			t.Error(err)
		}
		var capability artifactCapability
		if err := json.Unmarshal(payload, &capability); err != nil {
			t.Error(err)
		}
		if capability.Operation != "download" || capability.Digest != resultDigest || capability.MaxSizeBytes != 4096 {
			t.Errorf("metadata capability = %+v", capability)
		}
		response.Header().Set("X-Mindclade-Artifact-Digest", resultDigest)
		if wrongSize {
			response.Header().Set("X-Mindclade-Artifact-Size", "4095")
		} else {
			response.Header().Set("X-Mindclade-Artifact-Size", "4096")
		}
		response.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	verifier, err := NewHTTPResultArtifactVerifier(server.URL, issuer, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := verifier.VerifyCommitted(
		context.Background(), Scope{TenantID: "tenant-a", ProjectID: "project-a"}, resultDigest, 4096,
	); err != nil {
		t.Fatal(err)
	}
	wrongSize = true
	if err := verifier.VerifyCommitted(
		context.Background(), Scope{TenantID: "tenant-a", ProjectID: "project-a"}, resultDigest, 4096,
	); !errors.Is(err, ErrResultPublication) {
		t.Fatalf("metadata mismatch error = %v", err)
	}
}

func TestCanonicalWorkerManifestMatchesPythonJSONContract(t *testing.T) {
	manifest := workerJobManifest{
		JobID: testJobID, TenantID: "tenant-a", ProjectID: "project-a",
		ModelDigest: modelDigest, BundleManifestDigest: modelDigest,
		BundleArchiveDigest: bundleArchiveDigest, InputDigest: inputDigest,
		ServingRevisionDigest: servingRevisionDigest,
		BundlePath:            "/a/bundle", InputPath: "/a/input", OutputDirectory: "/r/output",
		BundleSigningKeyID: "bundle-key", SchedulerSigningKeyID: "scheduler-key",
		BundleDownloadCapability: "bundle-cap", InputDownloadCapability: "input-cap",
		CompletionCapability: "completion-cap", FencingToken: 3, Seed: 7,
		CompletionSigningPrivateKey: "completion-private-key",
		NumSamples:                  1, NumSteps: 16, Device: "cuda", SchemaVersion: "v1alpha1",
	}
	actual, err := canonicalUnsignedManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	expected := `{"bundle_archive_digest":"` + bundleArchiveDigest + `","bundle_download_capability":"bundle-cap","bundle_manifest_digest":"` + modelDigest + `","bundle_path":"/a/bundle","bundle_signing_key_id":"bundle-key","completion_capability":"completion-cap","completion_signing_private_key":"completion-private-key","device":"cuda","fencing_token":3,"input_digest":"` + inputDigest + `","input_download_capability":"input-cap","input_path":"/a/input","job_id":"` + testJobID + `","model_digest":"` + modelDigest + `","num_samples":1,"num_steps":16,"output_directory":"/r/output","project_id":"project-a","scheduler_signing_key_id":"scheduler-key","schema_version":"v1alpha1","seed":7,"serving_revision_digest":"` + servingRevisionDigest + `","tenant_id":"tenant-a"}`
	if string(actual) != expected {
		t.Fatalf("canonical manifest mismatch\nactual:   %s\nexpected: %s", actual, expected)
	}
}

func TestKubernetesLauncherRejectsCapabilityLifetimeShorterThanAttemptBounds(t *testing.T) {
	client := &recordingKubernetesClient{}
	launcher, _, _ := testLauncher(t, client)
	shortIssuer := newArtifactCapabilityIssuerForTest(
		testSigningKey(t, "artifact-short-ttl", 3),
		fixedClock{time.Unix(1_700_000_000, 0).UTC()},
		bytes.NewReader(bytes.Repeat([]byte{2}, 128)),
		15*time.Minute,
	)
	if _, err := NewKubernetesAttemptLauncher(
		client,
		launcher.catalog,
		shortIssuer,
		launcher.manifestSigningKey,
		launcher.config,
	); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("short staging capability TTL error = %v", err)
	}
}

func TestKubernetesLauncherRequiresValidResourceIncarnation(t *testing.T) {
	client := &recordingKubernetesClient{}
	launcher, _, _ := testLauncher(t, client)
	config := launcher.config
	config.ResourceIncarnation = "not-a-process-identity"
	if _, err := NewKubernetesAttemptLauncher(
		client,
		launcher.catalog,
		launcher.capabilityIssuer,
		launcher.manifestSigningKey,
		config,
	); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("invalid resource incarnation error = %v", err)
	}
}

func TestKubernetesLauncherRequiresScratchForTheAdmittedArtifactEnvelope(t *testing.T) {
	client := &recordingKubernetesClient{}
	launcher, _, _ := testLauncher(t, client)
	config := launcher.config
	config.ArtifactScratchLimit = "31Gi"
	if err := config.Validate(); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("undersized artifact scratch error = %v", err)
	}
	config.ArtifactScratchLimit = "32Gi"
	if err := config.Validate(); err != nil {
		t.Fatalf("minimum artifact scratch was rejected: %v", err)
	}
}

func TestKubernetesLauncherPublishesOnlyAfterSecretAndRetainedPVC(t *testing.T) {
	client := &recordingKubernetesClient{}
	launcher, attempt, schedulerPublicKey := testLauncher(t, client)
	if err := launcher.Launch(context.Background(), attempt); err != nil {
		t.Fatal(err)
	}
	if len(client.calls) != 4 {
		t.Fatalf("Kubernetes calls = %d, want 4", len(client.calls))
	}
	jobSet := client.calls[0].document.(map[string]any)
	jobSetJSON, _ := json.Marshal(jobSet)
	if int64(len(jobSetJSON))*int64(managedJobSetPageLimit) >= maximumAPIResponse/2 {
		t.Fatalf(
			"managed JobSet page has insufficient response-cap margin: item=%d, page=%d, cap=%d",
			len(jobSetJSON), managedJobSetPageLimit, maximumAPIResponse,
		)
	}
	if bytes.Contains(jobSetJSON, []byte(attempt.CompletionCapability)) || bytes.Contains(jobSetJSON, []byte("download_capability")) {
		t.Fatal("JobSet exposed bearer capability material")
	}
	labels := jobSet["metadata"].(map[string]any)["labels"].(map[string]string)
	if _, exists := labels["kueue.x-k8s.io/queue-name"]; exists {
		t.Fatal("JobSet was queue-visible before its dependencies existed")
	}
	if labels["mindclade.dev/managed-by"] != "control-plane" {
		t.Fatalf("JobSet managed label = %+v", labels)
	}
	annotations := jobSet["metadata"].(map[string]any)["annotations"].(map[string]string)
	if annotations["mindclade.dev/job-id"] != attempt.Job.ID ||
		annotations["mindclade.dev/resource-incarnation"] != testResourceIncarnation ||
		annotations["mindclade.dev/fencing-token"] != "3" ||
		annotations["mindclade.dev/queue-deadline"] != "600" {
		t.Fatalf("JobSet lifecycle identity = %+v", annotations)
	}
	secret := client.calls[1].document.(map[string]any)
	if secret["immutable"] != true || secret["kind"] != "Secret" {
		t.Fatalf("manifest resource = %+v", secret)
	}
	encodedManifest := secret["data"].(map[string]string)["job.json"]
	manifestJSON, err := base64.StdEncoding.DecodeString(encodedManifest)
	if err != nil {
		t.Fatal(err)
	}
	var manifest workerJobManifest
	if err := json.Unmarshal(manifestJSON, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Seed != 7 || manifest.NumSteps != 16 || manifest.FencingToken != 3 || manifest.OutputDirectory != workerOutputDirectory {
		t.Fatalf("worker request contract was not preserved: %+v", manifest)
	}
	capabilityPayload, err := base64.RawURLEncoding.DecodeString(manifest.BundleDownloadCapability)
	if err != nil {
		t.Fatal(err)
	}
	var capability artifactCapability
	if err := json.Unmarshal(capabilityPayload, &capability); err != nil {
		t.Fatal(err)
	}
	if lifetime := time.Duration(capability.ExpiresUnix-1_700_000_000) * time.Second; lifetime < launcher.config.minimumStagingCapabilityTTL() {
		t.Fatalf("staging capability lifetime %s is shorter than %s", lifetime, launcher.config.minimumStagingCapabilityTTL())
	}
	unsigned, _ := canonicalUnsignedManifest(manifest)
	signature, _ := base64.StdEncoding.DecodeString(manifest.ManifestSignature)
	if !ed25519.Verify(schedulerPublicKey, unsigned, signature) {
		t.Fatal("Python-compatible worker manifest signature did not verify")
	}
	pvc := client.calls[2].document.(map[string]any)
	if pvc["kind"] != "PersistentVolumeClaim" {
		t.Fatalf("result resource = %+v", pvc)
	}
	patch := client.calls[3].document.(map[string]any)
	patchLabels := patch["metadata"].(map[string]any)["labels"].(map[string]string)
	if patchLabels["kueue.x-k8s.io/queue-name"] != "inference" {
		t.Fatalf("queue publication patch = %+v", patch)
	}
	if !bytes.Contains(jobSetJSON, []byte(`"persistentVolumeClaim":{"claimName":"`+attemptPVCName(testResourceIncarnation, attempt.Job.ID, 3)+`"`)) || bytes.Contains(jobSetJSON, []byte(`"name":"job-results","emptyDir"`)) {
		t.Fatal("worker result volume is not the retained per-attempt PVC")
	}
	podSpec := jobSet["spec"].(map[string]any)["replicatedJobs"].([]any)[0].(map[string]any)["template"].(map[string]any)["spec"].(map[string]any)["template"].(map[string]any)["spec"].(map[string]any)
	containers := []any{podSpec["initContainers"].([]any)[0], podSpec["containers"].([]any)[0]}
	expectedEphemeralLimit := launcher.config.podEphemeralStorageLimit()
	if expectedEphemeralLimit != "50Gi" {
		t.Fatalf("pod ephemeral storage limit = %q", expectedEphemeralLimit)
	}
	for _, value := range containers {
		resources := value.(map[string]any)["resources"].(map[string]any)
		if resources["requests"].(map[string]string)["ephemeral-storage"] != expectedEphemeralLimit ||
			resources["limits"].(map[string]string)["ephemeral-storage"] != expectedEphemeralLimit {
			t.Fatalf("ephemeral storage resources = %+v", resources)
		}
	}
}

func TestKubernetesLauncherRollsBackPartialResources(t *testing.T) {
	client := &recordingKubernetesClient{failCreateAt: 3}
	launcher, attempt, _ := testLauncher(t, client)
	if err := launcher.Launch(context.Background(), attempt); err == nil {
		t.Fatal("PVC creation failure was accepted")
	}
	var deletes []string
	for _, call := range client.calls {
		if call.method == http.MethodDelete {
			deletes = append(deletes, call.path)
		}
	}
	if len(deletes) != 3 || !strings.Contains(deletes[0], "/jobsets/") || !strings.Contains(deletes[1], "/secrets/") || !strings.Contains(deletes[2], "/persistentvolumeclaims/") {
		t.Fatalf("partial launch cleanup = %v", deletes)
	}
}

func TestKubernetesLauncherCancellationTargetsExactFenceAndObservesDeletion(t *testing.T) {
	client := &recordingKubernetesClient{}
	launcher, attempt, _ := testLauncher(t, client)
	if err := launcher.Cancel(context.Background(), attempt.Job); err != nil {
		t.Fatal(err)
	}
	if len(client.calls) != 2 || client.calls[0].method != http.MethodDelete || client.calls[1].method != http.MethodGet {
		t.Fatalf("cancel calls = %+v", client.calls)
	}
	expected := "/jobsets/" + attemptResourceBase(testResourceIncarnation, attempt.Job.ID, attempt.Job.FencingToken)
	if !strings.HasSuffix(client.calls[0].path, expected) || client.calls[0].path != client.calls[1].path {
		t.Fatalf("cancel paths = %q, %q", client.calls[0].path, client.calls[1].path)
	}
}

func TestAttemptResourceIdentityIsDeterministicAndRestartUnique(t *testing.T) {
	const nextIncarnation = "ffeeddccbbaa99887766554433221100"
	jobID := testJobID
	current := attemptResourceBase(testResourceIncarnation, jobID, 3)
	if repeated := attemptResourceBase(testResourceIncarnation, jobID, 3); repeated != current {
		t.Fatalf("resource identity is not deterministic: %q != %q", repeated, current)
	}
	if restarted := attemptResourceBase(nextIncarnation, jobID, 3); restarted == current {
		t.Fatalf("process restart reused resource identity %q", current)
	}
	if nextFence := attemptResourceBase(testResourceIncarnation, jobID, 4); nextFence == current {
		t.Fatalf("new fence reused resource identity %q", current)
	}
	if len(current) > 54 || !dnsLabelPattern.MatchString(current) {
		t.Fatalf("resource base is not a bounded DNS label: %q", current)
	}
	currentPVC := attemptPVCName(testResourceIncarnation, jobID, 3)
	restartedPVC := attemptPVCName(nextIncarnation, jobID, 3)
	if currentPVC == restartedPVC || !dnsLabelPattern.MatchString(currentPVC) {
		t.Fatalf("retained PVC identities = %q, %q", currentPVC, restartedPVC)
	}
}

func TestResourceIncarnationGenerationUsesExactEntropy(t *testing.T) {
	incarnation, err := generateResourceIncarnation(
		bytes.NewReader(bytes.Repeat([]byte{0xab}, resourceIncarnationBytes)),
	)
	if err != nil {
		t.Fatal(err)
	}
	if incarnation != strings.Repeat("ab", resourceIncarnationBytes) {
		t.Fatalf("resource incarnation = %q", incarnation)
	}
	if _, err := generateResourceIncarnation(bytes.NewReader([]byte{1})); err == nil {
		t.Fatal("short resource-incarnation entropy was accepted")
	}
}

func TestKubernetesRESTClientRejectsCollisionWithoutLeakingDocument(t *testing.T) {
	const capability = "highly-sensitive-completion-capability"
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer test-token" {
			t.Error("Kubernetes bearer token missing")
		}
		response.WriteHeader(http.StatusConflict)
		_, _ = response.Write([]byte(`{"message":"already exists"}`))
	}))
	defer server.Close()
	tokenFile := t.TempDir() + "/token"
	if err := os.WriteFile(tokenFile, []byte("test-token\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	client := &KubernetesRESTClient{baseURL: server.URL, tokenFile: tokenFile, client: server.Client()}
	_, err := client.Create(context.Background(), "/api/v1/namespaces/test/secrets", map[string]string{"secret": capability})
	if err == nil || strings.Contains(err.Error(), capability) || !strings.Contains(err.Error(), "status 409") {
		t.Fatalf("collision error = %v", err)
	}
}
