package controlplane

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	workerArtifactRoot    = "/var/run/mindclade-artifacts"
	workerResultRoot      = "/var/run/mindclade-results"
	workerOutputDirectory = workerResultRoot + "/output"
	maximumBundleBytes    = uint64(8 * 1024 * 1024 * 1024)
	maximumInputBytes     = uint64(4 * 1024 * 1024 * 1024)
	maximumAPIResponse    = int64(1 << 20)
	minimumLifecycleSlack = time.Minute
)

var (
	digestPinnedImagePattern   = regexp.MustCompile(`^[^@\s]+@sha256:[0-9a-f]{64}$`)
	dnsLabelPattern            = regexp.MustCompile(`^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$`)
	dnsSubdomainPattern        = regexp.MustCompile(`^[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?$`)
	resourceIncarnationPattern = regexp.MustCompile(`^[0-9a-f]{32}$`)
)

const resourceIncarnationBytes = 16

// GenerateResourceIncarnation returns a process-unique identity used only in
// Kubernetes resource names. It independently namespaces each launcher process so
// a restarted development control plane cannot regenerate a retained object's name.
func GenerateResourceIncarnation() (string, error) {
	return generateResourceIncarnation(rand.Reader)
}

func generateResourceIncarnation(source io.Reader) (string, error) {
	raw := make([]byte, resourceIncarnationBytes)
	if source == nil {
		return "", errors.New("resource-incarnation entropy source is required")
	}
	if _, err := io.ReadFull(source, raw); err != nil {
		return "", fmt.Errorf("generate Kubernetes resource incarnation: %w", err)
	}
	return hex.EncodeToString(raw), nil
}

type kubernetesIdentity struct {
	Name string
	UID  string
}

type kubernetesResourceClient interface {
	Create(context.Context, string, any) (kubernetesIdentity, error)
	Patch(context.Context, string, any) error
	Delete(context.Context, string) error
	Exists(context.Context, string) (bool, error)
	Read(context.Context, string) ([]byte, error)
}

type kubernetesStatusError struct {
	method string
	status int
}

func (e kubernetesStatusError) Error() string {
	return fmt.Sprintf("Kubernetes %s rejected resource with status %d", e.method, e.status)
}

// KubernetesRESTClient is a bounded in-cluster REST client. It deliberately
// exposes only the three operations required by the attempt launcher.
type KubernetesRESTClient struct {
	baseURL   string
	tokenFile string
	client    *http.Client
}

func NewKubernetesRESTClient(apiServer, tokenFile, caFile string, timeout time.Duration) (*KubernetesRESTClient, error) {
	parsed, err := url.Parse(apiServer)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, fmt.Errorf("%w: Kubernetes API server must be an HTTPS origin", ErrInvalidRequest)
	}
	if timeout <= 0 || timeout > 30*time.Second {
		return nil, fmt.Errorf("%w: Kubernetes API timeout must be within 0..30s", ErrInvalidRequest)
	}
	if _, err := readKubernetesToken(tokenFile); err != nil {
		return nil, err
	}
	certificateBytes, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("read Kubernetes CA: %w", err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(certificateBytes) {
		return nil, errors.New("Kubernetes CA file contains no certificates")
	}
	transport := &http.Transport{
		TLSClientConfig:       &tls.Config{MinVersion: tls.VersionTLS12, RootCAs: roots},
		TLSHandshakeTimeout:   timeout,
		ResponseHeaderTimeout: timeout,
		ExpectContinueTimeout: time.Second,
		MaxIdleConns:          8,
		MaxIdleConnsPerHost:   8,
		IdleConnTimeout:       30 * time.Second,
	}
	return &KubernetesRESTClient{
		baseURL: strings.TrimRight(apiServer, "/"), tokenFile: tokenFile,
		client: &http.Client{Transport: transport, Timeout: timeout},
	}, nil
}

func readKubernetesToken(filename string) (string, error) {
	tokenBytes, err := os.ReadFile(filename)
	if err != nil {
		return "", fmt.Errorf("read Kubernetes service-account token: %w", err)
	}
	token := strings.TrimSpace(string(tokenBytes))
	if token == "" || strings.ContainsAny(token, "\r\n") {
		return "", errors.New("Kubernetes service-account token is invalid")
	}
	return token, nil
}

func (c *KubernetesRESTClient) Create(ctx context.Context, resourcePath string, document any) (kubernetesIdentity, error) {
	body, err := json.Marshal(document)
	if err != nil {
		return kubernetesIdentity{}, fmt.Errorf("marshal Kubernetes resource: %w", err)
	}
	response, err := c.request(ctx, http.MethodPost, resourcePath, "application/json", body)
	if err != nil {
		return kubernetesIdentity{}, err
	}
	var created struct {
		Metadata struct {
			Name string `json:"name"`
			UID  string `json:"uid"`
		} `json:"metadata"`
	}
	if err := json.Unmarshal(response, &created); err != nil || created.Metadata.Name == "" {
		return kubernetesIdentity{}, errors.New("Kubernetes create response omitted resource identity")
	}
	return kubernetesIdentity{Name: created.Metadata.Name, UID: created.Metadata.UID}, nil
}

func (c *KubernetesRESTClient) Patch(ctx context.Context, resourcePath string, document any) error {
	body, err := json.Marshal(document)
	if err != nil {
		return fmt.Errorf("marshal Kubernetes patch: %w", err)
	}
	_, err = c.request(ctx, http.MethodPatch, resourcePath, "application/merge-patch+json", body)
	return err
}

func (c *KubernetesRESTClient) Delete(ctx context.Context, resourcePath string) error {
	_, err := c.request(ctx, http.MethodDelete, resourcePath, "application/json", []byte(`{"gracePeriodSeconds":0,"propagationPolicy":"Foreground"}`))
	var statusError kubernetesStatusError
	if errors.As(err, &statusError) && statusError.status == http.StatusNotFound {
		return nil
	}
	return err
}

func (c *KubernetesRESTClient) Exists(ctx context.Context, resourcePath string) (bool, error) {
	_, err := c.request(ctx, http.MethodGet, resourcePath, "application/json", nil)
	if err == nil {
		return true, nil
	}
	var statusError kubernetesStatusError
	if errors.As(err, &statusError) && statusError.status == http.StatusNotFound {
		return false, nil
	}
	return false, err
}

func (c *KubernetesRESTClient) Read(ctx context.Context, resourcePath string) ([]byte, error) {
	return c.request(ctx, http.MethodGet, resourcePath, "application/json", nil)
}

func (c *KubernetesRESTClient) request(ctx context.Context, method, resourcePath, contentType string, body []byte) ([]byte, error) {
	if !strings.HasPrefix(resourcePath, "/") || strings.Contains(resourcePath, "..") {
		return nil, errors.New("Kubernetes resource path is invalid")
	}
	request, err := http.NewRequestWithContext(ctx, method, c.baseURL+resourcePath, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build Kubernetes request: %w", err)
	}
	token, err := readKubernetesToken(c.tokenFile)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", contentType)
	response, err := c.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("Kubernetes %s request failed: %w", method, err)
	}
	defer response.Body.Close()
	limited := io.LimitReader(response.Body, maximumAPIResponse+1)
	payload, readErr := io.ReadAll(limited)
	if readErr != nil {
		return nil, fmt.Errorf("read Kubernetes response: %w", readErr)
	}
	if int64(len(payload)) > maximumAPIResponse {
		return nil, errors.New("Kubernetes response exceeded the configured limit")
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, kubernetesStatusError{method: method, status: response.StatusCode}
	}
	return payload, nil
}

// KubernetesAttemptLauncherConfig contains only deployment policy, never
// request-controlled Kubernetes fields.
type KubernetesAttemptLauncherConfig struct {
	Namespace              string
	ResourceIncarnation    string
	QueueName              string
	WorkerServiceAccount   string
	WorkerTrustSecret      string
	WorkerImage            string
	ArtifactProxyURL       string
	ControlPlaneURL        string
	ResultStorageClass     string
	ResultStorageRequest   string
	ArtifactScratchLimit   string
	QueueDeadlineSeconds   int64
	StartupDeadlineSeconds int64
	ActiveDeadlineSeconds  int64
	TTLSecondsAfterFinish  int64
	LaunchTimeout          time.Duration
	ReconcileInterval      time.Duration
}

func (c KubernetesAttemptLauncherConfig) Validate() error {
	if !resourceIncarnationPattern.MatchString(c.ResourceIncarnation) {
		return fmt.Errorf("%w: resource incarnation must be 128 bits of lowercase hexadecimal", ErrInvalidRequest)
	}
	for name, value := range map[string]string{
		"namespace": c.Namespace, "queue": c.QueueName,
		"worker service account": c.WorkerServiceAccount, "worker trust secret": c.WorkerTrustSecret,
	} {
		if !dnsLabelPattern.MatchString(value) {
			return fmt.Errorf("%w: %s must be a DNS label", ErrInvalidRequest, name)
		}
	}
	if !dnsSubdomainPattern.MatchString(c.ResultStorageClass) || !digestPinnedImagePattern.MatchString(c.WorkerImage) {
		return fmt.Errorf("%w: storage class or worker image is invalid", ErrInvalidRequest)
	}
	for name, value := range map[string]string{
		"artifact proxy": c.ArtifactProxyURL,
		"control plane":  c.ControlPlaneURL,
	} {
		parsed, err := url.Parse(value)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
			return fmt.Errorf("%w: %s URL is invalid", ErrInvalidRequest, name)
		}
	}
	if !validQuantity(c.ResultStorageRequest) || !validQuantity(c.ArtifactScratchLimit) {
		return fmt.Errorf("%w: storage quantities must be explicit", ErrInvalidRequest)
	}
	if c.QueueDeadlineSeconds < 60 || c.QueueDeadlineSeconds > 86_400 ||
		c.StartupDeadlineSeconds < 60 || c.StartupDeadlineSeconds > 3_600 ||
		c.ActiveDeadlineSeconds < 60 || c.ActiveDeadlineSeconds > 86_400 ||
		c.TTLSecondsAfterFinish < 300 || c.TTLSecondsAfterFinish > 604_800 ||
		c.LaunchTimeout <= 0 || c.LaunchTimeout > 30*time.Second ||
		c.ReconcileInterval < time.Second || c.ReconcileInterval > 5*time.Minute {
		return fmt.Errorf("%w: attempt time bounds are invalid", ErrInvalidRequest)
	}
	return nil
}

func (c KubernetesAttemptLauncherConfig) minimumStagingCapabilityTTL() time.Duration {
	boundedLifecycle := time.Duration(
		c.QueueDeadlineSeconds+c.StartupDeadlineSeconds+c.ActiveDeadlineSeconds,
	) * time.Second
	return boundedLifecycle + c.LaunchTimeout + 2*c.ReconcileInterval + minimumLifecycleSlack
}

func validQuantity(value string) bool {
	matched, _ := regexp.MatchString(`^[1-9][0-9]*(?:Mi|Gi)$`, value)
	return matched
}

// KubernetesAttemptLauncher creates one suspended JobSet. The JobSet is not
// made visible to Kueue until its immutable manifest Secret and retained result
// PVC exist. Capabilities occur only in the Secret body, never metadata.
type KubernetesAttemptLauncher struct {
	client             kubernetesResourceClient
	catalog            ArtifactCatalog
	capabilityIssuer   *ArtifactCapabilityIssuer
	manifestSigningKey *Ed25519SigningKey
	config             KubernetesAttemptLauncherConfig
}

func NewKubernetesAttemptLauncher(
	client kubernetesResourceClient,
	catalog ArtifactCatalog,
	capabilityIssuer *ArtifactCapabilityIssuer,
	manifestSigningKey *Ed25519SigningKey,
	config KubernetesAttemptLauncherConfig,
) (*KubernetesAttemptLauncher, error) {
	if client == nil || catalog == nil || capabilityIssuer == nil || manifestSigningKey == nil {
		return nil, fmt.Errorf("%w: launcher dependencies are required", ErrInvalidRequest)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if capabilityIssuer.ttl < config.minimumStagingCapabilityTTL() {
		return nil, fmt.Errorf(
			"%w: staging capability TTL does not cover the bounded attempt lifecycle",
			ErrInvalidRequest,
		)
	}
	return &KubernetesAttemptLauncher{
		client: client, catalog: catalog, capabilityIssuer: capabilityIssuer,
		manifestSigningKey: manifestSigningKey, config: config,
	}, nil
}

func (l *KubernetesAttemptLauncher) Launch(parent context.Context, attempt AttemptLease) (launchErr error) {
	if attempt.Job.State != StateRunning || attempt.Job.FencingToken < 1 || attempt.CompletionCapability == "" {
		return fmt.Errorf("%w: launcher requires a fenced running attempt", ErrInvalidRequest)
	}
	model, ok := l.catalog.Resolve(attempt.Job.Scope, attempt.Job.ModelDigest, ArtifactModel)
	if !ok || model.BundleArchiveSizeBytes > maximumBundleBytes {
		return fmt.Errorf("%w: model staging metadata is unavailable or oversized", ErrArtifactForbidden)
	}
	input, ok := l.catalog.Resolve(attempt.Job.Scope, attempt.Job.InputArtifact, ArtifactInput)
	if !ok || input.SizeBytes > maximumInputBytes {
		return fmt.Errorf("%w: input staging metadata is unavailable or oversized", ErrArtifactForbidden)
	}
	bundleCapability, err := l.capabilityIssuer.IssueDownload(attempt.Job.Scope, model.BundleArchiveDigest, model.BundleArchiveSizeBytes)
	if err != nil {
		return err
	}
	inputCapability, err := l.capabilityIssuer.IssueDownload(attempt.Job.Scope, input.Digest, input.SizeBytes)
	if err != nil {
		return err
	}
	manifest, err := l.signedManifest(attempt, model, bundleCapability, inputCapability)
	if err != nil {
		return err
	}
	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("marshal signed worker manifest: %w", err)
	}

	baseName := attemptResourceBase(l.config.ResourceIncarnation, attempt.Job.ID, attempt.Job.FencingToken)
	jobSetPath := fmt.Sprintf("/apis/jobset.x-k8s.io/v1alpha2/namespaces/%s/jobsets", l.config.Namespace)
	secretPath := fmt.Sprintf("/api/v1/namespaces/%s/secrets", l.config.Namespace)
	pvcPath := fmt.Sprintf("/api/v1/namespaces/%s/persistentvolumeclaims", l.config.Namespace)
	operationContext, cancel := context.WithTimeout(context.WithoutCancel(parent), l.config.LaunchTimeout)
	defer cancel()

	jobSetCreated := false
	secretCreated := false
	pvcCreated := false
	defer func() {
		if launchErr == nil {
			return
		}
		cleanupContext := context.Background()
		if jobSetCreated {
			if err := deleteAndObserveKubernetesResource(
				cleanupContext,
				l.client,
				jobSetPath+"/"+baseName,
				l.config.LaunchTimeout,
			); err != nil {
				launchErr = errors.Join(launchErr, fmt.Errorf("observe launch rollback: %w", err))
				// Keep the Secret and PVC available if the JobSet might still run.
				return
			}
		}
		if secretCreated {
			deleteContext, cancelDelete := context.WithTimeout(cleanupContext, l.config.LaunchTimeout)
			err := l.client.Delete(deleteContext, secretPath+"/"+baseName+"-manifest")
			cancelDelete()
			if err != nil {
				launchErr = errors.Join(launchErr, fmt.Errorf("delete launch Secret: %w", err))
			}
		}
		if pvcCreated {
			deleteContext, cancelDelete := context.WithTimeout(cleanupContext, l.config.LaunchTimeout)
			err := l.client.Delete(deleteContext, pvcPath+"/"+baseName+"-result")
			cancelDelete()
			if err != nil {
				launchErr = errors.Join(launchErr, fmt.Errorf("delete launch PVC: %w", err))
			}
		}
	}()

	identity, err := l.client.Create(operationContext, jobSetPath, l.jobSet(attempt, baseName))
	if err != nil {
		return fmt.Errorf("create suspended JobSet: %w", err)
	}
	if identity.Name != baseName || identity.UID == "" {
		return errors.New("created JobSet identity did not match the launch request")
	}
	jobSetCreated = true
	owner := map[string]any{
		"apiVersion": "jobset.x-k8s.io/v1alpha2", "kind": "JobSet", "name": identity.Name,
		"uid": identity.UID, "controller": true, "blockOwnerDeletion": true,
	}
	secretIdentity, err := l.client.Create(operationContext, secretPath, l.manifestSecret(baseName, owner, manifestJSON))
	if err != nil {
		return fmt.Errorf("create attempt manifest Secret: %w", err)
	}
	secretCreated = true
	if secretIdentity.Name != baseName+"-manifest" {
		return errors.New("created Secret identity did not match the launch request")
	}
	pvcIdentity, err := l.client.Create(operationContext, pvcPath, l.resultPVC(baseName))
	if err != nil {
		return fmt.Errorf("create retained result PVC: %w", err)
	}
	pvcCreated = true
	if pvcIdentity.Name != baseName+"-result" {
		return errors.New("created PVC identity did not match the launch request")
	}
	patchPath := jobSetPath + "/" + baseName
	if err := l.client.Patch(operationContext, patchPath, map[string]any{
		"metadata": map[string]any{"labels": map[string]string{"kueue.x-k8s.io/queue-name": l.config.QueueName}},
	}); err != nil {
		return fmt.Errorf("publish JobSet to Kueue: %w", err)
	}
	return nil
}

// Cancel deletes and observes absence of the exact fence-derived JobSet before
// the dispatcher marks the job terminal. The retained result PVC is not
// deleted; it may contain partial data needed for incident analysis.
func (l *KubernetesAttemptLauncher) Cancel(parent context.Context, job Job) error {
	if job.State != StateRunning || job.FencingToken < 1 {
		return fmt.Errorf("%w: only a running fenced job can be cancelled", ErrInvalidRequest)
	}
	name := attemptResourceBase(l.config.ResourceIncarnation, job.ID, job.FencingToken)
	resourcePath := fmt.Sprintf(
		"/apis/jobset.x-k8s.io/v1alpha2/namespaces/%s/jobsets/%s",
		l.config.Namespace,
		name,
	)
	return deleteAndObserveKubernetesResource(parent, l.client, resourcePath, l.config.LaunchTimeout)
}

func deleteAndObserveKubernetesResource(
	parent context.Context,
	client kubernetesResourceClient,
	resourcePath string,
	timeout time.Duration,
) error {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), timeout)
	defer cancel()
	if err := client.Delete(ctx, resourcePath); err != nil {
		return err
	}
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		exists, err := client.Exists(ctx, resourcePath)
		if err != nil {
			return err
		}
		if !exists {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("observe JobSet deletion: %w", ctx.Err())
		case <-ticker.C:
		}
	}
}

type workerJobManifest struct {
	JobID                       string `json:"job_id"`
	TenantID                    string `json:"tenant_id"`
	ProjectID                   string `json:"project_id"`
	ModelDigest                 string `json:"model_digest"`
	BundleManifestDigest        string `json:"bundle_manifest_digest"`
	BundleArchiveDigest         string `json:"bundle_archive_digest"`
	InputDigest                 string `json:"input_digest"`
	ServingRevisionDigest       string `json:"serving_revision_digest"`
	BundlePath                  string `json:"bundle_path"`
	InputPath                   string `json:"input_path"`
	OutputDirectory             string `json:"output_directory"`
	BundleSigningKeyID          string `json:"bundle_signing_key_id"`
	SchedulerSigningKeyID       string `json:"scheduler_signing_key_id"`
	ManifestSignature           string `json:"manifest_signature"`
	BundleDownloadCapability    string `json:"bundle_download_capability"`
	InputDownloadCapability     string `json:"input_download_capability"`
	CompletionCapability        string `json:"completion_capability"`
	CompletionSigningPrivateKey string `json:"completion_signing_private_key"`
	FencingToken                int64  `json:"fencing_token"`
	Seed                        uint64 `json:"seed"`
	NumSamples                  int64  `json:"num_samples"`
	NumSteps                    uint32 `json:"num_steps"`
	Device                      string `json:"device"`
	SchemaVersion               string `json:"schema_version"`
}

func (l *KubernetesAttemptLauncher) signedManifest(
	attempt AttemptLease,
	model ArtifactMetadata,
	bundleCapability string,
	inputCapability string,
) (workerJobManifest, error) {
	manifest := workerJobManifest{
		JobID: attempt.Job.ID, TenantID: attempt.Job.Scope.TenantID, ProjectID: attempt.Job.Scope.ProjectID,
		ModelDigest: attempt.Job.ModelDigest, BundleManifestDigest: attempt.Job.ModelDigest,
		BundleArchiveDigest: model.BundleArchiveDigest, InputDigest: attempt.Job.InputArtifact,
		ServingRevisionDigest: attempt.Provenance.ServingRevisionDigest,
		BundlePath:            workerArtifactRoot + "/bundle", InputPath: workerArtifactRoot + "/input.safetensors",
		OutputDirectory: workerOutputDirectory, BundleSigningKeyID: model.BundleSigningKeyID,
		SchedulerSigningKeyID: l.manifestSigningKey.KeyID(), BundleDownloadCapability: bundleCapability,
		InputDownloadCapability: inputCapability, CompletionCapability: attempt.CompletionCapability,
		CompletionSigningPrivateKey: attempt.CompletionSigningPrivateKey,
		FencingToken:                attempt.Job.FencingToken, Seed: attempt.Job.Seed, NumSamples: 1,
		NumSteps: attempt.Job.DiffusionSteps, Device: "cuda", SchemaVersion: "v1alpha1",
	}
	unsigned, err := canonicalUnsignedManifest(manifest)
	if err != nil {
		return workerJobManifest{}, err
	}
	manifest.ManifestSignature = base64.StdEncoding.EncodeToString(l.manifestSigningKey.Sign(unsigned))
	return manifest, nil
}

func canonicalUnsignedManifest(manifest workerJobManifest) ([]byte, error) {
	value := map[string]any{
		"bundle_archive_digest":          manifest.BundleArchiveDigest,
		"bundle_download_capability":     manifest.BundleDownloadCapability,
		"bundle_manifest_digest":         manifest.BundleManifestDigest,
		"bundle_path":                    manifest.BundlePath,
		"bundle_signing_key_id":          manifest.BundleSigningKeyID,
		"completion_capability":          manifest.CompletionCapability,
		"completion_signing_private_key": manifest.CompletionSigningPrivateKey,
		"device":                         manifest.Device,
		"fencing_token":                  manifest.FencingToken,
		"input_digest":                   manifest.InputDigest,
		"input_download_capability":      manifest.InputDownloadCapability,
		"input_path":                     manifest.InputPath,
		"job_id":                         manifest.JobID,
		"model_digest":                   manifest.ModelDigest,
		"num_samples":                    manifest.NumSamples,
		"num_steps":                      manifest.NumSteps,
		"output_directory":               manifest.OutputDirectory,
		"project_id":                     manifest.ProjectID,
		"scheduler_signing_key_id":       manifest.SchedulerSigningKeyID,
		"schema_version":                 manifest.SchemaVersion,
		"seed":                           manifest.Seed,
		"serving_revision_digest":        manifest.ServingRevisionDigest,
		"tenant_id":                      manifest.TenantID,
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("marshal canonical worker manifest: %w", err)
	}
	return encoded, nil
}

func (l *KubernetesAttemptLauncher) manifestSecret(baseName string, owner map[string]any, manifestJSON []byte) map[string]any {
	return map[string]any{
		"apiVersion": "v1", "kind": "Secret",
		"metadata": map[string]any{
			"name": baseName + "-manifest", "namespace": l.config.Namespace,
			"labels":          map[string]string{"app.kubernetes.io/name": "inference-attempt-manifest"},
			"ownerReferences": []any{owner},
		},
		"immutable": true, "type": "Opaque",
		"data": map[string]string{"job.json": base64.StdEncoding.EncodeToString(manifestJSON)},
	}
}

func (l *KubernetesAttemptLauncher) resultPVC(baseName string) map[string]any {
	return map[string]any{
		"apiVersion": "v1", "kind": "PersistentVolumeClaim",
		"metadata": map[string]any{
			"name": baseName + "-result", "namespace": l.config.Namespace,
			"labels":      map[string]string{"app.kubernetes.io/name": "inference-result"},
			"annotations": map[string]string{"mindclade.dev/retention": "result retained after JobSet TTL; explicit cleanup required"},
		},
		"spec": map[string]any{
			"accessModes": []string{"ReadWriteOncePod"}, "storageClassName": l.config.ResultStorageClass,
			"volumeMode": "Filesystem",
			"resources":  map[string]any{"requests": map[string]string{"storage": l.config.ResultStorageRequest}},
		},
	}
}

func (l *KubernetesAttemptLauncher) jobSet(attempt AttemptLease, baseName string) map[string]any {
	containerSecurity := map[string]any{
		"allowPrivilegeEscalation": false, "readOnlyRootFilesystem": true,
		"capabilities": map[string]any{"drop": []string{"ALL"}},
		"runAsNonRoot": true, "runAsUser": int64(65532), "runAsGroup": int64(65532),
	}
	commonMounts := []any{
		map[string]any{"name": "scratch", "mountPath": "/tmp"},
		map[string]any{"name": "job-manifest", "mountPath": "/var/run/mindclade", "readOnly": true},
		map[string]any{"name": "signing-keys", "mountPath": "/var/run/mindclade-signing", "readOnly": true},
	}
	return map[string]any{
		"apiVersion": "jobset.x-k8s.io/v1alpha2", "kind": "JobSet",
		"metadata": map[string]any{
			"name": baseName, "namespace": l.config.Namespace,
			"labels": map[string]string{
				"app.kubernetes.io/name": "inference-attempt", "mindclade.dev/attempt": baseName,
				"mindclade.dev/managed-by": "control-plane",
			},
			"annotations": map[string]string{
				"mindclade.dev/job-id":               attempt.Job.ID,
				"mindclade.dev/resource-incarnation": l.config.ResourceIncarnation,
				"mindclade.dev/tenant-id":            attempt.Job.Scope.TenantID,
				"mindclade.dev/project-id":           attempt.Job.Scope.ProjectID,
				"mindclade.dev/fencing-token":        strconv.FormatInt(attempt.Job.FencingToken, 10),
				"mindclade.dev/queue-deadline":       strconv.FormatInt(l.config.QueueDeadlineSeconds, 10),
				"mindclade.dev/startup-deadline":     strconv.FormatInt(l.config.StartupDeadlineSeconds, 10),
				"mindclade.dev/active-deadline":      strconv.FormatInt(l.config.ActiveDeadlineSeconds, 10),
			},
		},
		"spec": map[string]any{
			"suspend": true, "ttlSecondsAfterFinished": l.config.TTLSecondsAfterFinish,
			"failurePolicy": map[string]any{"maxRestarts": int64(0)},
			"replicatedJobs": []any{map[string]any{
				"name": "worker", "replicas": int64(1),
				"template": map[string]any{"spec": map[string]any{
					"backoffLimit": int64(0), "activeDeadlineSeconds": l.config.ActiveDeadlineSeconds,
					"template": map[string]any{
						"metadata": map[string]any{"labels": map[string]string{"app.kubernetes.io/name": "inference-worker"}},
						"spec": map[string]any{
							"restartPolicy": "Never", "serviceAccountName": l.config.WorkerServiceAccount,
							"automountServiceAccountToken": false, "enableServiceLinks": false,
							"terminationGracePeriodSeconds": int64(30),
							"securityContext": map[string]any{
								"runAsNonRoot": true, "runAsUser": int64(65532), "runAsGroup": int64(65532),
								"fsGroup": int64(65532), "fsGroupChangePolicy": "OnRootMismatch",
								"seccompProfile": map[string]string{"type": "RuntimeDefault"},
							},
							"initContainers": []any{map[string]any{
								"name": "artifact-stager", "image": l.config.WorkerImage,
								"args": []string{
									"stage", "--job-manifest", "/var/run/mindclade/job.json",
									"--trusted-keyring", "/var/run/mindclade-signing/keyring.json",
									"--artifact-root", workerArtifactRoot,
									"--artifact-proxy-url", strings.TrimRight(l.config.ArtifactProxyURL, "/"),
								},
								"resources": map[string]any{
									"requests": map[string]string{"cpu": "500m", "memory": "512Mi", "ephemeral-storage": "1Gi"},
									"limits":   map[string]string{"cpu": "2", "memory": "2Gi", "ephemeral-storage": "2Gi"},
								},
								"securityContext": containerSecurity,
								"volumeMounts":    append(append([]any{}, commonMounts...), map[string]any{"name": "job-artifacts", "mountPath": workerArtifactRoot}),
							}},
							"containers": []any{map[string]any{
								"name": "worker", "image": l.config.WorkerImage,
								"args": []string{
									"run", "--job-manifest", "/var/run/mindclade/job.json",
									"--trusted-keyring", "/var/run/mindclade-signing/keyring.json",
									"--artifact-root", workerArtifactRoot, "--result-root", workerResultRoot,
									"--control-plane-url", strings.TrimRight(l.config.ControlPlaneURL, "/"),
									"--artifact-proxy-url", strings.TrimRight(l.config.ArtifactProxyURL, "/"),
								},
								"resources": map[string]any{
									"requests": map[string]string{"cpu": "4", "memory": "16Gi", "ephemeral-storage": "1Gi", "nvidia.com/gpu": "1"},
									"limits":   map[string]string{"cpu": "8", "memory": "32Gi", "ephemeral-storage": "2Gi", "nvidia.com/gpu": "1"},
								},
								"securityContext": containerSecurity,
								"volumeMounts": append(append(append([]any{}, commonMounts...),
									map[string]any{"name": "job-artifacts", "mountPath": workerArtifactRoot, "readOnly": true}),
									map[string]any{"name": "job-results", "mountPath": workerResultRoot}),
							}},
							"volumes": []any{
								map[string]any{"name": "scratch", "emptyDir": map[string]any{"sizeLimit": "2Gi"}},
								map[string]any{"name": "job-manifest", "secret": map[string]any{
									"secretName": baseName + "-manifest", "defaultMode": int64(256),
									"items": []any{map[string]any{"key": "job.json", "path": "job.json", "mode": int64(256)}},
								}},
								map[string]any{"name": "job-artifacts", "emptyDir": map[string]any{"sizeLimit": l.config.ArtifactScratchLimit}},
								map[string]any{"name": "job-results", "persistentVolumeClaim": map[string]any{"claimName": baseName + "-result", "readOnly": false}},
								map[string]any{"name": "signing-keys", "secret": map[string]any{"secretName": l.config.WorkerTrustSecret, "defaultMode": int64(292)}},
							},
						},
					},
				}},
			}},
		},
	}
}

func attemptResourceBase(resourceIncarnation, jobID string, fencingToken int64) string {
	normalized := strings.Builder{}
	for _, character := range strings.ToLower(jobID) {
		if (character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') {
			normalized.WriteRune(character)
		} else if normalized.Len() > 0 && !strings.HasSuffix(normalized.String(), "-") {
			normalized.WriteByte('-')
		}
	}
	stem := strings.Trim(normalized.String(), "-")
	if stem == "" {
		stem = "job"
	}
	digest := sha256.Sum256([]byte(resourceIncarnation + "\x00" + jobID + "\x00" + strconv.FormatInt(fencingToken, 10)))
	suffix := "-" + hex.EncodeToString(digest[:16])
	// The immutable manifest Secret adds "-manifest" (9 characters), so the
	// base must remain at most 54 characters to satisfy the DNS-label limit.
	maximumStem := 54 - len("mc-") - len(suffix)
	if len(stem) > maximumStem {
		stem = strings.TrimRight(stem[:maximumStem], "-")
	}
	return "mc-" + stem + suffix
}

func attemptPVCName(resourceIncarnation, jobID string, fencingToken int64) string {
	return attemptResourceBase(resourceIncarnation, jobID, fencingToken) + "-result"
}
