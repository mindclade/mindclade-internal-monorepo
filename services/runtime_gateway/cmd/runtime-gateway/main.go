package main

import (
	"log"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"time"

	runtimegateway "github.com/mindclade/mindclade-internal-monorepo/services/runtime_gateway"
)

func required(name string) string {
	value := os.Getenv(name)
	if value == "" {
		log.Fatalf("%s is required", name)
	}
	return value
}

func main() {
	publicKey, err := os.ReadFile(required("MINDCLADE_OIDC_PUBLIC_KEY_FILE"))
	if err != nil {
		log.Fatal(err)
	}
	verifier, err := runtimegateway.NewOIDCVerifier(
		required("MINDCLADE_OIDC_ISSUER"), required("MINDCLADE_OIDC_AUDIENCE"),
		required("MINDCLADE_OIDC_KEY_ID"), publicKey,
	)
	if err != nil {
		log.Fatal(err)
	}
	upstream, err := url.Parse(required("MINDCLADE_CONTROL_PLANE_URL"))
	if err != nil {
		log.Fatal(err)
	}
	internalSecret, err := os.ReadFile(required("MINDCLADE_INTERNAL_IDENTITY_SECRET_FILE"))
	if err != nil {
		log.Fatal(err)
	}
	identitySigner, err := runtimegateway.NewInternalIdentitySigner(internalSecret)
	if err != nil {
		log.Fatal(err)
	}
	address := os.Getenv("MINDCLADE_RUNTIME_GATEWAY_ADDRESS")
	if address == "" {
		address = "127.0.0.1:8080"
	}
	server := &http.Server{
		Addr:              address,
		Handler:           runtimegateway.New(verifier, upstream, nil, slog.Default(), identitySigner),
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       60 * time.Second,
	}
	log.Printf("runtime gateway listening on %s", address)
	log.Fatal(server.ListenAndServe())
}
