# Runtime gateway

The gateway validates RS256 OIDC tokens, matches tenant/project path scope to claims,
removes the bearer token, and overwrites all internal identity headers. Logs contain a
route template and identifiers, never request or response bodies. Public keys are
mounted configuration; the process never contains a development authentication mode.
