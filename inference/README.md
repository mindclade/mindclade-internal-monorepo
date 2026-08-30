# mindclade-inference

`mindclade-inference` is the internal, typed inference runtime for Mindclade model
families. It validates tensor-only requests, resolves immutable model identities,
batches compatible work, executes eager or qualified compiled variants, samples
deterministically, and commits integrity-checked result artifacts.

The initial `0.1.0` contract is `v1alpha1`. It accepts local tensors and immutable
`sha256` model identities; it does not parse FASTA, SMILES, or structure files,
download models, or claim scientific/clinical fitness.

Fold outputs carry canonical `sample_seeds` with shape `[B, S]`. Postprocessing
creates one candidate per sample with `batch_seeds[B]`, preserving the exact seed
for every coordinate row. For batch size one, `candidate.seed` remains the scalar
convenience API and a flat `[S]` seed sequence remains accepted. Request-envelope
grouping is not execution-time tensor collation; any future co-batched executor must
preserve each request's row-to-seed mapping.

## Development

From the repository root, use the workspace environment and run:

```console
pytest inference/tests
```

The default tests are CPU-only, make no network calls, and use temporary local
artifact stores. Accelerator and compiled variants require a separately recorded
qualification before selection.
