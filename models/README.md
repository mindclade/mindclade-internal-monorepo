# mindclade-models

`mindclade-models` contains internal, random-initialized model definitions and
integrity-checked local serialization. The first reference family is CladeFold
Q0, a trainable architecture used to exercise model, training, and serving
contracts. It ships no weights and makes no biological, clinical, or scientific
capability claim.

The package is a PEP 420 member of the `mindclade` namespace. A fast local check
is `PYTHONPATH=models/src python -m pytest models/tests`.
