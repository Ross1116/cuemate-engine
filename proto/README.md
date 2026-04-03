# Scoring Contract

The protobuf contract is the single language-neutral boundary between the Python analysis plane and the Go decision plane.

## Canonical file

- `proto/djengine/scoring/v1/scoring.proto`

## Contract rules from the setup baseline

- Python owns the authoritative scoring runtime
- Go may orchestrate and shape API responses, but it does not reimplement scoring semantics
- Compatibility is explicit through `analysis_signature`, `scoring_contract_id`, and `config_signature`
- `proto/` is the protobuf module root for repo tooling

The schema is intentionally committed before application code so both planes can build against one contract from the first implementation commit.
