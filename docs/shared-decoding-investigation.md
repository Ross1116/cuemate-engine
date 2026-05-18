# Shared Decoding Investigation

This note captures the current state of audio decode ownership after the Milestone 5 cleanup pass, as of 2026-04-12 at commit `80e3800`.

## Current state

- As of 2026-04-12 at commit `80e3800`, the repository does **not** track a shared-decoding refactor as an active milestone item.
- As of 2026-04-12 at commit `80e3800`, decode-heavy paths still live close to the service or runtime that owns the corresponding inference flow.
- The clearest example is the Essentia semantic service, which keeps its excerpt/window decode helpers local because they are tightly coupled to semantic-window selection and cache behavior.

## Observed hotspots

- `docker/essentia_semantics/service.py`
  - middle excerpt decode
  - ratio-window decode
  - peak excerpt decode
- Python analysis/runtime paths also perform their own decode or preprocessing steps when they need different invariants, caching, or error handling.

## Assessment

- The current duplication is partly structural, but it is also partly intentional specialization.
- As of 2026-04-12 at commit `80e3800`, a shared decode abstraction would only be worth the complexity if one of these becomes true:
  - decode cost becomes a measured top-level bottleneck across multiple services
  - multiple paths need the same excerpt/window semantics and are drifting
  - bug fixes repeatedly require editing the same decode behavior in several places

## Recommendation

- Keep service-local decoding for now.
- Revisit a shared decode layer only after collecting concrete timing or maintenance evidence from production-like runs or benchmark work.
- If revisited later, start with a narrow shared helper for excerpt/window selection semantics rather than a global “all decoding goes here” abstraction.
