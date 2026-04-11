# Local Feedback Loop Workflow

This is the shipped local-first Milestone 5 operator flow for recommendation outcome capture and per-playlist feedback tuning.

## Services

Start the scorer and API:

```powershell
python -m cuemate_analysis serve-scoring
go run ./go/cmd/apiserver
```

## Record recommendation outcomes

1. Request recommendations:

```powershell
$rec = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/recommendations `
  -ContentType "application/json" `
  -Body '{"playlist_name":"My Playlist","current_track_id":"trk_current","target":"maintain"}'
```

2. Choose a returned candidate and record the played outcome:

```powershell
$eventId = $rec.meta.recommendation_event_id
$chosen = $rec.lanes.maintain.items[0].track_id

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/events/played `
  -ContentType "application/json" `
  -Body "{`"recommendation_event_id`":`"$eventId`",`"chosen_track_id`":`"$chosen`"}"
```

This records the recommendation outcome and queues a pending `feedback_tuning_job` for the playlist.

## Inspect feedback state

Read the current playlist summary:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/feedback/summary `
  -ContentType "application/json" `
  -Body '{"playlist_name":"My Playlist"}'
```

Or from Python:

```powershell
python -m cuemate_analysis feedback-summary --playlist "My Playlist"
python -m cuemate_analysis inspect-scoring-weights --playlist "My Playlist"
```

The active weight precedence is:

1. `feedback_tuned_weights`
2. `adapted_weights`
3. static scoring weights

## Apply tuned weights

Preview tuning without writing:

```powershell
python -m cuemate_analysis feedback-tune --playlist "My Playlist" --preview-only
```

Run the worker that claims queued jobs and applies tuned weights when thresholds are met:

```powershell
python -m cuemate_analysis run-feedback-worker --limit 10
```

Automatic apply requires:

- at least 20 contributory events
- at least 40 pairwise comparisons
- at least 5 new contributory events since the last successful tune

## Acceptance checklist

- recommendations return `meta.recommendation_event_id`
- `/events/played` records the chosen outcome successfully
- `feedback-summary` reports updated event counts
- `run-feedback-worker` claims the playlist job
- `playlist_stats.feedback_tuned_weights` becomes non-null once thresholds are met
- later `/recommendations` responses report `meta.weight_adaptation.mode = "feedback_tuned_weights"`
