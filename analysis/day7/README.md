# Day 7 — spatial matching only

> Update, 10 October 2026: semantic annotations have now been obtained and checked in the separate [semantic extension](../day7_semantic/README.md). This file preserves the original spatial-only analysis and its access status at the time.
Prepared and verified by Codex on 10 October 2026, ahead of the planned 12 October task. The user has not yet personally reviewed the matches.

[Small chosen-versus-alternative table and all failures](tables.md) · [All candidate rows](chosen_vs_alternatives.csv) · [Decision log](decision_log.csv) · [Distance-band checks](balance_by_distance.csv)

## Result
Attempted all 17 Day 4 retained pre-target movements from the three fixed bottle training images. Fifteen decisions have eight alternatives each (15 chosen + 120 alternative rows). Two decisions failed the minimum of eight alternatives. No decision failed the additional feature-support check.

This is spatial matching only. Semantic categories, object sizes and segmentation-based target exclusions have not been verified or matched. “Non-target” here means outside the single released target bounding box, not verified outside every target-category object.

## What “chosen” and “alternative” mean
Chosen is the next recorded fixation. Alternatives are sampled locations in the same image that satisfy the comparison rules. They are not observed eye movements, proof of options considered by the participant, or locations that nobody ever looked at. Future fixations are not used to exclude alternatives.

The first example is image 496294, participant 5, movement 0→1. The chosen landing is 245.8 pixels from the starting gaze position. Its alternatives have similar distances from that same gaze, the display centre and the target box. The full small table records exact comparisons.

## Fixed rules
- Reuse the 17 Day 4 eligible original adjacent pairs; do not delete/reconnect fixation indices.
- Each comparison shares image, task, participant, origin fixation and landing index.
- Candidate lattice: x=0,8,…1672 and y=0,8,…1048 in full stimulus coordinates, then filter to actual image content.
- Chosen and alternatives need complete feature support: radius 88 pixels, inherited from Day 6's Gabor-plus-surround requirement. This intentionally strengthens the starter packet's 64-pixel rule so all three features can later be used. No feature values influence sampling.
- Alternatives lie outside the released target box and at least 32 pixels from both current gaze and the chosen landing.
- Absolute difference from the chosen point's gaze distance <=16 pixels; centre distance <=32 pixels; target-box distance <=32 pixels.
- Display centre is (840,525). Target-box distance is the shortest Euclidean distance to the closed rectangle, zero inside. It is a proxy for mask distance.
- Match previously-viewed status: within 32 pixels of any recorded fixation up to and including the current origin. All history used here is finite; no future fixations enter this status.
- Require at least eight candidates. Sample eight uniformly without replacement using NumPy default_rng, seeded by (20261006 + first 16 hexadecimal SHA256 digits of the decision ID) modulo 2^64. Decision ID includes split, task, image, participant and original adjacent indices. There is no mutual separation rule among alternatives beyond unique lattice points.
- Do not widen tolerances after seeing failures. Do not reuse failed decisions with fewer controls.

The fixed seed and rule make the sample reproducible in this saved NumPy environment. Matching is approximate within the declared tolerances, not exact equality.

## Failures and balance
Both failures occur in image 522129 at movement 0→1: participant 2 has six eligible candidates; participant 8 has two. They are retained in decision_log.csv with status failed and no candidate set is emitted for them.

| Gaze-distance band | Attempted | Matched | Failed | Mean absolute gaze difference | Centre difference | Target-box difference |
|---|---:|---:|---:|---:|---:|---:|
| Below 100 px | 0 | 0 | 0 | — | — | — |
| 100 to below 300 px | 8 | 6 | 2 | 7.67 px | 10.12 px | 16.82 px |
| 300 px and above | 9 | 9 | 0 | 6.88 px | 13.20 px | 17.44 px |

Differences summarize sampled alternatives versus their own chosen landing. These are descriptive checks on a tiny pilot; matching failure can select which decisions remain. No inference about contrast effects has been made.

## Semantic-annotation access check
No panoptic masks, category maps or instance annotations were found in the pilot outputs. The official COCO panoptic archive endpoint responded HTTP 200 to a HEAD request on 10 October 2026, with Content-Length 860,725,834 bytes and byte-range support:
http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip

The archive is publicly reachable; it is NOT accurate to call the source unavailable. Its contents were not downloaded for this spatial-only step, and per-image coverage, category labels, original dimensions and transforms into the padded Search18 stimuli have not been verified. A reachable archive alone does not establish usable aligned semantic labels.

The [official COCO panoptic API documentation](https://github.com/cocodataset/panopticapi) describes per-image segmentation PNGs and associated category information. Strict semantic matching remains pending until these are obtained and aligned, masks are visually checked, all relevant target instances are excluded, and category/area matching is implemented. The task label “bottle” does not identify the object category at a distractor location.

## Verification
- Known rectangle-distance reference cases pass, including a diagonal 3–4–5 case.
- Every retained input remains an original adjacent pair.
- Every matched set has exactly one chosen row and eight unique alternatives.
- Every alternative satisfies support, target-box exclusion, separation, all three calipers and prior-view status.
- Row counts reconcile to 15 × 9 = 135.
- All 135 exported distances were independently recalculated with Python math.hypot and agree within 1e-9.
- Rerunning produced byte-identical candidate, decision and balance CSVs.

## Rerun
From the packet directory:

```sh
.venv/bin/python -W error::RuntimeWarning day7/match_alternatives.py > day7/run.log
```

The script reads Day 4 results and raw train_split1 data. It creates comparison tables only; it does not fit a model or claim that contrast predicts gaze.
