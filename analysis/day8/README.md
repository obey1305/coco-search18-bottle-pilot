# Day 8 — preliminary descriptive figure
Completed by Codex on 11 October 2026, ahead of the scheduled 13 October task. This is a descriptive result, not a significance claim.

[Open the figure](descriptive_contrast.png) · [PDF export](descriptive_contrast.pdf) · [Ten plotted observations](observations.csv) · [Counts](counts.csv)

## What was run
Applied the Day 5 colour and Day 6 orientation/luminance computations to all three fixed training images. Used the original-image transforms established in the semantic extension, correcting the older one-pixel padding estimates. Saved full-resolution maps and validity masks for every image, including the image with no successful matches.

The primary comparison uses Day 7's fixed semantic-and-spatial matching, with a 16-pixel target-mask exclusion margin. Each chosen landing has eight sampled alternatives of the same category, comparable instance area where applicable, similar gaze/centre/target-mask distances, and the same prior-view status. Alternatives were selected without using contrast values. No matches were changed after examining these plots.

## Reading the figure
Each dot or diamond is one recorded landing, not one participant or one alternative. The same ten landings appear in each panel.

- x: Euclidean distance from the original gaze position to the recorded next fixation, in stimulus pixels; not degrees of visual angle.
- y: feature at the chosen landing minus the arithmetic mean of that feature at its eight alternatives.
- Above zero: chosen contrast exceeds the alternatives' mean. Below zero: chosen contrast is lower. Zero: equal values.
- Colour, orientation and luminance have different units and separate y-axis scales. Do not compare their raw magnitudes across panels.

For example, if chosen contrast is 6 and the eight alternatives average 4, the plotted difference is +2. This sign describes a local comparison; it does not show why the eyes moved there.

## Observations and counts
| Image | Day 4 eligible movements attempted | Successful semantic matches | Insufficient alternatives | Unlabelled chosen locations |
|---|---:|---:|---:|---:|
| 496294, restaurant | 1 | 1 | 0 | 0 |
| 522129, two laptops | 7 | 0 | 2 | 5 |
| 295317, office | 9 | 9 | 0 | 0 |
| Total | 17 | 10 | 2 | 5 |

There are ten plotted landings from eight distinct participant IDs, using 80 alternatives (90 feature rows including chosen locations). All three images were processed, but only two contribute plotted observations. The image with no matches is not represented by invented zero-valued observations.

For simple descriptive sign counts, chosen contrast exceeds the alternatives' mean at 3/10 landings for colour, 5/10 for orientation and 3/10 for luminance. These are not significance tests or evidence of absence/presence of a population effect. Exact values are in observations.csv.

## Computation details and checks
- Retained the existing feature definitions: CIELAB a*/b* centre-surround colour difference, four-Gabor response-vector difference, and absolute mean-L* difference.
- Used 64-pixel complete support for colour/luminance and 88 pixels for orientation. Every sampled row is valid for all three features.
- Feature values use the nearest integer stimulus pixel, floor(coordinate+0.5), consistent with semantic lookup. The x-axis retains the original floating-point gaze distance. The sampled pixel coordinates are exported alongside original coordinates.
- Checked input image SHA256 values against the recorded source images.
- Verified all 270 candidate-feature values (90 rows × 3 features) against direct disk/ring averaging, with maximum error below 1e-9.
- Also checked a fixed central point on all three images, including direct Gabor convolution for the image with no matched observations.
- Independently recomputed all 30 plotted differences from exported candidate values using math.fsum; confirmed eight unique controls per decision and x-axis values.
- The full run passed with RuntimeWarnings treated as errors. Visually inspected the PNG for all observations, counts, units, zero reference lines and readable labels; the PDF is exported from the same figure.

The independent checks validate computation, not the hypothesis that contrast predicts fixation selection.

## Limits
Nine of ten observations come from one image. Some participants contribute repeated observations, and locations/alternatives within images are related. This tiny selected pilot cannot establish a trend with distance, a population association, causal attention capture or a neural mechanism. No regression, correlation test, p-value, confidence interval or validation-set claim is supplied.

The input remains the Day 4 bbox-filtered set of 17 movements, further filtered by semantic matching; it is not every possible mask-eligible raw movement. Missing semantic labels and unsuccessful matching select which observations survive. Source annotation boundaries are approximate. Further evaluation needs more eligible images, participant/image dependence handling and held-out data.

## Files and rerun
- describe.py: reproducible three-image computation, exported tables and figure.
- *_features.npz: colour/orientation/luminance arrays with their masks and refined bounds.
- candidate_features.csv: all 90 locations, semantic metadata and sampled feature values.
- observations.csv: one row per landing with chosen, mean-control and difference values.
- counts.csv, verification.json, independent_checks.txt, run.log: audit evidence.

From the packet directory:

```sh
.venv/bin/python -W error::RuntimeWarning day8/describe.py > day8/run.log
```

The ten observed landings, and each set of eight alternatives, are fixed by the preceding semantic-matching stage. No new downloads are required.

## Your next step
Open the figure and find the dashed zero line. A point above that line means higher contrast at the chosen location than the alternatives' average; below means lower. The plot is a description of these observations, not proof of a general effect.
