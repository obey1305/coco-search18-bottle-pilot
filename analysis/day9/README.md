# Day 9 — robustness and limitations
Completed by Codex on 12 October 2026, ahead of the scheduled 14 October task. [Detailed robustness table](tables.md) · [All paired observations](observations.csv) · [Verification](verification.json)

## Short result
The descriptive pattern depends on patch size and, for some comparisons, on which alternatives are sampled. Do not call it a robust gaze-prediction effect.

| Setting | Colour: positive differences | Orientation: positive differences | Luminance: positive differences |
|---|---:|---:|---:|
| Original patch and seed | 3/10 | 5/10 | 3/10 |
| Smaller patch only | 4/10 | 6/10 | 4/10 |
| New control seed only | 3/10 | 5/10 | 5/10 |
| Smaller patch + new seed | 3/10 | 5/10 | 5/10 |

A positive difference means contrast at the chosen landing exceeds the mean of its eight alternatives. Every row uses the same ten landings. Equal counts can hide different observations changing sign; inspect the detailed paired table rather than judging stability by totals alone.

## What changed
The original centre radius is 16 pixels with a 32–64 pixel surround. The smaller centre radius is 8 pixels with a 16–32 pixel surround. All disk/ring boundaries remain inclusive. Gabor wavelength, sigma and kernel support remain 16, 8 and 24 pixels respectively: this isolates the averaging-patch scale, not the Gabor filter scale.

Original control seed base: 20261006. Repeat: 20261007, with the same SHA256 decision-ID addition and eight uniform samples without replacement. The new seed changes 56 of the 80 control positions, summed within decision sets. Eligibility, calipers, categories, area limits, mask dilation and previously-viewed rules remain unchanged; decision logs and pool sizes are exactly identical.

Although the smaller averaging patch requires less support (32 pixels for colour/lightness and 56 with the unchanged Gabor kernel), this paired comparison conservatively retains the existing 88-pixel support requirement and the same eligible candidate pools. It does not add newly eligible near-edge locations or new landings. Primary target-mask dilation remains 16 pixels.

## Does the pattern change?
- Smaller patch alone: three of ten differences change sign for each of colour, orientation and luminance. Orientation's median difference moves from −0.0191 to +0.0296.
- New seed alone: no colour/orientation sign changes, but two luminance sign changes. Magnitudes still change.
- Both changes: two colour, four orientation and two luminance differences change sign relative to baseline.
- The descriptive association with gaze distance also changes. Luminance's rank association changes from +0.091 at baseline to −0.588 with the smaller patch, and −0.661 with both changes. Colour's association remains positive across the four settings but varies from +0.321 to +0.661; orientation remains negative, from −0.248 to −0.115. These are Spearman rank summaries of ten dependent, selected observations, not significance or population claims. No p-values are calculated.

The orientation difference has a near-zero baseline median; a median sign change should not itself be presented as a large or significant effect.

## Limitations table
| Check or limitation | Status | Meaning |
|---|---|---|
| Another patch size | Done: radius 8 vs 16 | Several individual differences change sign. |
| Another sampling seed | Done: base 20261007 vs 20261006 | Luminance changes sign at two landings. One extra seed is not an exhaustive stability assessment. |
| Combined change | Done | Separates single changes from their combination. |
| Additional unseen images | Not tested in this step | Generalization remains untested; the optional unseen-image extension was not included. |
| Image coverage | Same 3 processed images; 2 contribute observations | Nine of ten landings come from one office image; the two-laptop image has no semantic matches. |
| Independence | 10 landings from 8 participants | Repeated observers and shared image context preclude treating all points as independent. |
| Selection | Existing 17 bbox-filtered movements, then semantic matching | Missing labels and unsuccessful matches affect which landings remain. |
| Filter scale | Held fixed | This does not test alternative Gabor wavelengths or sigmas. |
| Mechanism or significance | Not tested | No causal, neural, significance or held-out prediction claim. |

## Computation and validation
Reused the Day 8 full-resolution baseline maps, which use corrected padding geometry. Recomputed smaller-patch features for the same three photographs and sampled the same original and new-control locations. Feature sampling uses nearest stimulus pixel; distance retains original floating-point gaze coordinates.

- The baseline exactly reproduces every Day 8 plotted difference to within 1e-12.
- Original and repeated runs have identical decision IDs, chosen positions, eligibility and candidate pool sizes.
- Every set has eight unique controls; 56 alternative positions change under the new seed.
- Smaller-patch sampled values agree with direct disk/ring averaging within 1e-9; maximum observed error is below 1.3e-13. The image with no matched observations has no sampled-value comparisons and its zero error entry is not evidence of such a comparison.
- Exports contain 360 candidate-feature rows and 40 scenario-observation rows (ten per setting). The repeated rows are not new independent data.
- Synthetic smaller-patch uniform-field and known centre-change checks, plus independent recomputation of all 120 differences, are recorded in independent_checks.txt.

## Files and rerun
- robustness.py computes the paired sensitivity analysis.
- seed_repeat/semantic_match.py reruns the unchanged matching criteria with the new seed; its inputs and outputs are preserved.
- candidate_features.csv, observations.csv, summary.csv and seed_overlap.csv provide the values behind the tables.
- tables.md provides medians, sign changes and descriptive rank associations.

From the pilot packet directory:

```sh
.venv/bin/python -W error::RuntimeWarning day9/seed_repeat/semantic_match.py > day9/seed_repeat/run.log
.venv/bin/python -W error::RuntimeWarning day9/robustness.py > day9/run.log
```

Your learning note: changing patch size or the sampled alternatives can change this pilot's descriptive result. No unseen-image conclusion follows from these repeats.
