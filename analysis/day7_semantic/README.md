# Day 7 extension — semantic and spatial matching
Completed by Codex on 10 October 2026. The previous spatial-only outputs are preserved separately.

[Matched table and every failure](tables.md) · [All 90 primary rows](chosen_vs_alternatives.csv) · [Independent checks](independent_checks.txt)

## Result
Of the same 17 Day 4 eligible movements, 10 now have eight semantic-and-spatial alternatives each (10 chosen + 80 alternatives). Five chosen landings are in panoptic void/unlabelled regions and are excluded rather than assigned a guessed category. Two fail because only five semantic alternatives qualify; the eight-alternative requirement was not relaxed.

The ten matched landings have dataset categories: dining table (1), tv (1), laptop (4), paper-merged (2), wall-other-merged (2). “tv” is the dataset's label for the monitor in the office photograph. Labels are taken from COCO annotations, not inferred by this assistant or by a new object detector.

This is a feasibility extension of the existing 17-movement sample, which was already filtered using released boxes on Day 4. It is not a reconstruction of every mask-eligible movement in the raw trials. No model was fitted and no gaze-prediction effect was estimated.

## Semantic data and visual alignment
Downloaded the official COCO panoptic archive and extracted the three train2017 masks plus their segment/category metadata. Each original mask's segment IDs and pixel counts exactly match its JSON annotations. Selected originals, masks, metadata, transforms and source hashes are retained here; the large full archives are scratch files in work/semantic.

Official sources: [panoptic API and format documentation](https://github.com/cocodataset/panopticapi), [annotation archive](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip). Original photographs were retrieved from the COCO train2017 image endpoint, using the image IDs in the annotation metadata. source_manifest.json records SHA256 hashes.

Compared resized original photographs with the downloaded Search18 stimuli. Among floor/round/ceil aspect-preserving dimensions, centred offsets and bilinear/bicubic/Lanczos resampling, the lowest mean absolute RGB errors were obtained with bilinear resampling:

| Image | Original size | Resized size | Padding offset (left, top) | Mean RGB absolute error, 0–255 scale |
|---|---|---|---|---:|
| 496294 | 640 × 368 | 1680 × 966 | (0,42) | 0.8881 |
| 522129 | 640 × 480 | 1400 × 1050 | (140,0) | 1.8309 |
| 295317 | 640 × 480 | 1400 × 1050 | (140,0) | 0.7067 |

This provides numerical evidence for the geometric mapping, with remaining colour error consistent with image resampling/recompression. It is not evidence of subpixel eye-tracker accuracy. Raw transforms and candidate errors are saved. Integer segment IDs were then resized by nearest-neighbour, never by blending mask colours, and placed at those offsets.

Visually inspected all three overlay exports: the bottle contours follow the visible bottles; laptop, sandwich, monitor and broad scene regions also align. Annotation shapes remain approximate and some scene parts are unlabelled. Each of these three masks contains one annotated bottle instance. Target exclusion uses every annotated bottle instance present, but this cannot guarantee that source annotations contain every possible object.

- [Restaurant overlay](496294_semantic_overlay.png)
- [Two-laptop overlay](522129_semantic_overlay.png)
- [Office overlay](295317_semantic_overlay.png)

Cyan marks the bottle silhouette; yellow adds the primary 16-pixel exclusion margin. Region colours are illustrative segment colours, not a shared category legend.

### Padding correction
Day 4 inferred content edges from near-black pixels. JPEG artifacts made two boundaries appear one pixel larger: the restaurant content starts at y=42, not 41, and the two-laptop content starts at x=140, not 139. This extension uses the original-image transforms above. Earlier exported feature maps and spatial tables remain historical outputs based on their documented estimates; do not reuse their one-pixel fringe for later modelling without updating their validity masks. None of the 17 chosen points fails the refined 88-pixel support rule.

## Matching rules
The spatial calipers remain ±16 pixels for gaze distance and ±32 pixels for image-centre and target proximity. There are eight unique sampled alternatives per successful decision on the same 8-pixel full-stimulus lattice. Alternatives remain at least 32 pixels from the chosen point and current gaze. Both chosen and alternatives need the full 88-pixel feature support, using the refined content rectangle.

New semantic requirements:
- Same verified dataset category at the chosen and alternative locations.
- For a COCO “thing” instance, alternative instance area must be between one-half and twice the chosen instance's mapped mask area. Sampling another point on the same instance is allowed.
- For “stuff” categories such as wall-other-merged and paper-merged, require the same category without imposing an instance-area rule.
- Void and crowd-labelled locations are excluded. Category lookup rounds coordinates to the nearest stimulus pixel (floor(x+0.5)); gaze and centre distances still use original floating-point coordinates.
- A chosen point whose 3×3 pixel neighbourhood crosses a semantic-category boundary is excluded as a conservative one-pixel uncertainty guard. No current primary point was removed by this guard. Candidate labels are evaluated at their lattice pixels.
- Same previously-viewed status: within 32 pixels of any recorded fixation up to and including the current origin. No future fixation is consulted.

Target proximity now uses Euclidean distance to the nearest pixel of the union of annotated bottle masks, rather than the old box. This is a raster distance, not an exact subpixel contour distance. Exclude target masks dilated by 16 stimulus pixels. If any fixation from the initial point through the candidate landing has entered that exclusion region, exclude the decision. The distance caliper itself uses distance to the undilated bottle mask.

Uniform sampling without replacement uses the same decision-specific hash seed rule as Day 7. There is no tuning by feature contrast or model result. As before, alternatives are comparison locations, not measured eye movements or demonstrated choices considered by the observer.

## Failures and sensitivity
In the two-laptop image (522129), participants 1, 4, 7, 8 and 9 have unlabelled chosen pixels. Participant 2's laptop landing and participant 10's cell-phone landing each have only five semantic alternatives, so both fail. All attempts remain in decision_log.csv.

| Bottle exclusion margin | Attempted | Matched | Too few alternatives | Other exclusions |
|---|---:|---:|---:|---:|
| 0 px | 17 | 10 | 2 | 5 |
| 16 px (primary) | 17 | 10 | 2 | 5 |
| 32 px | 17 | 9 | 1 | 7 |

The 32-pixel margin excludes two additional decisions through target-mask entry and removes one previously failed decision from the matching stage. This is sensitivity to the stated margin, not evidence for selecting whichever setting gives more matches. Per-alternative distance differences are in the candidate tables; balance.csv summarizes mean absolute differences by chosen gaze-distance band.

The drop from 15 spatial-only matches to 10 here is not solely the effect of category matching: the extension also changes target distance from box to mask, adds mask dilation and void checks, and refines padding boundaries. Matching controls only the declared category/area and spatial variables, not every semantic or visual confound.

## Verification and rerun
- Verified original mask IDs and areas against metadata, and visually checked every alignment overlay.
- Checked category equality, thing-area ratios, eight unique alternatives, spatial tolerances, mask exclusion and row reconciliation.
- Independently checked all 90 primary rows against exhaustive distances to target-mask pixels, source category IDs, geometry and area ratios.
- Confirmed byte-identical CSVs on rerun.

From the packet directory:

```sh
.venv/bin/python day7_semantic/verify_alignment.py
.venv/bin/python -W error::RuntimeWarning day7_semantic/semantic_match.py
.venv/bin/python -W error::RuntimeWarning day7_semantic/independent_verify.py
```

The environment now includes SciPy for exact Euclidean distance transforms; requirements.txt records the installed versions. The saved selected inputs allow these checks without downloading the full archive again.

This completes semantic matching for this three-image pilot. The sample is very small and uneven across categories; it does not support a claim that contrast predicts gaze or identifies a neural mechanism.
