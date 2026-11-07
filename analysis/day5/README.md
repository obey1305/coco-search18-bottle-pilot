# Day 5 — colour centre–surround contrast
Implemented and verified by Codex on 8 October 2026, ahead of the scheduled 10 October task. Aditya has not yet personally checked these values.

[Open the contrast map](colour_contrast_map.png) · [Five numeric checks](five_values.csv) · [Verification results](verification.json)

## What is new
The restaurant photograph is downloaded data. The contrast map is newly computed by our script from its pixel colours. It does not use the recorded fixations or target box to compute contrast. Labels A–E are five fixed inspection points, not fixation numbers or ranked peaks.

Image: 000000496294.jpg, the first image under Day 4's fixed selection rule. Its SHA256 was checked against selection.json before computation.

## Definition
Assume the JPEG RGB samples are sRGB. Decode the sRGB transfer function, convert linear RGB to XYZ and then CIELAB using the D65 2-degree reference white (0.95047, 1, 1.08883). Average each pixel's a* and b* components after conversion; do not convert an averaged RGB colour instead.

At each integer stimulus coordinate:
- Centre: disk with squared radius <= 16²; 797 pixels.
- Surround: annulus with 32² <= squared radius <= 64²; 9,648 pixels. Both annular boundaries are included. The gap between the centre disk and annulus is unused.
- C = sqrt((mean centre a* − mean surround a*)² + (mean centre b* − mean surround b*)²).

This measures chromatic difference; it does not include the L* luminance component. It is not a full Delta E score, an attention probability, or a calibrated neural response. A larger value means the centre's average chromatic colour differs more from the surrounding ring.

## Five checked locations
| Label | x | y | Colour contrast |
|---|---:|---:|---:|
| A | 840 | 525 | 2.874116 |
| B | 400 | 500 | 9.257164 |
| C | 1050 | 350 | 10.217079 |
| D | 1480 | 350 | 24.608213 |
| E | 900 | 750 | 11.083881 |

Example A: centre (a*, b*) = (−0.038962, 0.682546); surround = (0.918862, 3.392364). The Euclidean difference is approximately sqrt(0.957824² + 2.709818²) = 2.874116.

The five coordinates were fixed before calculating the map, covering the sandwich, printed tabletop area, dumpling area and bottle. D lies inside the released target box; it is included to inspect feature computation only. This map is not a selected non-target candidate set. Later analyses still need target exclusion, eligible movements and matched controls.

## Padding and support exclusions
Use the visually checked Day 4 content rectangle: 0 <= x < 1680 and 41 <= y < 1009. Every centre must have its complete radius-64 support inside that rectangle. Valid centres are x=64…1615 and y=105…944, inclusive. There are 1,303,680 valid centres and 460,320 excluded centres on the full 1680 × 1050 grid.

Excluded values are NaN in the numeric map and grey in the figure, not zero. No truncated patches, reflection or border replication are used. FFT convolution uses sufficient zero-extension to prevent circular wrapping; all points whose support touches that extension are masked out.

Padding boundaries inherit Day 4's near-black-pixel estimate and visual check. They are not independently established original-COCO transform offsets; JPEG edge artifacts can affect the inferred content boundary at roughly the pixel level. Eligibility here is relative to that recorded rectangle.

## Verification performed
- Direct Boolean-mask averaging at all five locations agrees with the FFT map to < 1e-9; observed maximum error < 4e-13.
- Black, white and red sRGB values match expected Lab reference values within stated code tolerances.
- A uniform-colour synthetic image yields zero contrast within numerical tolerance.
- A synthetic centre whose a*/b* differences are 3 and 4 yields contrast 5.
- Eight exact boundary checks test one valid and one invalid point on each side.
- All excluded map entries are NaN; all valid entries are finite; the valid-centre count matches the rectangle calculation.
- The final figure was visually inspected for point alignment, readable labels, colour scale and excluded borders.

A runtime warning was isolated to NumPy's matrix-multiplication path with finite inputs; its result matched an explicit sum within 2.3e-16. The conversion now uses an explicit einsum without that path. The full final run passed with RuntimeWarnings treated as errors.

These are implementation checks, not validation that colour contrast predicts gaze.

## Files and rerun
- colour_contrast.py: complete computation and checks.
- colour_contrast_map.png: source image with disks/rings beside the computed map.
- colour_contrast.npz: full-resolution contrast, Boolean validity mask and content bounds.
- five_values.csv: exact centre/surround averages, direct/map results and errors.
- verification.json and run.log: provenance, counts and checks.

From the pilot packet directory:

```sh
.venv/bin/python -W error::RuntimeWarning day5/colour_contrast.py > day5/run.log
```

## Your next learning step
Open the map. Brighter colours on the right mean greater colour difference between centre and surround. Compare A (2.87) and D (24.61). The grey border means excluded, not low contrast. You can then check the five values against the labelled locations and record only the checks you personally perform.
