# Day 6 — orientation and luminance contrast
Implemented and checked by Codex on 9 October 2026, ahead of the planned 11 October task. Personal review by Aditya remains separate from the automated and visual checks below.

[Feature maps](feature_maps.png) · [Stripe tests](filter_checks.png) · [Five checked values](five_values.csv) · [Verification record](verification.json)

## What the maps measure
Orientation contrast compares four local filter-response averages in a centre disk with those in a surrounding ring. Luminance contrast compares mean CIELAB L* (perceptual lightness) between the same centre and ring. The latter is called luminance contrast in the pilot specification, but is not calibrated physical luminance.

Both are computed from downloaded photograph 000000496294.jpg, whose checksum matches Day 4. No fixations or target-box labels enter the feature calculation. Labels A–E mark the same inspection locations as Day 5. D is on the target and is for computation checks only.

## Frozen implementation choices
The packet specifies four Gabor orientations, wavelength 16 pixels, Gaussian sigma 8 pixels and unit-L2 kernels. Here the remaining choices are explicit:
- Grayscale input: relative linear-light Y = 0.2126729 R + 0.7151522 G + 0.0721750 B after sRGB decoding. Assumes sRGB and D65, as on Day 5.
- Four real, even-phase cosine Gabors with isotropic Gaussian envelopes; angles 0°, 45°, 90°, 135° specify the carrier normal in image coordinates (x right, y down). Thus 0° responds to vertical stripes and 90° to horizontal stripes; angles do not name the stripe direction.
- Kernels span −24…24 pixels on each axis (49 × 49, truncation at 3 sigma per axis). Subtract the discrete kernel mean to remove DC response, then normalize to unit Euclidean norm. This is a documented completion of the starter specification, not a parameter fit to gaze.
- Take the absolute value of each convolution response. Average each of the four response channels separately in the centre and surround. Orientation contrast is the Euclidean norm of the difference between those two four-component vectors.
- Centre disk radius <=16 pixels; annulus 32<=radius<=64 pixels, including both boundaries, as on Day 5.
- Luminance contrast is abs(mean centre L* − mean surround L*). L* uses the same CIELAB conversion as Day 5.

Orientation contrast is a filter-response difference, not an angular difference in degrees. It can change with edge strength and grayscale contrast as well as orientation distribution. The real-filter choice is phase-sensitive; it is not a quadrature-energy or neural-response model.

## Synthetic checks before the photograph
The script completes these checks before it loads the source photograph:

| Input pattern | 0° | 45° | 90° | 135° |
|---|---:|---:|---:|---:|
| Vertical sinusoidal stripes | 6.275202 | 0.349199 | 0.002929 | 0.349199 |
| Horizontal sinusoidal stripes | 0.002929 | 0.349199 | 6.275202 | 0.349199 |

Patterns use a 16-pixel period and values from 0 to 1. Responses are mean absolute filter responses over an interior region. The expected filter wins for both patterns, with >10× the orthogonal response. Rotating the pattern swaps the horizontal/vertical responses. The 0° and 90° kernels also match under transposition.

All kernels have zero sum and unit L2 norm within 1e-12. Uniform grayscale and L* fields give zero contrast within 1e-10. A synthetic centre with L*=60 in a surround of L*=50 gives luminance contrast 10 within 1e-10.

## Photograph checks
| Point | x | y | Orientation contrast | Luminance contrast |
|---|---:|---:|---:|---:|
| A | 840 | 525 | 0.076298 | 19.448426 |
| B | 400 | 500 | 0.085786 | 0.840040 |
| C | 1050 | 350 | 0.125097 | 3.761055 |
| D | 1480 | 350 | 0.060402 | 18.147906 |
| E | 900 | 750 | 0.094611 | 3.290395 |

All five values for each feature agree with direct centre/ring averaging within 1e-9. At these same points, all four Gabor responses agree with direct spatial convolution within 1e-10 (20 checks). These checks verify the numerical computation, not behavioural prediction.

The two maps have separate scales and units. Do not compare numerical magnitudes or displayed brightness across them as if their scales were equivalent. For example, A had low chromatic contrast on Day 5 (2.87), yet relatively high L* contrast here (19.45): colour and lightness differences are distinct quantities.

## Boundary protection
Use the recorded Day 4 image-content rectangle [0,1680) × [41,1009). Its boundaries were estimated from near-black rows/columns and visually reviewed; JPEG edge uncertainty and original-COCO transform verification remain as documented on Day 4.

Luminance needs a 64-pixel radius: valid x=64…1615 and y=105…944, for 1,303,680 centres.

Orientation needs the surround's 64 pixels plus the Gabor kernel's 24-pixel half-width. Conservatively require 88 pixels in each axis: valid x=88…1591 and y=129…920, for 1,191,168 centres. This keeps every input pixel contributing to each retained response inside the image content. The orientation mask is smaller for this reason.

Both maps use NaN outside their valid regions, displayed in grey. Sixteen one-pixel boundary cases (eight per mask) pass; valid arrays are finite and all excluded values are NaN. Linear FFT convolution is padded to avoid circular wrapping, and only complete-support locations are retained. No reflected or replicated image borders enter valid results.

## Deliverables and rerun
- features.py: documented implementation, synthetic tests first, then photograph checks and figures.
- feature_maps.png: source image, orientation map and luminance map.
- filter_checks.png: synthetic stripe tests and response bars.
- feature_maps.npz: full-resolution maps, separate valid masks, content bounds.
- five_values.csv, verification.json and run.log: inspectable values and validation evidence.

From the packet directory:

```sh
.venv/bin/python -W error::RuntimeWarning day6/features.py > day6/run.log
```

The final run exited successfully with runtime warnings treated as errors. Both exported figures were visually inspected for readability and alignment. The tests support this implementation at the chosen scales; they do not show that any feature predicts fixation choices or establish a neural mechanism.

## Your next learning step
Open the stripe-test figure. Find which bar is tallest for vertical stripes and which is tallest for horizontal stripes. Then open the photograph maps: grey still means excluded, not zero contrast.
