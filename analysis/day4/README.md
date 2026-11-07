# Day 4 — three bottle training images
Prepared and checked by Codex on 7 October 2026, ahead of the scheduled 9 October task. These are assistant-run checks; Aditya has not yet personally reviewed these three overlays.

## Selection fixed before inspecting images
From train_split1, take unique image filenames for task=bottle. Compute SHA256 of the UTF-8 string `20261006|bottle|<filename>`, including the zero-padded filename and `.jpg` extension. Sort hashes ascending and select the first three. Keep all ten participants per image. This operationalizes the packet's hash rule for a three-image pilot. No images were replaced based on their results. Each overlay shows participant 1, the smallest participant ID, selected independently of scanpath appearance.

Images were retrieved from the official TP stimulus ZIP by byte ranges. ZIP CRC checks passed. Selection hashes and downloaded-image SHA256 values are in selection.json; raw fixation input is the previously verified TP archive.

## Overlay review
| Image | Overlay | Estimated padding | Visual result |
|---|---|---|---|
| 496294 | [Restaurant bottle](000000496294_overlay.png) | 41 pixels at top and bottom | Blue box encloses the sauce bottle at right. Gold dashed content boundary follows the horizontal black bands. |
| 522129 | [Bottle between laptops](000000522129_overlay.png) | 139 pixels at left and right | Blue box encloses the dark bottle, clipped by the top of the photograph. Box y=0 is consistent with that clipping, not automatically an error. |
| 295317 | [Bottle beside laptop](000000295317_overlay.png) | 140 pixels at left and right | Blue box encloses the purple bottle at left. Content boundary follows the vertical black bands. |

All three stimuli are 1680 × 1050. Coordinates are plotted directly in stimulus pixels with equal x/y scaling, origin top-left and y increasing downward. Gold crosses show actual fixation coordinates; white numbered labels are offset for readability. The blue boxes align visually with the designated bottles. This checks the coordinate convention, not eye-tracker calibration accuracy. A released box is not an object silhouette or proof that all bottle instances are excluded.

Padding boundaries are estimated from full rows/columns whose maximum RGB value exceeds 16 and visually checked. JPEG edge artifacts can shift the inferred boundary by a pixel; these are not independently verified original-COCO transformation offsets. Original-image masks will require a separate alignment check.

## Exclusion table
Counts are adjacent fixation pairs across all 30 trials, not just the three displayed participants.

| Image | Original pairs | Retained | Incorrect trial | Target entry or later | Invalid coordinate endpoint | Invalid origin duration |
|---|---:|---:|---:|---:|---:|---:|
| 496294 | 61 | 1 | 19 | 41 | 0 | 0 |
| 522129 | 28 | 7 | 3 | 17 | 0 | 1 |
| 295317 | 28 | 9 | 0 | 16 | 0 | 3 |
| Total | 117 | 17 | 22 | 74 | 0 | 4 |

Mutually exclusive reason precedence: incorrect trial; first target-box entry or later; nonfinite/out-of-display/padding coordinate endpoint; origin duration outside inclusive 50–1000 ms; retained. Target entry is determined from the original sequence before exclusions. Duration filtering applies to the origin, as specified in the packet. After an invalid point, a later pair may qualify only if it was already adjacent in the original sequence. No points are deleted and reconnected.

All 147 fixation records have matching X/Y/T/length arrays and finite, in-content coordinates; four have out-of-range durations. They remain in observation_log.csv, even when a higher-priority transition exclusion applies. All ten participants share the same valid released box for each image.

Outputs: exclusion_table.csv summarizes exclusions; observation_log.csv retains every fixation and its flags; transition_log.csv retains every original pair and disposition. Verified all 117 pairs have destination=origin+1 and that every summary row reconciles.

The 17 retained movements pass these basic filters only. No segmentation masks, 64-pixel feature-support filter, matched controls or prediction model have been applied. This is a geometry and preprocessing pilot, not evidence of a contrast effect.

## Rerun
From the packet directory:

```sh
.venv/bin/python day4/check_images.py > day4/run.log
```

## Your next small check
Open the restaurant overlay. Find the black strips above and below the photograph. Those strips are padding, not part of the photographed scene. Then check that the blue rectangle encloses the bottle.
