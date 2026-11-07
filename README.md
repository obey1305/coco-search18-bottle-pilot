# Where do our eyes move next?
A small COCO-Search18 bottle-search pilot. When people search for a bottle, do their eyes pause at places that stand out more than similar spots nearby?

**Start here:** [Picture-story presentation](presentation/Find_the_bottle.pptx) · [One-page research update](presentation/Pilot_update.pdf) · [Results figure](analysis/day8/descriptive_contrast.png)

## What we found
We processed 30 searches across three pictures. Of 17 eligible eye movements, 10 had eight suitable comparison spots each. Five locations lacked a category label and two had too few alternatives. Compared with their alternatives’ average, 3 of 10 landings had higher colour contrast, 5 had higher orientation contrast, and 3 had higher lightness contrast. Nine landings came from one image. These are descriptive counts; they do not show what caused an eye movement.

## Explore the project
| Folder | Contents |
|---|---|
| [Presentation](presentation/) | Start with the picture story; the second section keeps the detailed graphs and counts. |
| [Audit](analysis/audit/) | Starter script, archive counts and teaching scanpath |
| [Image selection](analysis/day4/) | Fixed sample, exclusions and movement records |
| [Colour](analysis/day5/) and [orientation/lightness](analysis/day6/) | Feature calculations, visual checks and measured values |
| [Spatial matching](analysis/day7/) and [semantic matching](analysis/day7_semantic/) | Comparison locations, rules and failed matches |
| [Results](analysis/day8/) | Ten observations, their alternatives and plots |
| [Sensitivity checks](analysis/day9/) | Results with a smaller patch or a different set of comparison spots |

## Working with the analysis
To inspect the results you do not need to download any datasets. Start with [the ten plotted observations](analysis/day8/observations.csv) and [the sensitivity table](analysis/day9/tables.md).

To rerun the scripts, work inside `analysis/`, create a Python 3.9 virtual environment and install `requirements.txt`. The data-dependent stages need the original COCO-Search18 fixation files and stimulus JPEGs, and the COCO 2017 panoptic annotations. The raw archives, source photos, mask files and large generated arrays are excluded. See [archive metadata](analysis/audit/summary.json), [input hashes](analysis/day7_semantic/source_manifest.json) and the stage READMEs for input placement and commands. This repository preserves the analysis outputs; cloning it alone does not restore the raw inputs.
