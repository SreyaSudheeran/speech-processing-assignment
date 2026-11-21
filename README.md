# speech-processing-assignment

Speech and language processing assignments: forced alignment (Assignment 2a) and multimodal emotion recognition (Assignment 2b). The two assignments are independent; each folder has its own README, report and outputs.

| Folder | Task | Start here |
|---|---|---|
| [`assignment_2a/`](assignment_2a/) | Word- and phone-level forced alignment of six recordings with the Montreal Forced Aligner (MFA), run on Google Colab; includes an extra-credit comparison of two acoustic models | [README](assignment_2a/README.md), [Report](assignment_2a/Report.pdf) |
| [`assignment_2b/`](assignment_2b/) | Multimodal emotion recognition on TESS: speech-only, text-only and fusion models, evaluated on unseen words and unseen speakers | [README](assignment_2b/README.md), [Report](assignment_2b/Report.pdf) |

## Repository layout
```
speech-processing-assignment/
├── README.md
├── assignment_2a/
│   ├── README.md
│   ├── Report.pdf
│   ├── Assignment_2a_MFA.ipynb     Colab notebook (full pipeline)
│   ├── outputs/                    prepared corpus, custom dictionary, TextGrids, figures
│   └── praat_screenshots/
└── assignment_2b/
    ├── README.md
    ├── Report.pdf
    ├── assignment-2b.ipynb         Kaggle notebook (writes project/ and runs every experiment)
    └── project/
        ├── common/  models/  analyze.py  requirements.txt
        └── Results/                all experiment outputs (see Results_README.md)
```

## Assignment 2a in brief
Six recordings are converted to MFA's corpus format and aligned with `english_us_arpa` plus a custom dictionary (OOV pronunciations from G2P). The extra-credit run repeats the alignment with `english_mfa` (IPA). TextGrids from both runs, plots and Praat screenshots are in `assignment_2a/outputs/` and `assignment_2a/praat_screenshots/`.

## Assignment 2b in brief
Three pipelines (speech, text, fusion) are evaluated on TESS with a word-disjoint split and with unseen-speaker splits in both directions.
- Text-only is at chance (14.29%) by construction: TESS uses the same carrier phrase in every emotion.
- Frozen WavLM features beat log-mel by about 22 to 26 points on an unseen speaker (five seeds, both directions).
- Per-speaker normalisation adds little, and fusion does not beat speech-only.
- Run-to-run variance is large, so differences of a few points are not conclusive.

Reproduce 2b by running `assignment_2b/assignment-2b.ipynb` on Kaggle (GPU on, Internet on, TESS dataset attached).
