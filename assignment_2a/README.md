# Forced Alignment with the Montreal Forced Aligner (MFA) – Assignment 2A

Word- and phone-level forced alignment of six recordings with [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/), run entirely on Google Colab.

---

## What this project does

| Step | Description |
|------|-------------|
| 1 | Install MFA on Colab (conda-forge) |
| 2 | Prepare the data: convert NIST SPHERE audio to WAV, clean transcripts, build the MFA corpus layout |
| 3 | Download the pre-trained `english_us_arpa` acoustic model, dictionary and G2P model |
| 4 | Find out-of-vocabulary (OOV) words and generate pronunciations with G2P → custom dictionary |
| 5 | Align with `english_us_arpa` + custom dictionary (main run) |
| 6 | Extra credit: align again with the `english_mfa` (IPA) model and compare the two |
| 7 | Inspect TextGrids (tables, plots, automatic checks) and export everything |

## Data

Six recordings from the assignment:

| File | Source | Duration |
|------|--------|----------|
| `F2BJRLP1`, `F2BJRLP2`, `F2BJRLP3` | BU Radio News, speaker f2b | 25.3 s, 28.7 s, 30.7 s |
| `ISLE_SESS0131_BLOCKD02_01_sprt1` | ISLE learner speech – "I said white not bait" | 4.1 s |
| `ISLE_SESS0131_BLOCKD02_02_sprt1` | ISLE learner speech – "I said new not no" | 3.9 s |
| `ISLE_SESS0131_BLOCKD02_03_sprt1` | ISLE learner speech – "I said bad not bed" | 4.5 s |

### Data preparation (done in the notebook)

* **Audio:** the three `F2BJRLP*.wav` files are **NIST SPHERE** files (1024-byte text header + raw 16-bit PCM), not RIFF WAVs, so MFA cannot read them. They are converted to 16 kHz, mono, 16-bit WAV. ISLE files were already valid WAVs.
* **Text:** lower-cased, punctuation removed, hyphens split (`seventy- six` → `seventy six`), `S.J.C.'s` → `s j c's`.
* **Layout:** `corpus/<speaker>/<utterance>.wav` + `<utterance>.lab` (speakers: `f2b`, `isle_sess0131`).

## How to run

### Option 1 – Google Colab (recommended)

1. Open `Assignment_2a_MFA.ipynb` in [Google Colab](https://colab.research.google.com) (File → Upload notebook).
2. Run **Step 1** (installs conda; the runtime restarts automatically – do not re-run Step 1).
3. Run the remaining steps top to bottom. When asked, upload `Assignment_2.zip` (the original assignment data), or set `USE_DRIVE = True` to read it from Google Drive.
4. The last cell downloads `assignment2a_outputs.zip` containing all results.

### Option 2 – Local machine

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
```

Then run the commands below on a prepared corpus folder.

## MFA commands used

```bash
# Models
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p        english_us_arpa

# Check corpus / list OOV words
mfa validate corpus english_us_arpa english_us_arpa --clean

# Pronunciations for OOV words only (G2P)
mfa g2p corpus english_us_arpa oov_g2p.txt --dictionary_path english_us_arpa --clean
# -> merged with the original dictionary into custom_english_us_arpa.dict (see notebook)

# Main alignment: english_us_arpa + custom dictionary
mfa align corpus custom_english_us_arpa.dict english_us_arpa output_arpa \
    --clean --beam 100 --retry_beam 400

# Extra credit: second acoustic model (IPA)
mfa model download acoustic   english_mfa
mfa model download dictionary english_mfa
mfa model download g2p        english_us_mfa
mfa align corpus english_mfa english_mfa output_mfa \
    --clean --beam 100 --retry_beam 400 --g2p_model_path english_us_mfa
```

`--beam 100 --retry_beam 400` makes alignment of the ~30 s news recordings more robust.

## Repository structure

```
.
├── README.md
├── Assignment_2a_MFA.ipynb          # full Colab pipeline
├── Assignment_2A_Report.pdf         # report
├── outputs/
│   ├── textgrids_english_us_arpa/   # main run, 6 TextGrids (words + phones tiers)
│   ├── textgrids_english_mfa/       # extra credit, 6 TextGrids
│   ├── custom_english_us_arpa.dict  # original dictionary + G2P entries
│   ├── oov_g2p.txt                  # G2P output for the OOV words
│   ├── corpus_prepared/             # corpus as given to MFA (converted WAV + .lab)
│   └── figures/                     # alignment plots generated from the TextGrids
└── praat_screenshots/               # Praat screenshots per file and model
```

To inspect an alignment in Praat: open a `.wav` from `outputs/corpus_prepared/` and the matching `.TextGrid`, select both, and click **View & Edit**.

## Results in brief

* All 6 files aligned with both models; no `<unk>` / `spn` words after adding G2P pronunciations.
* OOV words handled via G2P: *dukakis, maffy, melnicove, politicize, wbur*.
* On the news recordings the two models agree closely: median word-boundary difference **10 ms**; only 8 of 452 boundaries differ by more than 100 ms.
* `english_mfa` stretches the final word of two ISLE clips (*bait*, *bed*) across 2–3 s of trailing silence; `english_us_arpa` aligns them correctly.
* The spelled-out acronym *WBUR* is mis-timed by both models because G2P gives it a one-syllable pronunciation.

See the report for figures and discussion.

## Extra credit completed

* Custom dictionary built with G2P for OOV words
* Two different acoustic models, with a quantitative comparison
* Fully automated pipeline in a single Colab notebook
