# Multimodal Emotion Recognition on TESS (Speech / Text / Fusion)

Three pipelines following the block diagram in the assignment (preprocessing -> feature extraction -> temporal/contextual modelling -> fusion -> classifier), evaluated under three protocols.

## Where to find each deliverable
- **Report** (A. architecture decisions per block, B. speech-only vs text-only vs fusion experiments, C. analysis: easiest/hardest emotions, when fusion helps, failure cases, separability of the temporal, contextual and fusion representations): [`Report.pdf`](Report.pdf)
- **Code**: [`project/common/`](project/common/), [`project/models/`](project/models/), [`project/analyze.py`](project/analyze.py); notebook runner [`assignment-2b.ipynb`](assignment-2b.ipynb)
- **Results**: [`project/Results/`](project/Results/) (guide: [`Results_README.md`](project/Results/Results_README.md))
- **Figures**: [`project/Results/3_speaker_FINAL/comparison/`](project/Results/3_speaker_FINAL/comparison/) (t-SNE plots, accuracy and per-emotion F1 plots)
- **Failure cases**: [`failure_cases.md`](project/Results/3_speaker_FINAL/comparison/failure_cases.md)

## Results (test accuracy / macro-F1, %)

| Protocol | Speech-only | Text-only | Multimodal (fusion) |
|---|---|---|---|
| 1. Word-disjoint (unseen words, both speakers in training), log-mel | 100.00 / 100.00 | 14.29 / 3.57 | 100.00 / 100.00 |
| 2. Unseen speaker (train OAF, test YAF), log-mel, baseline | 40.43 / 36.79 | 14.29 / 3.57 | 50.21 / 44.41 |
| 3. Unseen speaker (train OAF, test YAF), **WavLM + per-speaker norm (final)** | **69.71 / 63.79** | 14.29 / 3.57 | 64.86 / 62.43 |

Chance = 14.29%. Text-only sits exactly at chance (see dataset notes). Each cell is a single run (seed 42). Run-to-run variance is large: the same WavLM + speaker-norm configuration scored 69.71% (final run) and 58.29% (ablation run), and the log-mel baseline ranged 35-50% across runs. Treat differences of a few points (fusion vs speech-only, speaker norm on/off) as inconclusive; WavLM > log-mel held in every run.

**Five-seed repeats (speech-only accuracy %, mean ± std, seeds 1-5):**

| Configuration | Train OAF -> test YAF | Train YAF -> test OAF |
|---|---|---|
| Log-mel | 43.79 ± 3.43 | 37.87 ± 4.66 |
| WavLM | 66.57 ± 4.89 | 60.33 ± 2.90 |
| WavLM + speaker norm | 69.49 ± 4.03 | 61.44 ± 2.98 |

Fusion (WavLM + speaker norm, train OAF -> test YAF, 5 seeds): 68.74 ± 7.55. WavLM beats log-mel by ~22-26 points in both directions; speaker normalisation gives at most a small, inconsistent gain (+2.9 and +1.1 points); fusion does not beat speech-only.

## Pipelines

| Pipeline | Preprocessing | Feature extraction | Temporal / contextual model | Fusion | Classifier |
|---|---|---|---|---|---|
| Speech | resample 16 kHz, trim silence (30 dB), peak-normalise, crop/pad to 3 s | `logmel`: 64-band log-mel, 25 ms / 10 ms (time x 64). `wavlm`: frozen WavLM-base+ (mean of layers 4-8, time x 768, 20 ms frames). Standardisation: train-set statistics, or per-speaker with `--spk_norm` | 2x Conv1d + BiGRU + attention pooling | – | 2-layer MLP |
| Text | lowercase, strip punctuation, WordPiece tokenise ("say the word X") | frozen DistilBERT token embeddings (tokens x 768); `scratch` = learned word embeddings (offline only, must be requested explicitly) | projection + BiGRU + attention pooling | – | 2-layer MLP |
| Fusion | both of the above | both (frozen WavLM / DistilBERT) | speech and text modelling layers trained jointly with fusion and classifier; pretrained feature extractors stay frozen | gated fusion + modality dropout (0.15) | 2-layer MLP |

## Dataset notes
TESS = 2 speakers (OAF, YAF) x 200 words x 7 emotions = 2800 clips, all in the carrier phrase "Say the word ___". The Kaggle copy has no transcript files, so the transcript is rebuilt from the filename (`Say the word <word>`). Because every word occurs in every emotion, **the text carries no emotion information**: text-only accuracy is at chance and the text-only model predicts a single class. This is a property of the dataset, not a bug.

## Evaluation protocols and leakage controls
- **Word-disjoint split (`--split word`, default):** 70/15/15 by word (seed 42). All 14 clips of a word stay in one split; classes remain balanced. Both speakers appear in every split, so this is speaker-dependent.
- **Speaker-independent split (`--split speaker --test_speaker YAF`):** train/val on OAF only (val = 15% of OAF's words, disjoint from train), test = all YAF clips.
- **Checkpoint choice / early stopping** use only the validation split. The test split is evaluated only by `test.py`.
- **Augmentation** (SpecAugment-style masks, feature noise) is applied to the training set only.
- **Feature statistics:** by default computed on the training split only. `--spk_norm` standardises each speaker with that speaker's own clips (no labels used). This is *transductive*: it uses the unlabeled test speaker's audio statistics. Report it as such.
- **Pretrained encoders** (DistilBERT, WavLM) are frozen and were not trained on TESS labels.
- **Configuration comparison (exploratory):** the four speech configurations (`logmel`/`wavlm` x `--spk_norm`) were first compared in the reverse direction (train YAF -> test OAF, 1,399 clips; one OAF clip is missing from the Kaggle copy), where WavLM + speaker norm scored highest. This is **not** an independent model-selection procedure: YAF, the final test speaker, is the *training* speaker in that experiment. All four configurations were also evaluated on YAF (OAF -> YAF) as a labelled ablation, and five seeds were run on that direction. The WavLM advantage (~23 points) is large; the speaker-norm gain (~3 points) is modest and should be read cautiously.
- **Representation analysis:** silhouette is computed on test embeddings with emotion labels (descriptive only, nothing is fitted); the kNN probe is fitted on **train** embeddings and labels and scored on **test** embeddings; silhouette uses test labels purely as a descriptive measure; t-SNE is unsupervised. Test labels are never used for training or kNN fitting.
- DistilBERT failures raise an error (no silent fallback). The text encoder, speech features and split actually used are stored in each `metrics.json`.

## Run (Kaggle or local)
Run the commands below from `assignment_2b/project/`. Easiest: run [`assignment-2b.ipynb`](assignment-2b.ipynb) on Kaggle (add the TESS dataset, GPU on, Internet on). It writes this whole project and runs every experiment. Locally:
```bash
pip install -r requirements.txt
# --data_dir is auto-detected under /kaggle/input; locally pass the folder containing the .wav files
D=/path/to/TESS

# 1. word-disjoint
for p in speech text fusion; do
  python models/${p}_pipeline/train.py --data_dir $D
  python models/${p}_pipeline/test.py  --data_dir $D
done
python analyze.py

# 3. final: unseen speaker with WavLM + per-speaker norm
F="--data_dir $D --split speaker --test_speaker YAF --speech_feat wavlm --spk_norm --results_dir Results_final"
for p in speech text fusion; do
  python models/${p}_pipeline/train.py $F
  python models/${p}_pipeline/test.py  $F
done
python analyze.py --results_dir Results_final
```
For protocol 2 drop `--speech_feat wavlm --spk_norm`. For the reverse-direction comparison use `--test_speaker OAF`. Always pass identical flags to `train.py` and `test.py`.
Other flags: `--epochs 40 --lr 1e-3 --batch_size 32 --seed 42 --text_encoder {distilbert,scratch} --allow_fallback`.

## Outputs
Scripts write to `Results/` (inside `project/`) by default (or `--results_dir`). In this repo the runs are organised as follows (see `project/Results/Results_README.md` for details):
```
Results/
├── 1_word_disjoint/                protocol 1
├── 2_speaker_baseline/             protocol 2 (log-mel, no speaker norm)
├── 3_speaker_FINAL/                protocol 3, headline result
├── reverse_direction/              four speech configs, train YAF -> test OAF (exploratory comparison)
├── ablation_train_OAF_test_YAF/    four speech configs, train OAF -> test YAF
├── seeds_train_OAF_test_YAF/       3 configs x 5 seeds (+ fusion for WavLM + speaker norm)
└── seeds_train_YAF_test_OAF/       3 configs x 5 seeds, speech-only
```
Each full run folder contains:
- `speech/ text/ fusion/`: `metrics.json`, `per_class_report.csv`, `confusion_matrix.png`, `training_curves.png`, `history.csv`, `failure_cases.csv`, `test_predictions.csv`, `embeddings_train.npz`, `embeddings_test.npz`; fusion also has `ablation.json` (speech or text input zeroed at test time). Checkpoints (`best.pt`) are not included (size).
- `comparison/`: `accuracy_table.csv/.md`, `accuracy_bar.png`, `per_emotion_f1.csv/.png`, `fusion_vs_speech_delta.csv`, `modality_ablation.csv`, `tsne_temporal.png`, `tsne_contextual.png`, `tsne_fusion.png`, `tsne_all.png`, `separability.csv`, `failure_cases.md`
- `metadata_splits.csv`: every clip with its speaker, word, emotion and split.

## Code layout
```
assignment_2b/
├── README.md                 this file
├── Report.pdf                technical report (with references)
├── assignment-2b.ipynb       one-notebook runner for Kaggle (writes project/ and runs everything)
└── project/
    ├── common/               data.py (parsing, splits, preprocessing, features), models.py, train_utils.py
    ├── models/{speech,text,fusion}_pipeline/{train.py,test.py}
    ├── analyze.py            comparison tables, t-SNE, failure cases
    ├── requirements.txt
    └── Results/              all experiment outputs + Results_README.md
```

## External resources
- TESS dataset: Dupuis & Pichora-Fuller (2010), https://doi.org/10.5683/SP2/E8H2MF (Kaggle copy used).
- WavLM (`microsoft/wavlm-base-plus`): Chen et al., IEEE JSTSP 2022, https://arxiv.org/abs/2110.13900
- DistilBERT (`distilbert-base-uncased`): Sanh et al., 2019, https://arxiv.org/abs/1910.01108
- Gated fusion (Arevalo et al., 2017), modality dropout (Neverova et al., 2015), SpecAugment (Park et al., 2019), AdamW, one-cycle schedule, label smoothing, t-SNE and silhouette are cited in full in the report's References section.
- Libraries: PyTorch, scikit-learn, librosa, Hugging Face Transformers.
