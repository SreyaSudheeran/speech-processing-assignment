# Results folder guide

All runs use TESS (2 speakers: OAF, YAF; 7 emotions; 2,799 clips in the Kaggle copy, one OAF clip is missing). Accuracy / macro-F1 are on the test split of each protocol. Single runs use seed 42.

| Folder | What it contains | Protocol |
|---|---|---|
| `1_word_disjoint/` | Speech-only, text-only and fusion models with full outputs (`comparison/`, `speech/`, `text/`, `fusion/`, `metadata_splits.csv`) | Word-disjoint split, both speakers in training, log-mel (Protocol 1) |
| `2_speaker_baseline/` | Same outputs as above | Train OAF -> test YAF (1,400 clips), log-mel (Protocol 2) |
| `3_speaker_FINAL/` | Same outputs as above; headline run | Train OAF -> test YAF (1,400 clips), WavLM + per-speaker normalisation (Protocol 3) |
| `reverse_direction/` | `metrics.json` of the four speech configurations, single seed | Train YAF -> test OAF (1,399 clips). Exploratory comparison, not independent model selection (YAF is the training speaker here and the test speaker in Protocol 3) |
| `ablation_train_OAF_test_YAF/` | `metrics.json` of the same four speech configurations, single seed | Train OAF -> test YAF (1,400 clips) |
| `seeds_train_OAF_test_YAF/` | 15 runs (3 speech configurations x seeds 1-5); the five `wavlm_spk_norm` folders also contain the fusion run (`fusion/metrics.json`, `fusion/ablation.json`). | Train OAF -> test YAF (1,400 clips) |
| `seeds_train_YAF_test_OAF/` | 15 speech-only runs (3 configurations x seeds 1-5). | Train YAF -> test OAF (1,399 clips) |
| `first_run/` (optional) | Outputs of the first execution of the pipeline, before the WavLM and speaker-normalisation options were added | Cited in the report's run-to-run variance table only |

## Configurations
- `logmel`: 64-band log-mel features.
- `wavlm`: frozen WavLM-base+ features (mean of layers 4-8).
- `+spk_norm`: per-speaker feature standardisation (uses the unlabeled audio statistics of each speaker; transductive).

## Notes
- Text-only accuracy is at chance (14.29%) in every protocol: TESS transcripts ("Say the word ___") contain the same words across all emotions.
- Run-to-run variance is large (standard deviations of about 3 to 8 points across seeds), so single-run differences of a few points are not conclusive. Use the `seeds_*` folders for comparisons between configurations.
- Model checkpoints (`best.pt`) are not included because of size.
- Test labels are used only for evaluation and descriptive analysis (confusion matrices, silhouette), never for training or kNN fitting.
