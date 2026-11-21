### Speech-only  (424 errors / 1400)

| speaker   | word   | true   | pred    |   confidence |
|:----------|:-------|:-------|:--------|-------------:|
| YAF       | came   | angry  | disgust |        0.953 |
| YAF       | wire   | angry  | disgust |        0.932 |
| YAF       | dodge  | angry  | disgust |        0.931 |
| YAF       | youth  | angry  | disgust |        0.928 |
| YAF       | bath   | angry  | disgust |        0.924 |

Most frequent confusions (true -> predicted): happy->angry x170; angry->disgust x157; angry->sad x25; happy->disgust x14; sad->fear x13

Errors by speaker:

| speaker   | correct        |
|:----------|:---------------|
| YAF       | 424/1400 wrong |

### Multimodal (fusion)  (492 errors / 1400)

| speaker   | word   | true   | pred   |   confidence |
|:----------|:-------|:-------|:-------|-------------:|
| YAF       | raise  | happy  | angry  |        0.961 |
| YAF       | germ   | happy  | angry  |        0.96  |
| YAF       | raid   | happy  | angry  |        0.956 |
| YAF       | lot    | happy  | angry  |        0.955 |
| YAF       | hit    | happy  | angry  |        0.954 |

Most frequent confusions (true -> predicted): happy->angry x188; sad->neutral x126; angry->sad x63; ps->happy x50; angry->neutral x23

### Text-only  (1200 errors / 1400)

| speaker   | word   | true    | pred   |   confidence |
|:----------|:-------|:--------|:-------|-------------:|
| YAF       | yearn  | ps      | sad    |        0.144 |
| YAF       | yearn  | disgust | sad    |        0.144 |
| YAF       | yearn  | happy   | sad    |        0.144 |
| YAF       | yearn  | fear    | sad    |        0.144 |
| YAF       | yearn  | neutral | sad    |        0.144 |

