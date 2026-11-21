### Speech-only  (834 errors / 1400)

| speaker   | word   | true   | pred   |   confidence |
|:----------|:-------|:-------|:-------|-------------:|
| YAF       | match  | happy  | angry  |        0.98  |
| YAF       | yes    | happy  | angry  |        0.978 |
| YAF       | sheep  | happy  | angry  |        0.977 |
| YAF       | check  | happy  | angry  |        0.977 |
| YAF       | soup   | happy  | angry  |        0.977 |

Most frequent confusions (true -> predicted): happy->angry x200; ps->angry x153; disgust->angry x135; fear->happy x118; sad->angry x111

Errors by speaker:

| speaker   | correct        |
|:----------|:---------------|
| YAF       | 834/1400 wrong |

### Multimodal (fusion)  (697 errors / 1400)

| speaker   | word   | true    | pred   |   confidence |
|:----------|:-------|:--------|:-------|-------------:|
| YAF       | cause  | fear    | happy  |        0.982 |
| YAF       | death  | disgust | ps     |        0.974 |
| YAF       | week   | happy   | angry  |        0.97  |
| YAF       | check  | happy   | angry  |        0.968 |
| YAF       | pick   | happy   | angry  |        0.968 |

Most frequent confusions (true -> predicted): happy->angry x197; ps->angry x136; fear->happy x135; disgust->neutral x119; disgust->ps x47

### Text-only  (1200 errors / 1400)

| speaker   | word   | true    | pred   |   confidence |
|:----------|:-------|:--------|:-------|-------------:|
| YAF       | yearn  | ps      | sad    |        0.144 |
| YAF       | yearn  | disgust | sad    |        0.144 |
| YAF       | yearn  | happy   | sad    |        0.144 |
| YAF       | yearn  | fear    | sad    |        0.144 |
| YAF       | yearn  | neutral | sad    |        0.144 |

