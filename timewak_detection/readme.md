## Metrics

This directory deals with *detectability* metrics, whose code is taken from [TimeWak](https://github.com/soizhiwen/TimeWak)

The detectability metrics for this work are:
1. Context-FID
2. Correlational
3. Discriminative
4. Predictive
5. z-score
6. Mean Average Error (MAE)
7. Bit accuracy
The first 5 exist in TimeWak, where the first 4 are quality metrics, and z-score is a quantity metric. MAE and bit accuracy are not taken from TimeWak.

For each metric taken from TimeWak, the respective files and dependencies are as follows (locations shown are in TimeWak):
1. **Context-FID**
Utils/context_fid > Models/ts2vec/ts2vec/TS2Vec (class)
-> Models/models/encoder/TSEncoder (class) > Models/models/dilated_conv/DilatedConvEncoder (class)
-> Models/ts2vec/models/losses/hierarchical_contrastive_loss (function)
-> Models/ts2vec/utils
2. **Correlational**
Utils/cross_correlation
3. **Discriminative**
Utils/discriminative_metric > Utils/metric_utils/train_test_divide, extract_time (functions)
4. **Predictive**
Utils/predictive_metric > Utils/metric_utils/extract_time (function)
5. **z-score**
Experiments/metric_watermark/get_zscore (function), or if HTW, Utils/watermark_utils/htw_calculate_z_score (function)



The base code is originally taken from TimeWak, but is edited for readability
