# Enhanced synthetic multi-label fault training results

Selected overall model: **dense_pca_99**; best DL model: **dense_pca_99**.

Model selection used validation macro average precision. Test results were computed after freezing selection/calibration/thresholds.

| candidate | split | macro_AP | micro_AP | macro_F1 | macro_recall | micro_F1 | hamming_loss | exact_match | normal_false_alarm_rate | macro_Brier | rows | inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree_all | validation_tuning | 0.7946 | 0.7918 | 0.7712 | 0.8022 | 0.5825 | 0.0598 | 0.4610 | 0.5023 | 0.0244 | 16416 | 220 |
| tree_correlation_pruned | validation_tuning | 0.7980 | 0.7917 | 0.7762 | 0.8075 | 0.6053 | 0.0551 | 0.4912 | 0.4944 | 0.0251 | 16416 | 138 |
| tree_64 | validation_tuning | 0.7889 | 0.7894 | 0.7759 | 0.8085 | 0.5829 | 0.0607 | 0.4675 | 0.5278 | 0.0240 | 16416 | 64 |
| tree_32 | validation_tuning | 0.7869 | 0.7793 | 0.7660 | 0.7776 | 0.5893 | 0.0559 | 0.4683 | 0.5039 | 0.0242 | 16416 | 32 |
| tree_16 | validation_tuning | 0.7813 | 0.7877 | 0.7720 | 0.7877 | 0.6065 | 0.0533 | 0.5054 | 0.4963 | 0.0232 | 16416 | 16 |
| tree_8 | validation_tuning | 0.5059 | 0.5876 | 0.5324 | 0.6689 | 0.3880 | 0.1177 | 0.2558 | 0.5203 | 0.0336 | 16416 | 8 |
| tree_proxy_ablated | validation_tuning | 0.7875 | 0.7871 | 0.7734 | 0.8126 | 0.5871 | 0.0604 | 0.4679 | 0.5288 | 0.0245 | 16416 | 217 |
| tree_selected_full_fit | validation_tuning | 0.7919 | 0.7900 | 0.7740 | 0.7953 | 0.5778 | 0.0604 | 0.4529 | 0.5216 | 0.0233 | 16416 | 32 |
| logistic_selected | validation_tuning | 0.7729 | 0.7846 | 0.7683 | 0.7812 | 0.5798 | 0.0587 | 0.4136 | 0.5932 | 0.0624 | 16416 | 32 |
| dense_selected | validation_tuning | 0.8045 | 0.8230 | 0.7987 | 0.8143 | 0.6602 | 0.0445 | 0.5317 | 0.4649 | 0.0404 | 16416 | 32 |
| tcn_selected | validation_tuning | 0.8116 | 0.8363 | 0.8141 | 0.8295 | 0.6920 | 0.0391 | 0.5555 | 0.4940 | 0.0342 | 16416 | 32 |
| dense_pca_90 | validation_tuning | 0.8047 | 0.8189 | 0.7864 | 0.8050 | 0.6575 | 0.0436 | 0.5107 | 0.4924 | 0.0410 | 16416 | 220 |
| dense_pca_95 | validation_tuning | 0.8183 | 0.8311 | 0.7997 | 0.8055 | 0.6956 | 0.0366 | 0.5749 | 0.4167 | 0.0359 | 16416 | 220 |
| dense_pca_99 | validation_tuning | 0.8391 | 0.8573 | 0.8277 | 0.8472 | 0.7086 | 0.0362 | 0.5814 | 0.4507 | 0.0306 | 16416 | 220 |
| dense_autoencoder | validation_tuning | 0.6500 | 0.6800 | 0.6536 | 0.7087 | 0.4855 | 0.0799 | 0.3588 | 0.5887 | 0.0744 | 16416 | 220 |
| training_prevalence_baseline | validation_tuning | 0.0545 | 0.0705 | 0.1026 | 1.0000 | 0.1034 | 0.9455 | 0.0000 | 1.0000 | 0.0513 | 16416 | 0 |
| dense_pca_99_raw | validation_tuning | 0.8391 | 0.8573 | 0.8277 | 0.8472 | 0.7086 | 0.0362 | 0.5814 | 0.4507 | 0.0306 | 16416 | 220 |
| dense_pca_99_sigmoid_calibrated | validation_tuning | 0.8391 | 0.8384 | 0.7994 | 0.7751 | 0.8114 | 0.0183 | 0.7393 | 0.0474 | 0.0168 | 16416 | 220 |
| dense_pca_99_raw | test | 0.8326 | 0.8724 | 0.7880 | 0.8135 | 0.6719 | 0.0428 | 0.5453 | 0.4683 | 0.0337 | 24624 | 220 |
| dense_pca_99_sigmoid_calibrated | test | 0.8326 | 0.8751 | 0.7812 | 0.7621 | 0.8142 | 0.0188 | 0.7481 | 0.0602 | 0.0145 | 24624 | 220 |
| tree_all | test | 0.7847 | 0.8017 | 0.7496 | 0.8080 | 0.5766 | 0.0631 | 0.4319 | 0.5072 | 0.0242 | 24624 | 220 |
| tree_correlation_pruned | test | 0.7968 | 0.8134 | 0.7476 | 0.8066 | 0.6044 | 0.0566 | 0.4794 | 0.5042 | 0.0246 | 24624 | 138 |
| tree_64 | test | 0.7828 | 0.8051 | 0.7412 | 0.7971 | 0.5687 | 0.0645 | 0.4392 | 0.5320 | 0.0234 | 24624 | 64 |
| tree_32 | test | 0.7812 | 0.8052 | 0.7437 | 0.7775 | 0.5800 | 0.0591 | 0.4539 | 0.5089 | 0.0234 | 24624 | 32 |
| tree_16 | test | 0.7786 | 0.8106 | 0.7323 | 0.7683 | 0.5804 | 0.0586 | 0.4799 | 0.4955 | 0.0226 | 24624 | 16 |
| tree_8 | test | 0.5006 | 0.6176 | 0.4897 | 0.6478 | 0.3686 | 0.1260 | 0.2550 | 0.5296 | 0.0321 | 24624 | 8 |
| tree_proxy_ablated | test | 0.7811 | 0.8023 | 0.7320 | 0.8037 | 0.5755 | 0.0639 | 0.4293 | 0.5301 | 0.0241 | 24624 | 217 |
| tree_selected_full_fit | test | 0.7827 | 0.8101 | 0.7372 | 0.7888 | 0.5593 | 0.0655 | 0.4282 | 0.5237 | 0.0228 | 24624 | 32 |
| logistic_selected | test | 0.7828 | 0.8143 | 0.7553 | 0.7761 | 0.5721 | 0.0614 | 0.4147 | 0.5880 | 0.0625 | 24624 | 32 |
| dense_selected | test | 0.8128 | 0.8509 | 0.7875 | 0.8253 | 0.6462 | 0.0483 | 0.5039 | 0.4990 | 0.0415 | 24624 | 32 |
| tcn_selected | test | 0.7982 | 0.8457 | 0.7786 | 0.8233 | 0.6613 | 0.0451 | 0.5160 | 0.5172 | 0.0376 | 24624 | 32 |
| dense_pca_90 | test | 0.7985 | 0.8432 | 0.7564 | 0.7739 | 0.6383 | 0.0473 | 0.4983 | 0.4937 | 0.0409 | 24624 | 220 |
| dense_pca_95 | test | 0.8154 | 0.8601 | 0.7738 | 0.7867 | 0.6813 | 0.0396 | 0.5549 | 0.4100 | 0.0366 | 24624 | 220 |
| dense_pca_99 | test | 0.8326 | 0.8724 | 0.7880 | 0.8135 | 0.6719 | 0.0428 | 0.5453 | 0.4683 | 0.0337 | 24624 | 220 |
| dense_autoencoder | test | 0.6436 | 0.6854 | 0.6173 | 0.6692 | 0.4677 | 0.0824 | 0.3577 | 0.5633 | 0.0755 | 24624 | 220 |

## Limits
Synthetic fault detection only; predictions do not confirm real root causes. Health is a sensor heuristic. RUL unavailable.
Weak effects, partial matrix signatures, only three test trucks, route/time proxies and sensor masks limit generalization.
Calibration uses one separate validation truck (not field calibration). AE/PCA dimensions do not imply fewer physical sensors.
Missing-channel stress is partial masking of selected derived inputs, not a complete sensor-failure replay. SHAP was not run.

## Reproduce
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /tmp/opencode/fault-training/bin/python analysis/train_fault_models.py --output analysis/results_enhanced_no_gnss_brakes --epochs 18 --exclude-gnss-brakes
```
