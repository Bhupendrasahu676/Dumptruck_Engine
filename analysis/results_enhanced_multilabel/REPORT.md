# Enhanced synthetic multi-label fault training results

Selected overall model: **dense_pca_99**; best DL model: **dense_pca_99**.

Model selection used validation macro average precision. Test results were computed after freezing selection/calibration/thresholds.

| candidate | split | macro_AP | micro_AP | macro_F1 | macro_recall | micro_F1 | hamming_loss | exact_match | normal_false_alarm_rate | macro_Brier | rows | inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree_all | validation_tuning | 0.8405 | 0.8419 | 0.8119 | 0.8225 | 0.6996 | 0.0375 | 0.5564 | 0.4770 | 0.0225 | 16416 | 256 |
| tree_correlation_pruned | validation_tuning | 0.8475 | 0.8436 | 0.8180 | 0.8356 | 0.6768 | 0.0420 | 0.5026 | 0.5292 | 0.0227 | 16416 | 161 |
| tree_64 | validation_tuning | 0.8373 | 0.8398 | 0.8161 | 0.8186 | 0.6962 | 0.0375 | 0.5382 | 0.4845 | 0.0216 | 16416 | 64 |
| tree_32 | validation_tuning | 0.8419 | 0.8409 | 0.8157 | 0.8059 | 0.6896 | 0.0377 | 0.5482 | 0.5177 | 0.0215 | 16416 | 32 |
| tree_16 | validation_tuning | 0.8396 | 0.8485 | 0.8258 | 0.7969 | 0.7805 | 0.0236 | 0.6863 | 0.1824 | 0.0204 | 16416 | 16 |
| tree_8 | validation_tuning | 0.5058 | 0.5906 | 0.5330 | 0.6967 | 0.3762 | 0.1258 | 0.2495 | 0.5143 | 0.0335 | 16416 | 8 |
| tree_proxy_ablated | validation_tuning | 0.8319 | 0.8330 | 0.8039 | 0.8333 | 0.6638 | 0.0448 | 0.4936 | 0.5260 | 0.0228 | 16416 | 249 |
| tree_selected_full_fit | validation_tuning | 0.8445 | 0.8559 | 0.8324 | 0.8116 | 0.7838 | 0.0237 | 0.6913 | 0.2222 | 0.0194 | 16416 | 16 |
| logistic_selected | validation_tuning | 0.7940 | 0.8213 | 0.7929 | 0.8037 | 0.6680 | 0.0419 | 0.4982 | 0.4121 | 0.0568 | 16416 | 16 |
| dense_selected | validation_tuning | 0.8644 | 0.8854 | 0.8504 | 0.8322 | 0.8084 | 0.0209 | 0.7248 | 0.2882 | 0.0328 | 16416 | 16 |
| tcn_selected | validation_tuning | 0.8708 | 0.8916 | 0.8614 | 0.8512 | 0.8210 | 0.0196 | 0.7468 | 0.2275 | 0.0300 | 16416 | 16 |
| dense_pca_90 | validation_tuning | 0.8767 | 0.8946 | 0.8457 | 0.8296 | 0.8219 | 0.0191 | 0.7465 | 0.1769 | 0.0269 | 16416 | 256 |
| dense_pca_95 | validation_tuning | 0.8824 | 0.8992 | 0.8550 | 0.8454 | 0.8201 | 0.0195 | 0.7344 | 0.1969 | 0.0247 | 16416 | 256 |
| dense_pca_99 | validation_tuning | 0.8951 | 0.9165 | 0.8770 | 0.8752 | 0.8415 | 0.0174 | 0.7594 | 0.2295 | 0.0199 | 16416 | 256 |
| dense_autoencoder | validation_tuning | 0.6783 | 0.7169 | 0.6825 | 0.7081 | 0.5942 | 0.0520 | 0.4850 | 0.4708 | 0.0645 | 16416 | 256 |
| training_prevalence_baseline | validation_tuning | 0.0545 | 0.0705 | 0.1026 | 1.0000 | 0.1034 | 0.9455 | 0.0000 | 1.0000 | 0.0513 | 16416 | 0 |
| dense_pca_99_raw | validation_tuning | 0.8951 | 0.9165 | 0.8770 | 0.8752 | 0.8415 | 0.0174 | 0.7594 | 0.2295 | 0.0199 | 16416 | 256 |
| dense_pca_99_sigmoid_calibrated | validation_tuning | 0.8951 | 0.9101 | 0.8657 | 0.9091 | 0.5721 | 0.0711 | 0.0287 | 1.0000 | 0.0124 | 16416 | 256 |
| dense_pca_99_raw | test | 0.8907 | 0.9158 | 0.8462 | 0.8491 | 0.8103 | 0.0214 | 0.7247 | 0.2641 | 0.0223 | 24624 | 256 |
| dense_pca_99_sigmoid_calibrated | test | 0.8906 | 0.9133 | 0.8408 | 0.8925 | 0.5700 | 0.0723 | 0.0345 | 1.0000 | 0.0120 | 24624 | 256 |
| tree_all | test | 0.8323 | 0.8307 | 0.7912 | 0.8090 | 0.6815 | 0.0401 | 0.5378 | 0.4618 | 0.0230 | 24624 | 256 |
| tree_correlation_pruned | test | 0.8417 | 0.8375 | 0.7850 | 0.8152 | 0.6576 | 0.0453 | 0.4896 | 0.5331 | 0.0231 | 24624 | 161 |
| tree_64 | test | 0.8320 | 0.8348 | 0.8010 | 0.7979 | 0.6846 | 0.0389 | 0.5344 | 0.4831 | 0.0219 | 24624 | 64 |
| tree_32 | test | 0.8331 | 0.8407 | 0.7911 | 0.7816 | 0.6745 | 0.0398 | 0.5434 | 0.5153 | 0.0217 | 24624 | 32 |
| tree_16 | test | 0.8293 | 0.8442 | 0.7888 | 0.7602 | 0.7476 | 0.0269 | 0.6715 | 0.1897 | 0.0207 | 24624 | 16 |
| tree_8 | test | 0.5057 | 0.6245 | 0.5006 | 0.6825 | 0.3654 | 0.1322 | 0.2534 | 0.5174 | 0.0318 | 24624 | 8 |
| tree_proxy_ablated | test | 0.8283 | 0.8236 | 0.7823 | 0.8248 | 0.6437 | 0.0484 | 0.4723 | 0.5252 | 0.0233 | 24624 | 249 |
| tree_selected_full_fit | test | 0.8397 | 0.8527 | 0.8060 | 0.7878 | 0.7607 | 0.0261 | 0.6814 | 0.2234 | 0.0198 | 24624 | 16 |
| logistic_selected | test | 0.7936 | 0.8205 | 0.7790 | 0.7794 | 0.6653 | 0.0416 | 0.4972 | 0.4216 | 0.0569 | 24624 | 16 |
| dense_selected | test | 0.8538 | 0.8757 | 0.8121 | 0.7874 | 0.7704 | 0.0247 | 0.6937 | 0.3022 | 0.0345 | 24624 | 16 |
| tcn_selected | test | 0.8505 | 0.8741 | 0.8277 | 0.8189 | 0.7906 | 0.0230 | 0.7099 | 0.2469 | 0.0337 | 24624 | 16 |
| dense_pca_90 | test | 0.8791 | 0.9019 | 0.8214 | 0.8096 | 0.8055 | 0.0210 | 0.7252 | 0.2006 | 0.0270 | 24624 | 256 |
| dense_pca_95 | test | 0.8758 | 0.9007 | 0.8233 | 0.8078 | 0.7979 | 0.0220 | 0.7217 | 0.2084 | 0.0255 | 24624 | 256 |
| dense_pca_99 | test | 0.8907 | 0.9158 | 0.8462 | 0.8491 | 0.8103 | 0.0214 | 0.7247 | 0.2641 | 0.0223 | 24624 | 256 |
| dense_autoencoder | test | 0.6677 | 0.7119 | 0.6342 | 0.6393 | 0.5657 | 0.0537 | 0.4771 | 0.4473 | 0.0642 | 24624 | 256 |

## Limits
Synthetic fault detection only; predictions do not confirm real root causes. Health is a sensor heuristic. RUL unavailable.
Weak effects, partial matrix signatures, only three test trucks, route/time proxies and sensor masks limit generalization.
Calibration uses one separate validation truck (not field calibration). AE/PCA dimensions do not imply fewer physical sensors.
Missing-channel stress is partial masking of selected derived inputs, not a complete sensor-failure replay. SHAP was not run.

## Reproduce
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /tmp/opencode/fault-training/bin/python analysis/train_fault_models.py --output analysis/results_enhanced_multilabel --epochs 18
```
