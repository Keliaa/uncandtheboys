| option | split | composite | F1 | recall | precision | FA+ex/day | timeliness | type recall | class recall |
|---|---|---|---|---|---|---|---|---|---|
| D01_zscore | train | 41.908 | 0.585 | 0.462 | 0.8 | 0.061 | 0.375 | 0.0 | 0.423 |
| D01_zscore | validation | 35.797 | 0.462 | 0.333 | 0.75 | 0.071 | 0.333 | 0.0 | 0.333 |
| D01+E01+R01 | train | 43.004 | 0.545 | 0.462 | 0.667 | 0.122 | 0.39 | 0.115 | 0.385 |
| D01+E01+R01 | validation | 35.797 | 0.462 | 0.333 | 0.75 | 0.071 | 0.333 | 0.0 | 0.333 |
| +D04 | train | 52.103 | 0.653 | 0.615 | 0.696 | 0.143 | 0.536 | 0.154 | 0.538 |
| +D04 | validation | 42.976 | 0.571 | 0.444 | 0.8 | 0.071 | 0.444 | 0.0 | 0.444 |
| +D04+Q01 | train | 57.11 | 0.603 | 0.731 | 0.514 | 0.367 | 0.661 | 0.308 | 0.692 |
| +D04+Q01 | validation | 67.049 | 0.727 | 0.889 | 0.615 | 0.357 | 0.808 | 0.333 | 0.889 |
| +D04+Q01+D06 | train | 56.25 | 0.603 | 0.731 | 0.514 | 0.367 | 0.637 | 0.308 | 0.654 |
| +D04+Q01+D06 | validation | 72.925 | 0.857 | 1.0 | 0.75 | 0.214 | 0.783 | 0.333 | 1.0 |
| +D04+Q01+D06+E02 | train | 61.015 | 0.678 | 0.769 | 0.606 | 0.265 | 0.652 | 0.346 | 0.692 |
| +D04+Q01+D06+E02 | validation | 81.931 | 0.9 | 1.0 | 0.818 | 0.143 | 0.863 | 0.556 | 1.0 |
| +E02 rule gate | train | 60.517 | 0.667 | 0.769 | 0.588 | 0.286 | 0.652 | 0.346 | 0.692 |
| +E02 rule gate | validation | 81.931 | 0.9 | 1.0 | 0.818 | 0.143 | 0.863 | 0.556 | 1.0 |
| D03 PCA only (full system otherwise) | train | 63.963 | 0.636 | 0.808 | 0.525 | 0.388 | 0.72 | 0.462 | 0.769 |
| D03 PCA only (full system otherwise) | validation | 78.925 | 0.9 | 1.0 | 0.818 | 0.143 | 0.851 | 0.444 | 1.0 |
| D01 + D03 (option, off in frozen system) | train | 63.963 | 0.636 | 0.808 | 0.525 | 0.388 | 0.72 | 0.462 | 0.769 |
| D01 + D03 (option, off in frozen system) | validation | 81.43 | 0.9 | 1.0 | 0.818 | 0.143 | 0.838 | 0.556 | 1.0 |
| + V03 bearing alerts (option, off in frozen system) | train | 61.852 | 0.656 | 0.769 | 0.571 | 0.306 | 0.647 | 0.423 | 0.692 |
| + V03 bearing alerts (option, off in frozen system) | validation | 80.074 | 0.857 | 1.0 | 0.75 | 0.214 | 0.863 | 0.556 | 1.0 |
| + V02 spectral typing (option, off in frozen system) | train | 61.282 | 0.667 | 0.769 | 0.588 | 0.286 | 0.642 | 0.385 | 0.692 |
| + V02 spectral typing (option, off in frozen system) | validation | 84.709 | 0.9 | 1.0 | 0.818 | 0.143 | 0.863 | 0.667 | 1.0 |
| + T02 after-maintenance alerts (frozen system) | train | 63.111 | 0.667 | 0.808 | 0.568 | 0.327 | 0.676 | 0.423 | 0.731 |
| + T02 after-maintenance alerts (frozen system) | validation | 85.535 | 0.9 | 1.0 | 0.818 | 0.143 | 0.904 | 0.667 | 1.0 |
| T01 baseline: static (frozen) | train | 61.78 | 0.678 | 0.769 | 0.606 | 0.265 | 0.642 | 0.385 | 0.692 |
| T01 baseline: static (frozen) | validation | 81.931 | 0.9 | 1.0 | 0.818 | 0.143 | 0.863 | 0.556 | 1.0 |
| T01 baseline: always learning | train | 54.045 | 0.581 | 0.692 | 0.5 | 0.367 | 0.586 | 0.308 | 0.615 |
| T01 baseline: always learning | validation | 63.221 | 0.737 | 0.778 | 0.7 | 0.214 | 0.62 | 0.333 | 0.778 |
| T01 baseline: guarded | train | 61.181 | 0.667 | 0.769 | 0.588 | 0.286 | 0.637 | 0.385 | 0.692 |
| T01 baseline: guarded | validation | 81.931 | 0.9 | 1.0 | 0.818 | 0.143 | 0.863 | 0.556 | 1.0 |
| B0_scada_alarms | train | 27.59 | 0.308 | 0.231 | 0.462 | 0.143 | 0.213 | 0.038 | 0.231 |
| B0_scada_alarms | validation | 23.111 | 0.2 | 0.111 | 1.0 | 0.0 | 0.111 | 0.111 | 0.111 |
| B1_mode_aware_z | train | 58.246 | 0.72 | 0.692 | 0.75 | 0.122 | 0.577 | 0.269 | 0.538 |
| B1_mode_aware_z | validation | 78.013 | 0.889 | 0.889 | 0.889 | 0.071 | 0.863 | 0.444 | 0.889 |
| B2_cusum | train | 53.299 | 0.484 | 0.885 | 0.333 | 0.939 | 0.812 | 0.269 | 0.808 |
| B2_cusum | validation | 50.918 | 0.452 | 0.778 | 0.318 | 1.071 | 0.773 | 0.333 | 0.667 |
| B3_isolation_forest | train | 39.371 | 0.468 | 0.423 | 0.524 | 0.204 | 0.412 | 0.077 | 0.385 |
| B3_isolation_forest | validation | 69.162 | 0.737 | 0.778 | 0.7 | 0.214 | 0.778 | 0.444 | 0.778 |
| stress mean (T03): B1 baseline | stress | 52.103 | 0.62 | 0.567 | 0.75 | 0.113 | 0.515 | 0.246 | 0.453 |
| stress mean (T03): frozen system | stress | 57.873 | 0.659 | 0.712 | 0.618 | 0.184 | 0.618 | 0.318 | 0.541 |
