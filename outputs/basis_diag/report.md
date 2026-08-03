# Basis centering / thickness diagnosis

## Geometric features (mesh thickness + skeleton topology)

SDF = classic surface shape-diameter samples (mesh-derived). Skeletons have no radius; `sk_closest_diam` is `2*|signed_distance|` at skeleton nodes (underestimates if off-medial).

| spine | bbox_diagonal | watertight | sdf_med | sdf_cv | sk_closest_diam_med | sk_anchored_diam_med | mel_0.1D | mel_0.1D_over_sdf_med | D_over_sdf_med | skel_length | cyclomatic | skel_terminals | skel_branches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2710 | 1 | 78.99 | 0.5468 | 46.19 | 105.6 | 271 | 3.431 | 34.31 | 8829 | 8 | 17 | 25 |
| 2 | 616.8 | 1 | 75.84 | 0.4331 | 48.05 | 92.03 | 61.68 | 0.8134 | 8.134 | 657.8 | 1 | 4 | 4 |
| 3 | 1911 | 0 | 95.2 | 0.6719 | 58.77 | 125.9 | 191.1 | 2.008 | 20.08 | 1.039e+04 | 10 | 36 | 47 |

## Basis opt: batch vs notebook (mel_fraction=0.1 unless noted)

| spine | label | mel | pre_ratio_med | post_ratio_med | post_ratio_p90 | force_fallback_events | n_nodes_out | n_terminals | nodes_outside | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | batch_mel0.1 | 271 | 5.937 | 4.3 | 8.614 | 0 | 36 | 7 | 0 | True |
| 1 | notebook_mel0.06 | 162.6 | 7.03 | 7.191 | 16.48 | 0 | 39 | 7 | 0 | True |
| 2 | batch_mel0.1 | 61.68 | 6.086 | 4.211 | 9.345 | 0 | 13 | 2 | 0 | True |
| 2 | notebook_mel0.06 | 37.01 | 4.869 | 4.053 | 14.28 | 0 | 17 | 3 | 0 | True |
| 3 | batch_mel0.1 | 191.1 | 4.957 | 3.883 | 10.99 | 0 | 54 | 10 | 0 | True |
| 3 | notebook_mel0.06 | 114.7 | 5.037 | 5.14 | 14.73 | 0 | 69 | 17 | 0 | True |

## TS2 param sweep (lower post_ratio_med is better centering)

| label | post_ratio_med | post_ratio_p90 | force_fallback_events | n_nodes_out | ok |
| --- | --- | --- | --- | --- | --- |
| nr24_ss0.5_jit0.1_b2.0 | 3.423 | 6.202 | 0 | 14 | True |
| nr12_ss0.5_jit0.0_b2.0 | 3.562 | 6.607 | 0 | 15 | True |
| sdf_k1.5 | 3.603 | 4.532 | 0 | 14 | True |
| nr12_ss0.5_jit0.1_b2.0 | 3.61 | 6.64 | 0 | 15 | True |
| nr24_ss0.5_jit0.0_b2.0 | 3.625 | 6.509 | 0 | 13 | True |
| sdf_k2.5 | 3.741 | 5.889 | 0 | 16 | True |
| nr24_ss0.5_jit0.1_b1.0 | 3.742 | 6.533 | 0 | 13 | True |
| nr6_ss0.5_jit0.1_b1.0 | 3.747 | 6.129 | 0 | 15 | True |
| nr24_ss0.5_jit0.0_b1.0 | 3.748 | 6.519 | 0 | 13 | True |
| sdf_k3.5 | 3.752 | 5.901 | 0 | 16 | True |
| sdf_k3.0 | 3.753 | 5.926 | 0 | 16 | True |
| sdf_k2.0 | 3.772 | 5.92 | 0 | 16 | True |
| nr12_ss0.5_jit0.0_b1.0 | 3.807 | 6.522 | 0 | 14 | True |
| move_terminals | 3.808 | 8.266 | 0 | 12 | True |
| nr12_ss0.5_jit0.1_b1.0 | 3.85 | 6.562 | 0 | 14 | True |
| nr6_ss0.5_jit0.0_b1.0 | 3.863 | 6.18 | 0 | 15 | True |
| nr12_ss0.25_jit0.1_b2.0 | 3.901 | 7.955 | 0 | 15 | True |
| nr6_ss0.5_jit0.1_b2.0 | 3.904 | 6.019 | 0 | 15 | True |
| nr12_ss0.25_jit0.0_b2.0 | 3.916 | 8.001 | 0 | 15 | True |
| nr12_ss0.25_jit0.1_b1.0 | 4.013 | 8.18 | 0 | 14 | True |
