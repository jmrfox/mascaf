# TS pipeline validation

Spines: TS1, TS2, TS3

## Run / optimizer summary

| spine | mel | basis_n | opt_n | opt_e | outside | total_len |
| --- | --- | --- | --- | --- | --- | --- |
| TS1 | 133.7682 | 88.0000 | 86.0000 | 92.0000 | 0.0000 | 8785 |
| TS2 | 54.8127 | 17.0000 | 16.0000 | 16.0000 | 0.0000 | 679.1425 |
| TS3 | 91.9565 | 152.0000 | 129.0000 | 137.0000 | 0.0000 | 8826 |

## Geometry (account_for_overlaps=False)

| spine | pre_vol_ratio | pre_vol_rel_err | pre_area_ratio | pre_area_rel_err | post_vol_ratio | post_vol_rel_err | post_area_ratio | post_area_rel_err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TS1 | 2.3418 | 134.18% | 1.3865 | 38.65% | 1.3117 | 31.17% | 1.0000 | -0.00% |
| TS2 | 1.3005 | 30.05% | 1.0259 | 2.59% | 1.2443 | 24.43% | 1.0000 | 0.00% |
| TS3 | 2.4296 | 142.96% | 1.7253 | 72.53% | 1.0305 | 3.05% | 1.0000 | -0.00% |

## Geometry (account_for_overlaps=True when branches present)

| spine | pre_vol_ratio_ov | pre_area_ratio_ov | post_vol_ratio_ov | post_area_ratio_ov |
| --- | --- | --- | --- | --- |
| TS1 | 1.9140 | 1.1389 | 1.1324 | 0.8613 |
| TS2 | 1.1295 | 0.8949 | 1.0843 | 0.8746 |
| TS3 | 1.7673 | 1.3462 | 0.8475 | 0.8392 |

