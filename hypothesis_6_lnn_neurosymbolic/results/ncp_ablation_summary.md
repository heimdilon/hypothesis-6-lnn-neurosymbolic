# NCP / MLP ablation sonuçları (istatistiksel analiz dahil)

Bu dosya resmi `ncps.torch` CfC/LTC katmanları ve MLP baseline ile üretilen offline imitation,
kısa policy-gradient fine-tune, pure/residual ablation ve istatistiksel karşılaştırma sonuçlarını özetler.

## Deney konfigürasyonu

- Bağımsız eğitim seed sayısı: 10
- Eğitim senaryoları: train_like, shifted_clutter, narrow_gate, u_trap
- Test senaryoları: train_like, shifted_clutter, narrow_gate, u_trap, zigzag_corridor, dense_maze, deceptive_u_trap, sensor_shadow, labyrinth_maze
- Imitation sequence sayısı: 48, doğrulama: 12, sequence length: 24
- Imitation epoch: 5, RL fine-tune episode: 6
- NCP hidden: 24, sparsity: 0.5, residual scale: 0.35
- Değerlendirme episode: 8 (senaryo başına)

## Eğitim özeti (son seed)

| cell | phase | epoch_or_episode | loss | val_accuracy | episode_return | success | collision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cfc | imitation | 5 | 1.8289 | 0.177 |  |  |  |
| cfc | rl_finetune | 6 | -0.4727 |  | -7.040 | 0 | 1 |
| ltc | imitation | 5 | 1.8277 | 0.608 |  |  |  |
| ltc | rl_finetune | 6 | -0.0388 |  | -8.837 | 0 | 1 |
| mlp | imitation | 5 | 1.8358 | 0.191 |  |  |  |
| mlp | rl_finetune | 6 | -0.4206 |  | -3.671 | 0 | 1 |

## Residual vs pure fark tablosu

| cell | stage | scenario_group | pure_success | residual_success | delta_success | pure_collision | residual_collision | delta_collision | delta_min_clearance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc | imitation | default | 0.000 | 0.934 | 0.934 | 0.859 | 0.009 | -0.850 | 0.527 |
| cfc | imitation | hard | 0.000 | 0.370 | 0.370 | 0.940 | 0.630 | -0.310 | 0.124 |
| cfc | rl_finetune | default | 0.000 | 0.919 | 0.919 | 0.850 | 0.000 | -0.850 | 0.527 |
| cfc | rl_finetune | hard | 0.000 | 0.375 | 0.375 | 0.943 | 0.618 | -0.325 | 0.134 |
| ltc | imitation | default | 0.000 | 0.897 | 0.897 | 0.991 | 0.016 | -0.975 | 0.567 |
| ltc | imitation | hard | 0.000 | 0.355 | 0.355 | 0.993 | 0.642 | -0.350 | 0.141 |
| ltc | rl_finetune | default | 0.000 | 0.909 | 0.909 | 0.991 | 0.016 | -0.975 | 0.567 |
| ltc | rl_finetune | hard | 0.000 | 0.355 | 0.355 | 0.993 | 0.640 | -0.353 | 0.142 |
| mlp | imitation | default | 0.000 | 0.944 | 0.944 | 1.000 | 0.003 | -0.997 | 0.538 |
| mlp | imitation | hard | 0.000 | 0.347 | 0.347 | 1.000 | 0.650 | -0.350 | 0.139 |
| mlp | rl_finetune | default | 0.000 | 0.912 | 0.912 | 1.000 | 0.009 | -0.991 | 0.550 |
| mlp | rl_finetune | hard | 0.000 | 0.338 | 0.338 | 1.000 | 0.652 | -0.348 | 0.139 |

## CfC vs LTC fark tablosu

| stage | variant | scenario_group | cfc_success | ltc_success | delta_success_cfc_minus_ltc | cfc_collision | ltc_collision | delta_collision_cfc_minus_ltc | delta_min_clearance_cfc_minus_ltc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imitation | pure | default | 0.000 | 0.000 | 0.000 | 0.859 | 0.991 | -0.131 | 0.051 |
| imitation | pure | hard | 0.000 | 0.000 | 0.000 | 0.940 | 0.993 | -0.053 | 0.021 |
| imitation | residual | default | 0.934 | 0.897 | 0.037 | 0.009 | 0.016 | -0.006 | 0.011 |
| imitation | residual | hard | 0.370 | 0.355 | 0.015 | 0.630 | 0.642 | -0.012 | 0.005 |
| rl_finetune | pure | default | 0.000 | 0.000 | 0.000 | 0.850 | 0.991 | -0.141 | 0.059 |
| rl_finetune | pure | hard | 0.000 | 0.000 | 0.000 | 0.943 | 0.993 | -0.050 | 0.020 |
| rl_finetune | residual | default | 0.919 | 0.909 | 0.009 | 0.000 | 0.016 | -0.016 | 0.019 |
| rl_finetune | residual | hard | 0.375 | 0.355 | 0.020 | 0.618 | 0.640 | -0.022 | 0.012 |

## Hard map ortalaması (95% CI dahil)

| controller | cell | stage | variant | n | success_rate | success_ci_lower | success_ci_upper | collision_rate | mean_steps | mean_min_clearance | mean_near_misses | mean_action_disagreements | mean_beneficial_disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc_imitation_pure | cfc | imitation | pure | 400 | 0.000 | 0.000 | 0.010 | 0.940 | 17.4 | -0.102 | 4.3 | 15.5 | 6.9 |
| cfc_imitation_residual | cfc | imitation | residual | 400 | 0.370 | 0.324 | 0.418 | 0.630 | 40.7 | 0.022 | 6.9 | 11.6 | 10.2 |
| cfc_random_residual | cfc | random | residual | 400 | 0.365 | 0.319 | 0.413 | 0.627 | 41.0 | 0.018 | 6.8 | 4.7 | 4.0 |
| cfc_rl_finetune_pure | cfc | rl_finetune | pure | 400 | 0.000 | 0.000 | 0.010 | 0.943 | 17.7 | -0.101 | 4.3 | 15.8 | 7.4 |
| cfc_rl_finetune_residual | cfc | rl_finetune | residual | 400 | 0.375 | 0.329 | 0.423 | 0.618 | 40.6 | 0.032 | 6.6 | 11.6 | 10.1 |
| fixed_policy | fixed | fixed | baseline | 400 | 0.370 | 0.324 | 0.418 | 0.623 | 41.3 | 0.027 | 6.8 | 0.0 | 0.0 |
| ltc_imitation_pure | ltc | imitation | pure | 400 | 0.000 | 0.000 | 0.010 | 0.993 | 9.3 | -0.123 | 3.2 | 7.7 | 4.2 |
| ltc_imitation_residual | ltc | imitation | residual | 400 | 0.355 | 0.310 | 0.403 | 0.642 | 41.2 | 0.018 | 6.3 | 9.8 | 8.6 |
| ltc_random_residual | ltc | random | residual | 400 | 0.362 | 0.317 | 0.411 | 0.637 | 41.2 | 0.020 | 6.6 | 8.3 | 7.1 |
| ltc_rl_finetune_pure | ltc | rl_finetune | pure | 400 | 0.000 | 0.000 | 0.010 | 0.993 | 9.2 | -0.122 | 3.2 | 7.6 | 4.2 |
| ltc_rl_finetune_residual | ltc | rl_finetune | residual | 400 | 0.355 | 0.310 | 0.403 | 0.640 | 40.6 | 0.020 | 6.1 | 9.6 | 8.5 |
| mlp_imitation_pure | mlp | imitation | pure | 400 | 0.000 | 0.000 | 0.010 | 1.000 | 8.2 | -0.126 | 2.9 | 7.3 | 3.8 |
| mlp_imitation_residual | mlp | imitation | residual | 400 | 0.347 | 0.302 | 0.395 | 0.650 | 39.9 | 0.013 | 6.4 | 9.9 | 8.7 |
| mlp_rl_finetune_pure | mlp | rl_finetune | pure | 400 | 0.000 | 0.000 | 0.010 | 1.000 | 8.1 | -0.126 | 2.7 | 7.0 | 3.9 |
| mlp_rl_finetune_residual | mlp | rl_finetune | residual | 400 | 0.338 | 0.293 | 0.385 | 0.652 | 39.3 | 0.013 | 6.1 | 10.2 | 9.0 |

## Default map ortalaması (95% CI dahil)

| controller | cell | stage | variant | n | success_rate | success_ci_lower | success_ci_upper | collision_rate | mean_steps | mean_min_clearance | mean_near_misses | mean_action_disagreements | mean_beneficial_disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc_imitation_pure | cfc | imitation | pure | 320 | 0.000 | 0.000 | 0.012 | 0.859 | 29.8 | -0.071 | 5.9 | 25.7 | 8.8 |
| cfc_imitation_residual | cfc | imitation | residual | 320 | 0.934 | 0.902 | 0.957 | 0.009 | 57.3 | 0.457 | 0.7 | 20.4 | 16.3 |
| cfc_random_residual | cfc | random | residual | 320 | 0.916 | 0.880 | 0.941 | 0.006 | 59.8 | 0.458 | 0.7 | 9.5 | 7.4 |
| cfc_rl_finetune_pure | cfc | rl_finetune | pure | 320 | 0.000 | 0.000 | 0.012 | 0.850 | 30.1 | -0.061 | 5.8 | 26.3 | 9.3 |
| cfc_rl_finetune_residual | cfc | rl_finetune | residual | 320 | 0.919 | 0.884 | 0.944 | 0.000 | 58.1 | 0.467 | 0.8 | 20.5 | 16.6 |
| fixed_policy | fixed | fixed | baseline | 320 | 0.878 | 0.838 | 0.910 | 0.003 | 62.3 | 0.459 | 0.8 | 0.0 | 0.0 |
| ltc_imitation_pure | ltc | imitation | pure | 320 | 0.000 | 0.000 | 0.012 | 0.991 | 11.3 | -0.122 | 2.9 | 9.2 | 4.7 |
| ltc_imitation_residual | ltc | imitation | residual | 320 | 0.897 | 0.859 | 0.926 | 0.016 | 61.4 | 0.446 | 1.0 | 17.3 | 13.9 |
| ltc_random_residual | ltc | random | residual | 320 | 0.884 | 0.845 | 0.915 | 0.000 | 61.3 | 0.454 | 0.8 | 15.3 | 12.3 |
| ltc_rl_finetune_pure | ltc | rl_finetune | pure | 320 | 0.000 | 0.000 | 0.012 | 0.991 | 11.3 | -0.120 | 2.9 | 9.2 | 4.7 |
| ltc_rl_finetune_residual | ltc | rl_finetune | residual | 320 | 0.909 | 0.873 | 0.936 | 0.016 | 61.6 | 0.447 | 0.9 | 17.5 | 14.1 |
| mlp_imitation_pure | mlp | imitation | pure | 320 | 0.000 | 0.000 | 0.012 | 1.000 | 9.4 | -0.130 | 2.5 | 8.2 | 3.0 |
| mlp_imitation_residual | mlp | imitation | residual | 320 | 0.944 | 0.913 | 0.964 | 0.003 | 58.2 | 0.409 | 1.2 | 16.1 | 13.0 |
| mlp_rl_finetune_pure | mlp | rl_finetune | pure | 320 | 0.000 | 0.000 | 0.012 | 1.000 | 9.2 | -0.134 | 2.3 | 7.9 | 3.5 |
| mlp_rl_finetune_residual | mlp | rl_finetune | residual | 320 | 0.912 | 0.876 | 0.939 | 0.009 | 59.3 | 0.416 | 1.1 | 17.0 | 13.6 |

## İstatistiksel karşılaştırmalar (Mann-Whitney U, BH düzeltmeli)

| scenario_group | comparison | controller_a | controller_b | n_a | n_b | small_n_warning | mean_a | mean_b | delta | mann_whitney_U | p_value_raw | p_value_bh_corrected | cohens_d | significant_005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | residual_vs_pure | cfc_imitation_residual | cfc_imitation_pure | 320 | 320 | no | 0.934 | 0.000 | 0.934 | 99040.0 | 0.0000 | 0.0000 | 5.328 | yes |
| default | ncp_vs_fixed | cfc_imitation_residual | fixed_policy | 320 | 320 | no | 0.934 | 0.878 | 0.056 | 54080.0 | 0.0147 | 0.0478 | 0.194 | yes |
| default | residual_vs_pure | cfc_rl_finetune_residual | cfc_rl_finetune_pure | 320 | 320 | no | 0.919 | 0.000 | 0.919 | 98240.0 | 0.0000 | 0.0000 | 4.748 | yes |
| default | ncp_vs_fixed | cfc_rl_finetune_residual | fixed_policy | 320 | 320 | no | 0.919 | 0.878 | 0.041 | 53280.0 | 0.0892 | 0.2108 | 0.135 | no |
| default | trained_vs_random | cfc_rl_finetune_residual | cfc_random_residual | 320 | 320 | no | 0.919 | 0.916 | 0.003 | 51360.0 | 0.8860 | 1.0000 | 0.011 | no |
| default | residual_vs_pure | ltc_imitation_residual | ltc_imitation_pure | 320 | 320 | no | 0.897 | 0.000 | 0.897 | 97120.0 | 0.0000 | 0.0000 | 4.164 | yes |
| default | ncp_vs_fixed | ltc_imitation_residual | fixed_policy | 320 | 320 | no | 0.897 | 0.878 | 0.019 | 52160.0 | 0.4533 | 0.7856 | 0.059 | no |
| default | residual_vs_pure | ltc_rl_finetune_residual | ltc_rl_finetune_pure | 320 | 320 | no | 0.909 | 0.000 | 0.909 | 97760.0 | 0.0000 | 0.0000 | 4.473 | yes |
| default | ncp_vs_fixed | ltc_rl_finetune_residual | fixed_policy | 320 | 320 | no | 0.909 | 0.878 | 0.031 | 52800.0 | 0.1999 | 0.3999 | 0.101 | no |
| default | trained_vs_random | ltc_rl_finetune_residual | ltc_random_residual | 320 | 320 | no | 0.909 | 0.884 | 0.025 | 52480.0 | 0.2988 | 0.5549 | 0.082 | no |
| default | residual_vs_pure | mlp_imitation_residual | mlp_imitation_pure | 320 | 320 | no | 0.944 | 0.000 | 0.944 | 99520.0 | 0.0000 | 0.0000 | 5.784 | yes |
| default | mlp_vs_fixed | mlp_imitation_residual | fixed_policy | 320 | 320 | no | 0.944 | 0.878 | 0.066 | 54560.0 | 0.0036 | 0.0133 | 0.232 | yes |
| default | residual_vs_pure | mlp_rl_finetune_residual | mlp_rl_finetune_pure | 320 | 320 | no | 0.912 | 0.000 | 0.912 | 97920.0 | 0.0000 | 0.0000 | 4.560 | yes |
| default | mlp_vs_fixed | mlp_rl_finetune_residual | fixed_policy | 320 | 320 | no | 0.912 | 0.878 | 0.034 | 52960.0 | 0.1559 | 0.3377 | 0.112 | no |
| default | ncp_vs_mlp | cfc_imitation_pure | mlp_imitation_pure | 320 | 320 | no | 0.000 | 0.000 | 0.000 | 51200.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | cfc_imitation_residual | mlp_imitation_residual | 320 | 320 | no | 0.934 | 0.944 | -0.009 | 50720.0 | 0.6204 | 1.0000 | -0.039 | no |
| default | ncp_vs_mlp | cfc_rl_finetune_pure | mlp_rl_finetune_pure | 320 | 320 | no | 0.000 | 0.000 | 0.000 | 51200.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | cfc_rl_finetune_residual | mlp_rl_finetune_residual | 320 | 320 | no | 0.919 | 0.912 | 0.006 | 51520.0 | 0.7763 | 1.0000 | 0.022 | no |
| default | ncp_vs_mlp | ltc_imitation_pure | mlp_imitation_pure | 320 | 320 | no | 0.000 | 0.000 | 0.000 | 51200.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | ltc_imitation_residual | mlp_imitation_residual | 320 | 320 | no | 0.897 | 0.944 | -0.047 | 48800.0 | 0.0287 | 0.0829 | -0.173 | no |
| default | ncp_vs_mlp | ltc_rl_finetune_pure | mlp_rl_finetune_pure | 320 | 320 | no | 0.000 | 0.000 | 0.000 | 51200.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | ltc_rl_finetune_residual | mlp_rl_finetune_residual | 320 | 320 | no | 0.909 | 0.912 | -0.003 | 51040.0 | 0.8897 | 1.0000 | -0.011 | no |
| default | cfc_vs_ltc | cfc_imitation_pure | ltc_imitation_pure | 320 | 320 | no | 0.000 | 0.000 | 0.000 | 51200.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | cfc_vs_ltc | cfc_imitation_residual | ltc_imitation_residual | 320 | 320 | no | 0.934 | 0.897 | 0.037 | 53120.0 | 0.0881 | 0.2108 | 0.135 | no |
| default | cfc_vs_ltc | cfc_rl_finetune_pure | ltc_rl_finetune_pure | 320 | 320 | no | 0.000 | 0.000 | 0.000 | 51200.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | cfc_vs_ltc | cfc_rl_finetune_residual | ltc_rl_finetune_residual | 320 | 320 | no | 0.919 | 0.909 | 0.009 | 51680.0 | 0.6725 | 1.0000 | 0.033 | no |
| hard | residual_vs_pure | cfc_imitation_residual | cfc_imitation_pure | 400 | 400 | no | 0.370 | 0.000 | 0.370 | 109600.0 | 0.0000 | 0.0000 | 1.082 | yes |
| hard | ncp_vs_fixed | cfc_imitation_residual | fixed_policy | 400 | 400 | no | 0.370 | 0.370 | 0.000 | 80000.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | residual_vs_pure | cfc_rl_finetune_residual | cfc_rl_finetune_pure | 400 | 400 | no | 0.375 | 0.000 | 0.375 | 110000.0 | 0.0000 | 0.0000 | 1.094 | yes |
| hard | ncp_vs_fixed | cfc_rl_finetune_residual | fixed_policy | 400 | 400 | no | 0.375 | 0.370 | 0.005 | 80400.0 | 0.8838 | 1.0000 | 0.010 | no |
| hard | trained_vs_random | cfc_rl_finetune_residual | cfc_random_residual | 400 | 400 | no | 0.375 | 0.365 | 0.010 | 80800.0 | 0.7697 | 1.0000 | 0.021 | no |
| hard | residual_vs_pure | ltc_imitation_residual | ltc_imitation_pure | 400 | 400 | no | 0.355 | 0.000 | 0.355 | 108400.0 | 0.0000 | 0.0000 | 1.048 | yes |
| hard | ncp_vs_fixed | ltc_imitation_residual | fixed_policy | 400 | 400 | no | 0.355 | 0.370 | -0.015 | 78800.0 | 0.6592 | 1.0000 | -0.031 | no |
| hard | residual_vs_pure | ltc_rl_finetune_residual | ltc_rl_finetune_pure | 400 | 400 | no | 0.355 | 0.000 | 0.355 | 108400.0 | 0.0000 | 0.0000 | 1.048 | yes |
| hard | ncp_vs_fixed | ltc_rl_finetune_residual | fixed_policy | 400 | 400 | no | 0.355 | 0.370 | -0.015 | 78800.0 | 0.6592 | 1.0000 | -0.031 | no |
| hard | trained_vs_random | ltc_rl_finetune_residual | ltc_random_residual | 400 | 400 | no | 0.355 | 0.362 | -0.008 | 79400.0 | 0.8251 | 1.0000 | -0.016 | no |
| hard | residual_vs_pure | mlp_imitation_residual | mlp_imitation_pure | 400 | 400 | no | 0.347 | 0.000 | 0.347 | 107800.0 | 0.0000 | 0.0000 | 1.031 | yes |
| hard | mlp_vs_fixed | mlp_imitation_residual | fixed_policy | 400 | 400 | no | 0.347 | 0.370 | -0.023 | 78200.0 | 0.5073 | 1.0000 | -0.047 | no |
| hard | residual_vs_pure | mlp_rl_finetune_residual | mlp_rl_finetune_pure | 400 | 400 | no | 0.338 | 0.000 | 0.338 | 107000.0 | 0.0000 | 0.0000 | 1.008 | yes |
| hard | mlp_vs_fixed | mlp_rl_finetune_residual | fixed_policy | 400 | 400 | no | 0.338 | 0.370 | -0.032 | 77400.0 | 0.3367 | 1.0000 | -0.068 | no |
| hard | ncp_vs_mlp | cfc_imitation_pure | mlp_imitation_pure | 400 | 400 | no | 0.000 | 0.000 | 0.000 | 80000.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | cfc_imitation_residual | mlp_imitation_residual | 400 | 400 | no | 0.370 | 0.347 | 0.023 | 81800.0 | 0.5073 | 1.0000 | 0.047 | no |
| hard | ncp_vs_mlp | cfc_rl_finetune_pure | mlp_rl_finetune_pure | 400 | 400 | no | 0.000 | 0.000 | 0.000 | 80000.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | cfc_rl_finetune_residual | mlp_rl_finetune_residual | 400 | 400 | no | 0.375 | 0.338 | 0.037 | 83000.0 | 0.2684 | 0.9970 | 0.078 | no |
| hard | ncp_vs_mlp | ltc_imitation_pure | mlp_imitation_pure | 400 | 400 | no | 0.000 | 0.000 | 0.000 | 80000.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | ltc_imitation_residual | mlp_imitation_residual | 400 | 400 | no | 0.355 | 0.347 | 0.008 | 80600.0 | 0.8243 | 1.0000 | 0.016 | no |
| hard | ncp_vs_mlp | ltc_rl_finetune_pure | mlp_rl_finetune_pure | 400 | 400 | no | 0.000 | 0.000 | 0.000 | 80000.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | ltc_rl_finetune_residual | mlp_rl_finetune_residual | 400 | 400 | no | 0.355 | 0.338 | 0.017 | 81400.0 | 0.6032 | 1.0000 | 0.037 | no |
| hard | cfc_vs_ltc | cfc_imitation_pure | ltc_imitation_pure | 400 | 400 | no | 0.000 | 0.000 | 0.000 | 80000.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | cfc_vs_ltc | cfc_imitation_residual | ltc_imitation_residual | 400 | 400 | no | 0.370 | 0.355 | 0.015 | 81200.0 | 0.6592 | 1.0000 | 0.031 | no |
| hard | cfc_vs_ltc | cfc_rl_finetune_pure | ltc_rl_finetune_pure | 400 | 400 | no | 0.000 | 0.000 | 0.000 | 80000.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | cfc_vs_ltc | cfc_rl_finetune_residual | ltc_rl_finetune_residual | 400 | 400 | no | 0.375 | 0.355 | 0.020 | 81600.0 | 0.5571 | 1.0000 | 0.041 | no |

## Bilimsel okuma

- `pure` varyantında karar tamamen NCP/MLP logits ile verilir; bu saf kapasite ve kararlılık testidir.
- `residual` varyantında sabit uzman politikanın üstüne NCP/MLP düzeltmesi eklenir; bu daha güvenli dağıtım senaryosunu temsil eder.
- `random_residual` eğitilmemiş (rastgele ağırlıklı) NCP ile residual yapıyı test eder; öğrenilmiş bilginin etkisini izole eder.
- `mlp` baseline, recurrent olmayan feedforward ağdır; NCP mimarisinin (sürekli zaman dinamikleri) etkisini ayırt etmeyi sağlar.
- CfC/LTC kıyası aynı veri, seed ailesi ve harita setinde yapılır; farklar mimari ve kısa fine-tune dinamiğinden gelir.
- 10 bağımsız seed ile eğitim yapılmış, sonuçlar tüm seed'ler üzerinden toplanmıştır.
- Wilson score interval (95% CI) küçük örneklem için uygun binomial güven aralığı sağlar.
- Benjamini-Hochberg FDR düzeltmesi çoklu karşılaştırma hatasını kontrol eder.
- `small_n_warning=yes`: Grup başına n<8 olduğunda Mann-Whitney U normal yaklaşımı güvenilirliğini yitirir; bu satırlardaki p-değerleri dikkatli yorumlanmalıdır.
- Aksiyon uyuşmazlığı (action disagreement): NCP/MLP'nin sabit politikadan farklı karar verdiği adım sayısı.
- Yararlı uyuşmazlık (beneficial disagreement): Farklı kararın daha iyi clearance veya ilerleme sağladığı adım sayısı.