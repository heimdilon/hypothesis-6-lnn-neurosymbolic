# NCP / MLP ablation sonuçları (istatistiksel analiz dahil)

Bu dosya resmi `ncps.torch` CfC/LTC katmanları ve MLP baseline ile üretilen offline imitation,
kısa policy-gradient fine-tune, pure/residual ablation ve istatistiksel karşılaştırma sonuçlarını özetler.

## Deney konfigürasyonu

- Bağımsız eğitim seed sayısı: 2
- Eğitim senaryoları: train_like, shifted_clutter, narrow_gate, u_trap
- Test senaryoları: train_like, shifted_clutter, narrow_gate, u_trap, zigzag_corridor, dense_maze, deceptive_u_trap, sensor_shadow, labyrinth_maze
- Imitation sequence sayısı: 24, doğrulama: 6, sequence length: 16
- Imitation epoch: 2, RL fine-tune episode: 3
- NCP hidden: 16, sparsity: 0.5, residual scale: 0.35
- Değerlendirme episode: 3 (senaryo başına)

## Eğitim özeti (son seed)

| cell | phase | epoch_or_episode | loss | val_accuracy | episode_return | success | collision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cfc | imitation | 2 | 1.9076 | 0.031 |  |  |  |
| cfc | rl_finetune | 3 | 0.3281 |  | 3.238 | 0 | 1 |
| ltc | imitation | 2 | 1.9607 | 0.000 |  |  |  |
| ltc | rl_finetune | 3 | -0.4427 |  | -10.116 | 0 | 1 |
| mlp | imitation | 2 | 1.9797 | 0.104 |  |  |  |
| mlp | rl_finetune | 3 | 0.0288 |  | -7.944 | 0 | 1 |

## Residual vs pure fark tablosu

| cell | stage | scenario_group | pure_success | residual_success | delta_success | pure_collision | residual_collision | delta_collision | delta_min_clearance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc | imitation | default | 0.000 | 0.958 | 0.958 | 0.708 | 0.000 | -0.708 | 0.419 |
| cfc | imitation | hard | 0.000 | 0.333 | 0.333 | 0.867 | 0.667 | -0.200 | 0.104 |
| cfc | rl_finetune | default | 0.000 | 0.958 | 0.958 | 0.708 | 0.000 | -0.708 | 0.436 |
| cfc | rl_finetune | hard | 0.000 | 0.267 | 0.267 | 0.867 | 0.733 | -0.133 | 0.070 |
| ltc | imitation | default | 0.000 | 1.000 | 1.000 | 0.958 | 0.000 | -0.958 | 0.497 |
| ltc | imitation | hard | 0.000 | 0.367 | 0.367 | 0.967 | 0.633 | -0.333 | 0.159 |
| ltc | rl_finetune | default | 0.000 | 1.000 | 1.000 | 0.958 | 0.000 | -0.958 | 0.510 |
| ltc | rl_finetune | hard | 0.000 | 0.400 | 0.400 | 0.967 | 0.600 | -0.367 | 0.160 |
| mlp | imitation | default | 0.000 | 0.917 | 0.917 | 1.000 | 0.042 | -0.958 | 0.592 |
| mlp | imitation | hard | 0.000 | 0.333 | 0.333 | 1.000 | 0.667 | -0.333 | 0.146 |
| mlp | rl_finetune | default | 0.000 | 0.833 | 0.833 | 1.000 | 0.042 | -0.958 | 0.588 |
| mlp | rl_finetune | hard | 0.000 | 0.333 | 0.333 | 1.000 | 0.667 | -0.333 | 0.146 |

## CfC vs LTC fark tablosu

| stage | variant | scenario_group | cfc_success | ltc_success | delta_success_cfc_minus_ltc | cfc_collision | ltc_collision | delta_collision_cfc_minus_ltc | delta_min_clearance_cfc_minus_ltc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imitation | pure | default | 0.000 | 0.000 | 0.000 | 0.708 | 0.958 | -0.250 | 0.034 |
| imitation | pure | hard | 0.000 | 0.000 | 0.000 | 0.867 | 0.967 | -0.100 | 0.014 |
| imitation | residual | default | 0.958 | 1.000 | -0.042 | 0.000 | 0.000 | 0.000 | -0.043 |
| imitation | residual | hard | 0.333 | 0.367 | -0.033 | 0.667 | 0.633 | 0.033 | -0.042 |
| rl_finetune | pure | default | 0.000 | 0.000 | 0.000 | 0.708 | 0.958 | -0.250 | 0.034 |
| rl_finetune | pure | hard | 0.000 | 0.000 | 0.000 | 0.867 | 0.967 | -0.100 | 0.016 |
| rl_finetune | residual | default | 0.958 | 1.000 | -0.042 | 0.000 | 0.000 | 0.000 | -0.040 |
| rl_finetune | residual | hard | 0.267 | 0.400 | -0.133 | 0.733 | 0.600 | 0.133 | -0.074 |

## Hard map ortalaması (95% CI dahil)

| controller | cell | stage | variant | n | success_rate | success_ci_lower | success_ci_upper | collision_rate | mean_steps | mean_min_clearance | mean_near_misses | mean_action_disagreements | mean_beneficial_disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc_imitation_pure | cfc | imitation | pure | 30 | 0.000 | 0.000 | 0.114 | 0.867 | 20.2 | -0.053 | 9.7 | 19.2 | 6.0 |
| cfc_imitation_residual | cfc | imitation | residual | 30 | 0.333 | 0.192 | 0.512 | 0.667 | 36.3 | 0.051 | 6.0 | 3.9 | 3.6 |
| cfc_random_residual | cfc | random | residual | 30 | 0.333 | 0.192 | 0.512 | 0.667 | 39.3 | 0.077 | 6.4 | 4.2 | 3.7 |
| cfc_rl_finetune_pure | cfc | rl_finetune | pure | 30 | 0.000 | 0.000 | 0.114 | 0.867 | 20.1 | -0.051 | 9.7 | 19.2 | 6.0 |
| cfc_rl_finetune_residual | cfc | rl_finetune | residual | 30 | 0.267 | 0.142 | 0.444 | 0.733 | 35.6 | 0.019 | 6.3 | 4.0 | 3.7 |
| fixed_policy | fixed | fixed | baseline | 30 | 0.367 | 0.219 | 0.545 | 0.633 | 38.3 | 0.079 | 6.6 | 0.0 | 0.0 |
| ltc_imitation_pure | ltc | imitation | pure | 30 | 0.000 | 0.000 | 0.114 | 0.967 | 10.1 | -0.067 | 4.1 | 9.3 | 1.8 |
| ltc_imitation_residual | ltc | imitation | residual | 30 | 0.367 | 0.219 | 0.545 | 0.633 | 35.8 | 0.093 | 6.0 | 2.3 | 2.1 |
| ltc_random_residual | ltc | random | residual | 30 | 0.300 | 0.167 | 0.479 | 0.700 | 33.1 | 0.051 | 6.0 | 3.9 | 3.5 |
| ltc_rl_finetune_pure | ltc | rl_finetune | pure | 30 | 0.000 | 0.000 | 0.114 | 0.967 | 10.1 | -0.067 | 4.1 | 9.3 | 1.8 |
| ltc_rl_finetune_residual | ltc | rl_finetune | residual | 30 | 0.400 | 0.246 | 0.577 | 0.600 | 36.3 | 0.093 | 6.2 | 2.3 | 2.1 |
| mlp_imitation_pure | mlp | imitation | pure | 30 | 0.000 | 0.000 | 0.114 | 1.000 | 8.1 | -0.119 | 2.9 | 7.3 | 4.6 |
| mlp_imitation_residual | mlp | imitation | residual | 30 | 0.333 | 0.192 | 0.512 | 0.667 | 34.4 | 0.026 | 6.8 | 6.2 | 5.3 |
| mlp_rl_finetune_pure | mlp | rl_finetune | pure | 30 | 0.000 | 0.000 | 0.114 | 1.000 | 10.9 | -0.119 | 3.5 | 9.8 | 5.5 |
| mlp_rl_finetune_residual | mlp | rl_finetune | residual | 30 | 0.333 | 0.192 | 0.512 | 0.667 | 35.5 | 0.027 | 7.2 | 6.2 | 5.3 |

## Default map ortalaması (95% CI dahil)

| controller | cell | stage | variant | n | success_rate | success_ci_lower | success_ci_upper | collision_rate | mean_steps | mean_min_clearance | mean_near_misses | mean_action_disagreements | mean_beneficial_disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc_imitation_pure | cfc | imitation | pure | 24 | 0.000 | 0.000 | 0.138 | 0.708 | 38.5 | -0.018 | 16.2 | 38.0 | 7.8 |
| cfc_imitation_residual | cfc | imitation | residual | 24 | 0.958 | 0.798 | 0.993 | 0.000 | 55.7 | 0.401 | 1.8 | 7.2 | 5.5 |
| cfc_random_residual | cfc | random | residual | 24 | 0.958 | 0.798 | 0.993 | 0.000 | 62.0 | 0.432 | 1.1 | 9.5 | 7.5 |
| cfc_rl_finetune_pure | cfc | rl_finetune | pure | 24 | 0.000 | 0.000 | 0.138 | 0.708 | 38.5 | -0.019 | 16.2 | 38.0 | 7.9 |
| cfc_rl_finetune_residual | cfc | rl_finetune | residual | 24 | 0.958 | 0.798 | 0.993 | 0.000 | 56.7 | 0.417 | 1.7 | 7.4 | 5.7 |
| fixed_policy | fixed | fixed | baseline | 24 | 0.833 | 0.641 | 0.933 | 0.000 | 59.0 | 0.406 | 1.4 | 0.0 | 0.0 |
| ltc_imitation_pure | ltc | imitation | pure | 24 | 0.000 | 0.000 | 0.138 | 0.958 | 9.7 | -0.053 | 3.9 | 9.7 | 1.4 |
| ltc_imitation_residual | ltc | imitation | residual | 24 | 1.000 | 0.862 | 1.000 | 0.000 | 49.5 | 0.444 | 1.5 | 4.1 | 3.0 |
| ltc_random_residual | ltc | random | residual | 24 | 0.958 | 0.798 | 0.993 | 0.000 | 58.9 | 0.475 | 1.3 | 7.9 | 6.2 |
| ltc_rl_finetune_pure | ltc | rl_finetune | pure | 24 | 0.000 | 0.000 | 0.138 | 0.958 | 9.7 | -0.053 | 3.9 | 9.7 | 1.4 |
| ltc_rl_finetune_residual | ltc | rl_finetune | residual | 24 | 1.000 | 0.862 | 1.000 | 0.000 | 50.7 | 0.457 | 1.3 | 4.1 | 3.0 |
| mlp_imitation_pure | mlp | imitation | pure | 24 | 0.000 | 0.000 | 0.138 | 1.000 | 8.8 | -0.135 | 2.2 | 7.4 | 3.8 |
| mlp_imitation_residual | mlp | imitation | residual | 24 | 0.917 | 0.742 | 0.977 | 0.042 | 58.7 | 0.457 | 1.1 | 9.8 | 8.2 |
| mlp_rl_finetune_pure | mlp | rl_finetune | pure | 24 | 0.000 | 0.000 | 0.138 | 1.000 | 8.8 | -0.146 | 2.2 | 7.5 | 3.8 |
| mlp_rl_finetune_residual | mlp | rl_finetune | residual | 24 | 0.833 | 0.641 | 0.933 | 0.042 | 56.7 | 0.442 | 1.1 | 8.5 | 6.9 |

## İstatistiksel karşılaştırmalar (Mann-Whitney U, BH düzeltmeli)

| scenario_group | comparison | controller_a | controller_b | n_a | n_b | mean_a | mean_b | delta | mann_whitney_U | p_value_raw | p_value_bh_corrected | cohens_d | significant_005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | residual_vs_pure | cfc_imitation_residual | cfc_imitation_pure | 24 | 24 | 0.958 | 0.000 | 0.958 | 564.0 | 0.0000 | 0.0000 | 6.640 | yes |
| default | ncp_vs_fixed | cfc_imitation_residual | fixed_policy | 24 | 24 | 0.958 | 0.833 | 0.125 | 324.0 | 0.1265 | 0.2070 | 0.409 | no |
| default | residual_vs_pure | cfc_rl_finetune_residual | cfc_rl_finetune_pure | 24 | 24 | 0.958 | 0.000 | 0.958 | 564.0 | 0.0000 | 0.0000 | 6.640 | yes |
| default | ncp_vs_fixed | cfc_rl_finetune_residual | fixed_policy | 24 | 24 | 0.958 | 0.833 | 0.125 | 324.0 | 0.1265 | 0.2070 | 0.409 | no |
| default | trained_vs_random | cfc_rl_finetune_residual | cfc_random_residual | 24 | 24 | 0.958 | 0.958 | 0.000 | 288.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | residual_vs_pure | ltc_imitation_residual | ltc_imitation_pure | 24 | 24 | 1.000 | 0.000 | 1.000 | 576.0 | 0.0000 | 0.0000 | 0.000 | yes |
| default | ncp_vs_fixed | ltc_imitation_residual | fixed_policy | 24 | 24 | 1.000 | 0.833 | 0.167 | 336.0 | 0.0293 | 0.0660 | 0.619 | no |
| default | residual_vs_pure | ltc_rl_finetune_residual | ltc_rl_finetune_pure | 24 | 24 | 1.000 | 0.000 | 1.000 | 576.0 | 0.0000 | 0.0000 | 0.000 | yes |
| default | ncp_vs_fixed | ltc_rl_finetune_residual | fixed_policy | 24 | 24 | 1.000 | 0.833 | 0.167 | 336.0 | 0.0293 | 0.0660 | 0.619 | no |
| default | trained_vs_random | ltc_rl_finetune_residual | ltc_random_residual | 24 | 24 | 1.000 | 0.958 | 0.042 | 300.0 | 0.2593 | 0.3890 | 0.289 | no |
| default | residual_vs_pure | mlp_imitation_residual | mlp_imitation_pure | 24 | 24 | 0.917 | 0.000 | 0.917 | 552.0 | 0.0000 | 0.0000 | 4.592 | yes |
| default | ncp_vs_fixed | mlp_imitation_residual | fixed_policy | 24 | 24 | 0.917 | 0.833 | 0.083 | 312.0 | 0.3222 | 0.4461 | 0.249 | no |
| default | residual_vs_pure | mlp_rl_finetune_residual | mlp_rl_finetune_pure | 24 | 24 | 0.833 | 0.000 | 0.833 | 528.0 | 0.0000 | 0.0000 | 3.096 | yes |
| default | ncp_vs_fixed | mlp_rl_finetune_residual | fixed_policy | 24 | 24 | 0.833 | 0.833 | 0.000 | 288.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | cfc_imitation_pure | mlp_imitation_pure | 24 | 24 | 0.000 | 0.000 | 0.000 | 288.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | cfc_imitation_residual | mlp_imitation_residual | 24 | 24 | 0.958 | 0.917 | 0.042 | 300.0 | 0.4809 | 0.6182 | 0.169 | no |
| default | ncp_vs_mlp | cfc_rl_finetune_pure | mlp_rl_finetune_pure | 24 | 24 | 0.000 | 0.000 | 0.000 | 288.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | cfc_rl_finetune_residual | mlp_rl_finetune_residual | 24 | 24 | 0.958 | 0.833 | 0.125 | 324.0 | 0.1265 | 0.2070 | 0.409 | no |
| hard | residual_vs_pure | cfc_imitation_residual | cfc_imitation_pure | 30 | 30 | 0.333 | 0.000 | 0.333 | 600.0 | 0.0004 | 0.0016 | 0.983 | yes |
| hard | ncp_vs_fixed | cfc_imitation_residual | fixed_policy | 30 | 30 | 0.333 | 0.367 | -0.033 | 435.0 | 0.7301 | 0.9445 | -0.069 | no |
| hard | residual_vs_pure | cfc_rl_finetune_residual | cfc_rl_finetune_pure | 30 | 30 | 0.267 | 0.000 | 0.267 | 570.0 | 0.0019 | 0.0057 | 0.838 | yes |
| hard | ncp_vs_fixed | cfc_rl_finetune_residual | fixed_policy | 30 | 30 | 0.267 | 0.367 | -0.100 | 405.0 | 0.3416 | 0.7929 | -0.213 | no |
| hard | trained_vs_random | cfc_rl_finetune_residual | cfc_random_residual | 30 | 30 | 0.267 | 0.333 | -0.067 | 420.0 | 0.5020 | 0.9036 | -0.143 | no |
| hard | residual_vs_pure | ltc_imitation_residual | ltc_imitation_pure | 30 | 30 | 0.367 | 0.000 | 0.367 | 615.0 | 0.0002 | 0.0016 | 1.058 | yes |
| hard | ncp_vs_fixed | ltc_imitation_residual | fixed_policy | 30 | 30 | 0.367 | 0.367 | 0.000 | 450.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | residual_vs_pure | ltc_rl_finetune_residual | ltc_rl_finetune_pure | 30 | 30 | 0.400 | 0.000 | 0.400 | 630.0 | 0.0001 | 0.0016 | 1.135 | yes |
| hard | ncp_vs_fixed | ltc_rl_finetune_residual | fixed_policy | 30 | 30 | 0.400 | 0.367 | 0.033 | 465.0 | 0.7346 | 0.9445 | 0.067 | no |
| hard | trained_vs_random | ltc_rl_finetune_residual | ltc_random_residual | 30 | 30 | 0.400 | 0.300 | 0.100 | 495.0 | 0.3524 | 0.7929 | 0.207 | no |
| hard | residual_vs_pure | mlp_imitation_residual | mlp_imitation_pure | 30 | 30 | 0.333 | 0.000 | 0.333 | 600.0 | 0.0004 | 0.0016 | 0.983 | yes |
| hard | ncp_vs_fixed | mlp_imitation_residual | fixed_policy | 30 | 30 | 0.333 | 0.367 | -0.033 | 435.0 | 0.7301 | 0.9445 | -0.069 | no |
| hard | residual_vs_pure | mlp_rl_finetune_residual | mlp_rl_finetune_pure | 30 | 30 | 0.333 | 0.000 | 0.333 | 600.0 | 0.0004 | 0.0016 | 0.983 | yes |
| hard | ncp_vs_fixed | mlp_rl_finetune_residual | fixed_policy | 30 | 30 | 0.333 | 0.367 | -0.033 | 435.0 | 0.7301 | 0.9445 | -0.069 | no |
| hard | ncp_vs_mlp | cfc_imitation_pure | mlp_imitation_pure | 30 | 30 | 0.000 | 0.000 | 0.000 | 450.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | cfc_imitation_residual | mlp_imitation_residual | 30 | 30 | 0.333 | 0.333 | 0.000 | 450.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | cfc_rl_finetune_pure | mlp_rl_finetune_pure | 30 | 30 | 0.000 | 0.000 | 0.000 | 450.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | cfc_rl_finetune_residual | mlp_rl_finetune_residual | 30 | 30 | 0.267 | 0.333 | -0.067 | 420.0 | 0.5020 | 0.9036 | -0.143 | no |

## Bilimsel okuma

- `pure` varyantında karar tamamen NCP/MLP logits ile verilir; bu saf kapasite ve kararlılık testidir.
- `residual` varyantında sabit uzman politikanın üstüne NCP/MLP düzeltmesi eklenir; bu daha güvenli dağıtım senaryosunu temsil eder.
- `random_residual` eğitilmemiş (rastgele ağırlıklı) NCP ile residual yapıyı test eder; öğrenilmiş bilginin etkisini izole eder.
- `mlp` baseline, recurrent olmayan feedforward ağdır; NCP mimarisinin (sürekli zaman dinamikleri) etkisini ayırt etmeyi sağlar.
- CfC/LTC kıyası aynı veri, seed ailesi ve harita setinde yapılır; farklar mimari ve kısa fine-tune dinamiğinden gelir.
- 2 bağımsız seed ile eğitim yapılmış, sonuçlar tüm seed'ler üzerinden toplanmıştır.
- Wilson score interval (95% CI) küçük örneklem için uygun binomial güven aralığı sağlar.
- Benjamini-Hochberg FDR düzeltmesi çoklu karşılaştırma hatasını kontrol eder.
- Aksiyon uyuşmazlığı (action disagreement): NCP/MLP'nin sabit politikadan farklı karar verdiği adım sayısı.
- Yararlı uyuşmazlık (beneficial disagreement): Farklı kararın daha iyi clearance veya ilerleme sağladığı adım sayısı.