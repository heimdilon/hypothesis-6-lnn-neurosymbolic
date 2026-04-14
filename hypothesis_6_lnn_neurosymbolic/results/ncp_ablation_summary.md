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
| cfc | imitation | 2 | 1.8723 | 0.708 |  |  |  |
| cfc | rl_finetune | 3 | -1.9045 |  | 4.031 | 0 | 1 |
| ltc | imitation | 2 | 2.0405 | 0.000 |  |  |  |
| ltc | rl_finetune | 3 | 0.3050 |  | -7.107 | 0 | 1 |
| mlp | imitation | 2 | 1.9886 | 0.104 |  |  |  |
| mlp | rl_finetune | 3 | 0.2415 |  | -3.493 | 0 | 1 |

## Residual vs pure fark tablosu

| cell | stage | scenario_group | pure_success | residual_success | delta_success | pure_collision | residual_collision | delta_collision | delta_min_clearance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc | imitation | default | 0.000 | 0.958 | 0.958 | 1.000 | 0.042 | -0.958 | 0.491 |
| cfc | imitation | hard | 0.000 | 0.400 | 0.400 | 1.000 | 0.600 | -0.400 | 0.141 |
| cfc | rl_finetune | default | 0.000 | 0.958 | 0.958 | 1.000 | 0.042 | -0.958 | 0.472 |
| cfc | rl_finetune | hard | 0.000 | 0.400 | 0.400 | 1.000 | 0.600 | -0.400 | 0.153 |
| ltc | imitation | default | 0.000 | 0.875 | 0.875 | 0.458 | 0.042 | -0.417 | 0.348 |
| ltc | imitation | hard | 0.000 | 0.333 | 0.333 | 0.500 | 0.667 | 0.167 | -0.102 |
| ltc | rl_finetune | default | 0.000 | 0.917 | 0.917 | 0.500 | 0.042 | -0.458 | 0.364 |
| ltc | rl_finetune | hard | 0.000 | 0.367 | 0.367 | 0.500 | 0.633 | 0.133 | -0.051 |
| mlp | imitation | default | 0.000 | 0.917 | 0.917 | 1.000 | 0.042 | -0.958 | 0.580 |
| mlp | imitation | hard | 0.000 | 0.333 | 0.333 | 1.000 | 0.667 | -0.333 | 0.149 |
| mlp | rl_finetune | default | 0.000 | 0.875 | 0.875 | 1.000 | 0.042 | -0.958 | 0.597 |
| mlp | rl_finetune | hard | 0.000 | 0.333 | 0.333 | 1.000 | 0.667 | -0.333 | 0.143 |

## CfC vs LTC fark tablosu

| stage | variant | scenario_group | cfc_success | ltc_success | delta_success_cfc_minus_ltc | cfc_collision | ltc_collision | delta_collision_cfc_minus_ltc | delta_min_clearance_cfc_minus_ltc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imitation | pure | default | 0.000 | 0.000 | 0.000 | 1.000 | 0.458 | 0.542 | -0.166 |
| imitation | pure | hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 | 0.500 | -0.223 |
| imitation | residual | default | 0.958 | 0.875 | 0.083 | 0.042 | 0.042 | 0.000 | -0.023 |
| imitation | residual | hard | 0.400 | 0.333 | 0.067 | 0.600 | 0.667 | -0.067 | 0.020 |
| rl_finetune | pure | default | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 | 0.500 | -0.144 |
| rl_finetune | pure | hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 | 0.500 | -0.207 |
| rl_finetune | residual | default | 0.958 | 0.917 | 0.042 | 0.042 | 0.042 | 0.000 | -0.035 |
| rl_finetune | residual | hard | 0.400 | 0.367 | 0.033 | 0.600 | 0.633 | -0.033 | -0.002 |

## Hard map ortalaması (95% CI dahil)

| controller | cell | stage | variant | n | success_rate | success_ci_lower | success_ci_upper | collision_rate | mean_steps | mean_min_clearance | mean_near_misses | mean_action_disagreements | mean_beneficial_disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc_imitation_pure | cfc | imitation | pure | 30 | 0.000 | 0.000 | 0.114 | 1.000 | 10.0 | -0.115 | 3.1 | 7.6 | 5.2 |
| cfc_imitation_residual | cfc | imitation | residual | 30 | 0.400 | 0.246 | 0.577 | 0.600 | 37.0 | 0.026 | 7.9 | 7.1 | 6.1 |
| cfc_random_residual | cfc | random | residual | 30 | 0.367 | 0.219 | 0.545 | 0.633 | 35.7 | 0.072 | 7.1 | 7.8 | 7.2 |
| cfc_rl_finetune_pure | cfc | rl_finetune | pure | 30 | 0.000 | 0.000 | 0.114 | 1.000 | 9.8 | -0.122 | 3.0 | 7.3 | 5.0 |
| cfc_rl_finetune_residual | cfc | rl_finetune | residual | 30 | 0.400 | 0.246 | 0.577 | 0.600 | 36.7 | 0.031 | 7.6 | 6.9 | 6.2 |
| fixed_policy | fixed | fixed | baseline | 30 | 0.367 | 0.219 | 0.545 | 0.633 | 38.3 | 0.079 | 6.6 | 0.0 | 0.0 |
| ltc_imitation_pure | ltc | imitation | pure | 30 | 0.000 | 0.000 | 0.114 | 0.500 | 62.4 | 0.107 | 14.0 | 44.3 | 15.8 |
| ltc_imitation_residual | ltc | imitation | residual | 30 | 0.333 | 0.192 | 0.512 | 0.667 | 36.4 | 0.006 | 7.3 | 5.8 | 5.0 |
| ltc_random_residual | ltc | random | residual | 30 | 0.333 | 0.192 | 0.512 | 0.667 | 38.9 | 0.075 | 6.8 | 5.6 | 4.6 |
| ltc_rl_finetune_pure | ltc | rl_finetune | pure | 30 | 0.000 | 0.000 | 0.114 | 0.500 | 62.8 | 0.085 | 14.2 | 44.7 | 15.8 |
| ltc_rl_finetune_residual | ltc | rl_finetune | residual | 30 | 0.367 | 0.219 | 0.545 | 0.633 | 35.3 | 0.034 | 6.9 | 5.6 | 4.8 |
| mlp_imitation_pure | mlp | imitation | pure | 30 | 0.000 | 0.000 | 0.114 | 1.000 | 8.1 | -0.124 | 3.0 | 7.3 | 4.6 |
| mlp_imitation_residual | mlp | imitation | residual | 30 | 0.333 | 0.192 | 0.512 | 0.667 | 33.7 | 0.025 | 6.7 | 6.0 | 5.0 |
| mlp_rl_finetune_pure | mlp | rl_finetune | pure | 30 | 0.000 | 0.000 | 0.114 | 1.000 | 10.5 | -0.124 | 3.5 | 9.6 | 5.4 |
| mlp_rl_finetune_residual | mlp | rl_finetune | residual | 30 | 0.333 | 0.192 | 0.512 | 0.667 | 35.4 | 0.018 | 7.3 | 6.1 | 5.2 |

## Default map ortalaması (95% CI dahil)

| controller | cell | stage | variant | n | success_rate | success_ci_lower | success_ci_upper | collision_rate | mean_steps | mean_min_clearance | mean_near_misses | mean_action_disagreements | mean_beneficial_disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc_imitation_pure | cfc | imitation | pure | 24 | 0.000 | 0.000 | 0.138 | 1.000 | 11.2 | -0.062 | 2.4 | 8.6 | 4.8 |
| cfc_imitation_residual | cfc | imitation | residual | 24 | 0.958 | 0.798 | 0.993 | 0.042 | 50.0 | 0.429 | 1.5 | 13.3 | 11.0 |
| cfc_random_residual | cfc | random | residual | 24 | 0.958 | 0.798 | 0.993 | 0.042 | 46.5 | 0.438 | 0.8 | 14.7 | 12.3 |
| cfc_rl_finetune_pure | cfc | rl_finetune | pure | 24 | 0.000 | 0.000 | 0.138 | 1.000 | 11.2 | -0.062 | 2.4 | 8.6 | 4.8 |
| cfc_rl_finetune_residual | cfc | rl_finetune | residual | 24 | 0.958 | 0.798 | 0.993 | 0.042 | 50.8 | 0.410 | 1.7 | 13.0 | 10.6 |
| fixed_policy | fixed | fixed | baseline | 24 | 0.833 | 0.641 | 0.933 | 0.000 | 59.0 | 0.406 | 1.4 | 0.0 | 0.0 |
| ltc_imitation_pure | ltc | imitation | pure | 24 | 0.000 | 0.000 | 0.138 | 0.458 | 67.2 | 0.104 | 18.2 | 48.1 | 12.3 |
| ltc_imitation_residual | ltc | imitation | residual | 24 | 0.875 | 0.690 | 0.957 | 0.042 | 56.7 | 0.452 | 1.0 | 10.5 | 8.2 |
| ltc_random_residual | ltc | random | residual | 24 | 0.917 | 0.742 | 0.977 | 0.000 | 55.1 | 0.429 | 1.5 | 11.0 | 8.8 |
| ltc_rl_finetune_pure | ltc | rl_finetune | pure | 24 | 0.000 | 0.000 | 0.138 | 0.500 | 63.1 | 0.081 | 17.0 | 44.0 | 10.9 |
| ltc_rl_finetune_residual | ltc | rl_finetune | residual | 24 | 0.917 | 0.742 | 0.977 | 0.042 | 54.2 | 0.446 | 1.2 | 9.9 | 8.0 |
| mlp_imitation_pure | mlp | imitation | pure | 24 | 0.000 | 0.000 | 0.138 | 1.000 | 8.8 | -0.135 | 2.2 | 7.4 | 3.8 |
| mlp_imitation_residual | mlp | imitation | residual | 24 | 0.917 | 0.742 | 0.977 | 0.042 | 58.8 | 0.445 | 1.2 | 9.9 | 8.3 |
| mlp_rl_finetune_pure | mlp | rl_finetune | pure | 24 | 0.000 | 0.000 | 0.138 | 1.000 | 8.8 | -0.145 | 2.2 | 7.5 | 3.8 |
| mlp_rl_finetune_residual | mlp | rl_finetune | residual | 24 | 0.875 | 0.690 | 0.957 | 0.042 | 55.8 | 0.452 | 1.1 | 8.4 | 6.8 |

## İstatistiksel karşılaştırmalar (Mann-Whitney U, BH düzeltmeli)

| scenario_group | comparison | controller_a | controller_b | n_a | n_b | small_n_warning | mean_a | mean_b | delta | mann_whitney_U | p_value_raw | p_value_bh_corrected | cohens_d | significant_005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | residual_vs_pure | cfc_imitation_residual | cfc_imitation_pure | 24 | 24 | no | 0.958 | 0.000 | 0.958 | 564.0 | 0.0000 | 0.0000 | 6.640 | yes |
| default | ncp_vs_fixed | cfc_imitation_residual | fixed_policy | 24 | 24 | no | 0.958 | 0.833 | 0.125 | 324.0 | 0.1607 | 0.3616 | 0.409 | no |
| default | residual_vs_pure | cfc_rl_finetune_residual | cfc_rl_finetune_pure | 24 | 24 | no | 0.958 | 0.000 | 0.958 | 564.0 | 0.0000 | 0.0000 | 6.640 | yes |
| default | ncp_vs_fixed | cfc_rl_finetune_residual | fixed_policy | 24 | 24 | no | 0.958 | 0.833 | 0.125 | 324.0 | 0.1607 | 0.3616 | 0.409 | no |
| default | trained_vs_random | cfc_rl_finetune_residual | cfc_random_residual | 24 | 24 | no | 0.958 | 0.958 | 0.000 | 288.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | residual_vs_pure | ltc_imitation_residual | ltc_imitation_pure | 24 | 24 | no | 0.875 | 0.000 | 0.875 | 540.0 | 0.0000 | 0.0000 | 3.663 | yes |
| default | ncp_vs_fixed | ltc_imitation_residual | fixed_policy | 24 | 24 | no | 0.875 | 0.833 | 0.042 | 300.0 | 0.6857 | 0.8816 | 0.116 | no |
| default | residual_vs_pure | ltc_rl_finetune_residual | ltc_rl_finetune_pure | 24 | 24 | no | 0.917 | 0.000 | 0.917 | 552.0 | 0.0000 | 0.0000 | 4.592 | yes |
| default | ncp_vs_fixed | ltc_rl_finetune_residual | fixed_policy | 24 | 24 | no | 0.917 | 0.833 | 0.083 | 312.0 | 0.3877 | 0.6345 | 0.249 | no |
| default | trained_vs_random | ltc_rl_finetune_residual | ltc_random_residual | 24 | 24 | no | 0.917 | 0.917 | 0.000 | 288.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | residual_vs_pure | mlp_imitation_residual | mlp_imitation_pure | 24 | 24 | no | 0.917 | 0.000 | 0.917 | 552.0 | 0.0000 | 0.0000 | 4.592 | yes |
| default | ncp_vs_fixed | mlp_imitation_residual | fixed_policy | 24 | 24 | no | 0.917 | 0.833 | 0.083 | 312.0 | 0.3877 | 0.6345 | 0.249 | no |
| default | residual_vs_pure | mlp_rl_finetune_residual | mlp_rl_finetune_pure | 24 | 24 | no | 0.875 | 0.000 | 0.875 | 540.0 | 0.0000 | 0.0000 | 3.663 | yes |
| default | ncp_vs_fixed | mlp_rl_finetune_residual | fixed_policy | 24 | 24 | no | 0.875 | 0.833 | 0.042 | 300.0 | 0.6857 | 0.8816 | 0.116 | no |
| default | ncp_vs_mlp | cfc_imitation_pure | mlp_imitation_pure | 24 | 24 | no | 0.000 | 0.000 | 0.000 | 288.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | cfc_imitation_residual | mlp_imitation_residual | 24 | 24 | no | 0.958 | 0.917 | 0.042 | 300.0 | 0.5552 | 0.8327 | 0.169 | no |
| default | ncp_vs_mlp | cfc_rl_finetune_pure | mlp_rl_finetune_pure | 24 | 24 | no | 0.000 | 0.000 | 0.000 | 288.0 | 1.0000 | 1.0000 | 0.000 | no |
| default | ncp_vs_mlp | cfc_rl_finetune_residual | mlp_rl_finetune_residual | 24 | 24 | no | 0.958 | 0.875 | 0.083 | 312.0 | 0.3014 | 0.6027 | 0.299 | no |
| hard | residual_vs_pure | cfc_imitation_residual | cfc_imitation_pure | 30 | 30 | no | 0.400 | 0.000 | 0.400 | 630.0 | 0.0001 | 0.0011 | 1.135 | yes |
| hard | ncp_vs_fixed | cfc_imitation_residual | fixed_policy | 30 | 30 | no | 0.400 | 0.367 | 0.033 | 465.0 | 0.7923 | 0.9508 | 0.067 | no |
| hard | residual_vs_pure | cfc_rl_finetune_residual | cfc_rl_finetune_pure | 30 | 30 | no | 0.400 | 0.000 | 0.400 | 630.0 | 0.0001 | 0.0011 | 1.135 | yes |
| hard | ncp_vs_fixed | cfc_rl_finetune_residual | fixed_policy | 30 | 30 | no | 0.400 | 0.367 | 0.033 | 465.0 | 0.7923 | 0.9508 | 0.067 | no |
| hard | trained_vs_random | cfc_rl_finetune_residual | cfc_random_residual | 30 | 30 | no | 0.400 | 0.367 | 0.033 | 465.0 | 0.7923 | 0.9508 | 0.067 | no |
| hard | residual_vs_pure | ltc_imitation_residual | ltc_imitation_pure | 30 | 30 | no | 0.333 | 0.000 | 0.333 | 600.0 | 0.0006 | 0.0018 | 0.983 | yes |
| hard | ncp_vs_fixed | ltc_imitation_residual | fixed_policy | 30 | 30 | no | 0.333 | 0.367 | -0.033 | 435.0 | 0.7884 | 0.9508 | -0.069 | no |
| hard | residual_vs_pure | ltc_rl_finetune_residual | ltc_rl_finetune_pure | 30 | 30 | no | 0.367 | 0.000 | 0.367 | 615.0 | 0.0003 | 0.0016 | 1.058 | yes |
| hard | ncp_vs_fixed | ltc_rl_finetune_residual | fixed_policy | 30 | 30 | no | 0.367 | 0.367 | 0.000 | 450.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | trained_vs_random | ltc_rl_finetune_residual | ltc_random_residual | 30 | 30 | no | 0.367 | 0.333 | 0.033 | 465.0 | 0.7884 | 0.9508 | 0.069 | no |
| hard | residual_vs_pure | mlp_imitation_residual | mlp_imitation_pure | 30 | 30 | no | 0.333 | 0.000 | 0.333 | 600.0 | 0.0006 | 0.0018 | 0.983 | yes |
| hard | ncp_vs_fixed | mlp_imitation_residual | fixed_policy | 30 | 30 | no | 0.333 | 0.367 | -0.033 | 435.0 | 0.7884 | 0.9508 | -0.069 | no |
| hard | residual_vs_pure | mlp_rl_finetune_residual | mlp_rl_finetune_pure | 30 | 30 | no | 0.333 | 0.000 | 0.333 | 600.0 | 0.0006 | 0.0018 | 0.983 | yes |
| hard | ncp_vs_fixed | mlp_rl_finetune_residual | fixed_policy | 30 | 30 | no | 0.333 | 0.367 | -0.033 | 435.0 | 0.7884 | 0.9508 | -0.069 | no |
| hard | ncp_vs_mlp | cfc_imitation_pure | mlp_imitation_pure | 30 | 30 | no | 0.000 | 0.000 | 0.000 | 450.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | cfc_imitation_residual | mlp_imitation_residual | 30 | 30 | no | 0.400 | 0.333 | 0.067 | 480.0 | 0.5952 | 0.9508 | 0.136 | no |
| hard | ncp_vs_mlp | cfc_rl_finetune_pure | mlp_rl_finetune_pure | 30 | 30 | no | 0.000 | 0.000 | 0.000 | 450.0 | 1.0000 | 1.0000 | 0.000 | no |
| hard | ncp_vs_mlp | cfc_rl_finetune_residual | mlp_rl_finetune_residual | 30 | 30 | no | 0.400 | 0.333 | 0.067 | 480.0 | 0.5952 | 0.9508 | 0.136 | no |

## Bilimsel okuma

- `pure` varyantında karar tamamen NCP/MLP logits ile verilir; bu saf kapasite ve kararlılık testidir.
- `residual` varyantında sabit uzman politikanın üstüne NCP/MLP düzeltmesi eklenir; bu daha güvenli dağıtım senaryosunu temsil eder.
- `random_residual` eğitilmemiş (rastgele ağırlıklı) NCP ile residual yapıyı test eder; öğrenilmiş bilginin etkisini izole eder.
- `mlp` baseline, recurrent olmayan feedforward ağdır; NCP mimarisinin (sürekli zaman dinamikleri) etkisini ayırt etmeyi sağlar.
- CfC/LTC kıyası aynı veri, seed ailesi ve harita setinde yapılır; farklar mimari ve kısa fine-tune dinamiğinden gelir.
- 2 bağımsız seed ile eğitim yapılmış, sonuçlar tüm seed'ler üzerinden toplanmıştır.
- Wilson score interval (95% CI) küçük örneklem için uygun binomial güven aralığı sağlar.
- Benjamini-Hochberg FDR düzeltmesi çoklu karşılaştırma hatasını kontrol eder.
- `small_n_warning=yes`: Grup başına n<8 olduğunda Mann-Whitney U normal yaklaşımı güvenilirliğini yitirir; bu satırlardaki p-değerleri dikkatli yorumlanmalıdır.
- Aksiyon uyuşmazlığı (action disagreement): NCP/MLP'nin sabit politikadan farklı karar verdiği adım sayısı.
- Yararlı uyuşmazlık (beneficial disagreement): Farklı kararın daha iyi clearance veya ilerleme sağladığı adım sayısı.