# Saf NCP imitation/RL ve ablation sonuçları

Bu dosya resmi `ncps.torch` CfC/LTC katmanları ile üretilen offline imitation, kısa policy-gradient fine-tune ve pure/residual ablation sonuçlarını özetler.

## Deney konfigürasyonu

- Eğitim senaryoları: train_like, shifted_clutter, narrow_gate, u_trap
- Test senaryoları: train_like, shifted_clutter, narrow_gate, u_trap, zigzag_corridor, dense_maze, deceptive_u_trap, sensor_shadow, labyrinth_maze
- Imitation sequence sayısı: 48, doğrulama: 12, sequence length: 24
- Imitation epoch: 5, RL fine-tune episode: 6
- NCP hidden: 24, sparsity: 0.5, residual scale: 0.35

## Eğitim özeti

| cell | phase | epoch_or_episode | loss | val_accuracy | episode_return | success | collision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cfc | imitation | 5 | 1.7778 | 0.205 |  |  |  |
| cfc | rl_finetune | 6 | -0.8532 |  | 0.490 | 0 | 1 |
| ltc | imitation | 5 | 1.7810 | 0.562 |  |  |  |
| ltc | rl_finetune | 6 | 0.1809 |  | -9.360 | 0 | 1 |

## Residual vs pure fark tablosu

| cell | stage | scenario_group | pure_success | residual_success | delta_success | pure_collision | residual_collision | delta_collision | delta_min_clearance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc | imitation | default | 0.000 | 0.750 | 0.750 | 1.000 | 0.000 | -1.000 | 0.547 |
| cfc | imitation | hard | 0.000 | 0.300 | 0.300 | 1.000 | 0.700 | -0.300 | 0.156 |
| cfc | rl_finetune | default | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | -1.000 | 0.557 |
| cfc | rl_finetune | hard | 0.000 | 0.600 | 0.600 | 1.000 | 0.400 | -0.600 | 0.201 |
| ltc | imitation | default | 0.000 | 0.875 | 0.875 | 1.000 | 0.000 | -1.000 | 0.575 |
| ltc | imitation | hard | 0.000 | 0.300 | 0.300 | 1.000 | 0.700 | -0.300 | 0.143 |
| ltc | rl_finetune | default | 0.000 | 0.875 | 0.875 | 1.000 | 0.000 | -1.000 | 0.552 |
| ltc | rl_finetune | hard | 0.000 | 0.300 | 0.300 | 1.000 | 0.700 | -0.300 | 0.137 |

## CfC vs LTC fark tablosu

| stage | variant | scenario_group | cfc_success | ltc_success | delta_success_cfc_minus_ltc | cfc_collision | ltc_collision | delta_collision_cfc_minus_ltc | delta_min_clearance_cfc_minus_ltc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imitation | pure | default | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | -0.001 |
| imitation | pure | hard | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | -0.031 |
| imitation | residual | default | 0.750 | 0.875 | -0.125 | 0.000 | 0.000 | 0.000 | -0.029 |
| imitation | residual | hard | 0.300 | 0.300 | 0.000 | 0.700 | 0.700 | 0.000 | -0.018 |
| rl_finetune | pure | default | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | -0.025 |
| rl_finetune | pure | hard | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | -0.044 |
| rl_finetune | residual | default | 1.000 | 0.875 | 0.125 | 0.000 | 0.000 | 0.000 | -0.020 |
| rl_finetune | residual | hard | 0.600 | 0.300 | 0.300 | 0.400 | 0.700 | -0.300 | 0.020 |

## Hard map ortalaması

| controller | cell | stage | variant | n | success_rate | collision_rate | mean_steps | mean_min_clearance | mean_near_misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc_imitation_pure | cfc | imitation | pure | 10 | 0.000 | 1.000 | 6.1 | -0.141 | 2.0 |
| cfc_imitation_residual | cfc | imitation | residual | 10 | 0.300 | 0.700 | 40.0 | 0.015 | 8.7 |
| cfc_rl_finetune_pure | cfc | rl_finetune | pure | 10 | 0.000 | 1.000 | 6.2 | -0.157 | 2.1 |
| cfc_rl_finetune_residual | cfc | rl_finetune | residual | 10 | 0.600 | 0.400 | 45.0 | 0.043 | 8.8 |
| fixed_policy | fixed | fixed | baseline | 10 | 0.500 | 0.500 | 48.2 | 0.081 | 7.8 |
| ltc_imitation_pure | ltc | imitation | pure | 10 | 0.000 | 1.000 | 9.7 | -0.110 | 3.2 |
| ltc_imitation_residual | ltc | imitation | residual | 10 | 0.300 | 0.700 | 47.3 | 0.033 | 7.0 |
| ltc_rl_finetune_pure | ltc | rl_finetune | pure | 10 | 0.000 | 1.000 | 9.9 | -0.113 | 3.2 |
| ltc_rl_finetune_residual | ltc | rl_finetune | residual | 10 | 0.300 | 0.700 | 44.6 | 0.023 | 6.0 |

## Default map ortalaması

| controller | cell | stage | variant | n | success_rate | collision_rate | mean_steps | mean_min_clearance | mean_near_misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfc_imitation_pure | cfc | imitation | pure | 8 | 0.000 | 1.000 | 7.4 | -0.128 | 2.2 |
| cfc_imitation_residual | cfc | imitation | residual | 8 | 0.750 | 0.000 | 63.9 | 0.419 | 0.8 |
| cfc_rl_finetune_pure | cfc | rl_finetune | pure | 8 | 0.000 | 1.000 | 7.6 | -0.151 | 2.2 |
| cfc_rl_finetune_residual | cfc | rl_finetune | residual | 8 | 1.000 | 0.000 | 47.8 | 0.406 | 1.1 |
| fixed_policy | fixed | fixed | baseline | 8 | 0.750 | 0.000 | 73.0 | 0.475 | 0.0 |
| ltc_imitation_pure | ltc | imitation | pure | 8 | 0.000 | 1.000 | 15.0 | -0.126 | 2.5 |
| ltc_imitation_residual | ltc | imitation | residual | 8 | 0.875 | 0.000 | 60.0 | 0.448 | 0.6 |
| ltc_rl_finetune_pure | ltc | rl_finetune | pure | 8 | 0.000 | 1.000 | 14.0 | -0.127 | 2.2 |
| ltc_rl_finetune_residual | ltc | rl_finetune | residual | 8 | 0.875 | 0.000 | 67.4 | 0.426 | 1.1 |

## Bilimsel okuma

- `pure` varyantında karar tamamen NCP logits ile verilir; bu saf kapasite ve kararlılık testidir.
- `residual` varyantında sabit uzman politikanın üstüne NCP düzeltmesi eklenir; bu daha güvenli dağıtım senaryosunu temsil eder.
- CfC/LTC kıyası aynı veri, seed ailesi ve harita setinde yapılır; farklar mimari ve kısa fine-tune dinamiğinden gelir.
- Bu çalışma hâlâ küçük ölçekli simülasyondur; sonuçlar hipotez taraması için kullanılır, robotik sistem iddiası için daha büyük seed sayısı ve fiziksel validasyon gerekir.