# H6 ilk deney sonuçları

Bu deney gerçek DRL eğitimi değil, H6 için hızlı bir simülasyon prototipidir.
Sabit politika, LNN-benzeri çevrimiçi sıvı durumlu politika ve LNN + sembolik güvenlik süpervizörü karşılaştırıldı.

## Değişmiş dağılım senaryoları ortalaması

| Denetleyici | Başarı | Çarpışma | Adaptasyon gecikmesi | Yakın geçiş |
|---|---:|---:|---:|---:|
| fixed_policy | 0.250 | 0.750 | 8.0 | 6.0 |
| liquid_online | 0.281 | 0.719 | 11.5 | 6.7 |
| liquid_supervisor | 1.000 | 0.000 | 8.0 | 3.6 |

## Yorum

H6'nın beklediği ana desen şudur: sıvı durumlu çevrimiçi politika dağılım kaymasında sabit politikadan daha hızlı uyum sağlayabilir, ancak güvenlik kısıtı eklenmezse yakın geçiş ve çarpışma riski artabilir. Sembolik süpervizör, seçilen aksiyonu kısa ufuklu çarpışma/clearance kuralıyla maskeleyerek bu riski sınırlamayı hedefler.

Bir sonraki adım, bu prototipi gerçek LNN/LTC hücresi eğitimi, daha zengin sensör modeli ve ablation çalışmalarıyla genişletmektir.