# H6 ilk deney sonuçları

Liquid politika backend'i: `cfc`.
Bu deney hâlâ küçük ölçekli bir navigasyon benzetimidir; liquid denetleyici artık resmi MIT `ncps.torch` CfC/LTC katmanlarıyla çalıştırılabilir.
Sabit politika, resmi NCP tabanlı çevrimiçi sıvı politika ve liquid + sembolik güvenlik süpervizörü karşılaştırıldı.

## Değişmiş dağılım senaryoları ortalaması

| Denetleyici | Başarı | Çarpışma | Adaptasyon gecikmesi | Yakın geçiş |
|---|---:|---:|---:|---:|
| fixed_policy | 1.000 | 0.000 | 8.0 | 0.0 |
| liquid_online | 1.000 | 0.000 | 8.0 | 0.0 |
| liquid_supervisor | 1.000 | 0.000 | 8.0 | 0.0 |

## Yorum

H6'nın beklediği ana desen şudur: sıvı durumlu çevrimiçi politika dağılım kaymasında sabit politikadan daha hızlı uyum sağlayabilir, ancak güvenlik kısıtı eklenmezse yakın geçiş ve çarpışma riski artabilir. Sembolik süpervizör, seçilen aksiyonu kısa ufuklu çarpışma/clearance kuralıyla maskeleyerek bu riski sınırlamayı hedefler.

Bir sonraki adım, bu prototipi gerçek LNN/LTC hücresi eğitimi, daha zengin sensör modeli ve ablation çalışmalarıyla genişletmektir.