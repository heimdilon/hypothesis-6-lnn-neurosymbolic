# H6 ilk deney sonuçları

Bu deney gerçek DRL eğitimi değil, H6 için hızlı bir simülasyon prototipidir.
Sabit politika, LNN-benzeri çevrimiçi sıvı durumlu politika ve LNN + sembolik güvenlik süpervizörü karşılaştırıldı.

## Değişmiş dağılım senaryoları ortalaması

| Denetleyici | Başarı | Çarpışma | Adaptasyon gecikmesi | Yakın geçiş |
|---|---:|---:|---:|---:|
| fixed_policy | 0.967 | 0.017 | 10.0 | 0.4 |
| liquid_online | 0.950 | 0.050 | 10.0 | 2.1 |
| liquid_supervisor | 1.000 | 0.000 | 8.1 | 0.0 |

## Yorum

İlk sonuç H6'nın güvenlik tarafını destekliyor: `liquid_online` sabit politikaya yakın başarı üretti, fakat daha fazla yakın geçiş ve daha yüksek çarpışma oranı taşıdı. `liquid_supervisor` ise değişmiş dağılım senaryolarında başarıyı 1.000'e, çarpışmayı 0.000'a getirdi ve adaptasyon gecikmesini biraz düşürdü.

Sabit politika bu basit simülasyonda beklenenden güçlü kaldı. Bu nedenle sonuç, "LNN tek başına sabit politikadan üstündür" iddiasını henüz kanıtlamıyor; daha doğru okuma, LNN-benzeri çevrimiçi adaptasyonun güvenlik süpervizörü olmadan riskli olduğu ve sembolik kural katmanının bu riski belirgin biçimde sınırladığıdır.

Bir sonraki adım, bu prototipi gerçek LNN/LTC hücresi eğitimi, daha zengin sensör modeli ve ablation çalışmalarıyla genişletmektir.
