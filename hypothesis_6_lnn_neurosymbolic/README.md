# Hipotez 6: Liquid Neural Network ile Güvenli Mobil Robot Navigasyonu

Bu proje, yüksek lisans dersi kapsamında seçilen mobil robot navigasyonu literatüründen çıkarılan bir araştırma hipotezini küçük, tekrarlanabilir ve görselleştirilebilir bir simülasyon ortamında incelemek için hazırlandı.

Çalışmanın merkezindeki soru şudur: **Liquid Neural Network / Neural Circuit Policy tabanlı sürekli öğrenen bir kontrolcü, değişen engel konfigürasyonlarında sabit bir navigasyon politikasına göre daha iyi adaptasyon sağlayabilir mi; yoksa güvenlik için mutlaka sembolik veya kural tabanlı bir denetleyiciye mi ihtiyaç duyar?**

> Not: Bu repoda "açık" kelimesi güvenlik açığı anlamında değil, literatürdeki **araştırma boşluğu** anlamında kullanılmıştır.

## Kısa Özet

- Literatür taramasında iki 2024 derleme makalesi incelendi.
- Bu makalelerden mobil robot navigasyonunda öğrenen sistemler, nöro-sembolik güvenlik ve Liquid Neural Network potansiyeliyle ilgili araştırma boşlukları çıkarıldı.
- Seçilen araştırma boşluğu: **Nöro-sembolik öz-öğrenmede temsil uyumsuzluğu ve Liquid Neural Network potansiyeli**.
- Seçilen hipotez: **Hipotez 6**.
- Python ile 2D mobil robot navigasyon simülasyonu geliştirildi.
- Zorlu haritalar, labirentler, GIF/PNG çıktıları ve etkileşimli harita çizim arayüzü eklendi.
- MIT'nin resmi `ncps` kütüphanesi kullanılarak `CfC` ve `LTC` tabanlı Neural Circuit Policy modelleri denendi.
- Saf NCP, residual NCP, CfC-LTC karşılaştırması ve kısa imitation/RL ablation deneyleri yapıldı.

Ana bulgu: **Saf NCP kontrolcü bu küçük deney düzeninde kararsız kaldı; ancak sabit güvenli politika üzerine residual NCP düzeltmesi eklemek, özellikle CfC + RL fine-tune konfigürasyonunda zorlu haritalarda başarıyı artırdı ve çarpışmayı azalttı.** Bu sonuç, hipotezin "sürekli öğrenme potansiyeli vardır ama güvenlik süpervizörüyle sınırlandırılmalıdır" kısmını destekleyen ön kanıt sağlar.

## Seçilen Makaleler

Çalışmanın literatür temeli şu iki derleme makalesine dayanır:

1. S. Al Mahmud, A. Kamarulariffin, A. M. Ibrahim, and A. J. H. Mohideen, "Advancements and Challenges in Mobile Robot Navigation: A Comprehensive Review of Algorithms and Potential for Self-Learning Approaches," *Journal of Intelligent & Robotic Systems*, vol. 110, article 120, 2024. DOI: [10.1007/s10846-024-02149-5](https://doi.org/10.1007/s10846-024-02149-5). Doğrudan PDF: [Springer PDF](https://link.springer.com/content/pdf/10.1007/s10846-024-02149-5.pdf?utm_source=clarivate&getft_integrator=clarivate)
2. K. Katona, H. A. Neamah, and P. Korondi, "Obstacle Avoidance and Path Planning Methods for Autonomous Navigation of Mobile Robot," *Sensors*, vol. 24, article 3573, 2024. DOI: [10.3390/s24113573](https://doi.org/10.3390/s24113573). Makale sayfası: [MDPI Sensors](https://www.mdpi.com/1424-8220/24/11/3573)

İlk makale öz-öğrenme, derin pekiştirmeli öğrenme, nöro-sembolik yaklaşımlar ve Liquid Neural Network potansiyeline odaklanır. İkinci makale ise klasik, sezgisel, hibrit ve öğrenme tabanlı engel kaçınma/yol planlama yöntemlerini sınıflandırır. Bu iki makale birlikte okunduğunda, öğrenen robot politikalarının adaptasyon potansiyeli ile güvenlik/kararlılık garantisi eksikliği arasında önemli bir gerilim olduğu görülür.

## Seçilen Araştırma Boşluğu

Seçilen boşluk şudur:

> **Nöro-sembolik öz-öğrenmede temsil uyumsuzluğu ve Liquid Neural Network potansiyeli**

Bu boşlukta temel problem, öğrenen sinir ağı politikasının dinamik ortamlara uyum sağlayabilmesi fakat sembolik güvenlik kuralları, açıklanabilirlik ve kararlı kontrol ile nasıl birleştirileceğinin net olmamasıdır. Liquid Neural Network ve Neural Circuit Policy mimarileri zamana bağlı, kompakt ve sürekli dinamiklere sahip oldukları için bu alanda ilginç bir adaydır. Ancak vanishing gradient, parametre hassasiyeti ve dağılım dışı ortamda kararsız davranış riski devam eder.

## Hipotez 6

> **H6:** LNN tabanlı sürekli öğrenen bir navigasyon politikası, sabit ağırlıklı DRL politikasına göre dağıtım sonrası ortam değişikliklerine, örneğin yeni engel konfigürasyonlarına ve değişen algı koşullarına, daha hızlı adaptasyon gösterir; ancak vanishing gradient ve parametre hassasiyeti nedeniyle kararlılık riski taşır ve bu risk güvenlik süpervizörüyle sınırlandırılmalıdır.

Bu proje H6'yı tam ölçekli bir robotik sistem olarak kanıtlamaz. Ama hipotezi sınamak için küçük ölçekli bir deney zemini kurar: farklı haritalar, sabit politika, saf NCP, residual NCP, CfC-LTC karşılaştırması ve görsel simülasyon çıktıları.

## Kısaltmalar ve Temel Kavramlar

| Terim | Açılım | Açıklama |
| --- | --- | --- |
| LNN | Liquid Neural Network | Girdiye ve zamana bağlı dinamikleri olan sinir ağı ailesidir. Bu çalışmada LNN fikri, değişen haritalara uyum sağlayabilecek öğrenen kontrolcü adayı olarak ele alındı. |
| NCP | Neural Circuit Policy | MIT'nin `ncps` kütüphanesinde yer alan, biyolojik sinir devrelerinden esinlenen seyrek bağlantılı politika mimarisidir. Robotun hangi yöne hareket edeceğine karar veren öğrenen kontrolcü olarak kullanıldı. |
| CfC | Closed-form Continuous-time | Sürekli zamanlı nöral dinamikleri kapalı form yaklaşımla hesaplayan NCP katmanıdır. Bu çalışmada LTC'ye göre daha kararlı sonuç verdi. |
| LTC | Liquid Time-Constant | Öğrenilebilir zaman sabitleri kullanan liquid neural network katmanıdır. Dinamik sistem gibi davranması beklenir, fakat parametre hassasiyeti daha belirgin olabilir. |
| RL | Reinforcement Learning | Pekiştirmeli öğrenmedir. Robot doğru ilerleme ve hedefe ulaşma için ödül, çarpışma ve riskli davranış için ceza alır. |
| DRL | Deep Reinforcement Learning | Pekiştirmeli öğrenmenin derin sinir ağlarıyla yapılan halidir. Hipotezde sabit ağırlıklı DRL politikası, dağıtım sonrası adaptasyonu sınırlı bir referans fikir olarak kullanıldı. |
| Imitation learning | Gösterimden öğrenme | Modelin önce uzman/planner davranışını taklit ederek başlangıç politikası öğrenmesidir. Bu, RL öncesi daha kontrollü bir başlangıç sağlar. |
| Fine-tune | İnce ayar | Önceden eğitilmiş politikanın kısa ek eğitimle belirli göreve uyarlanmasıdır. Bu projede imitation sonrası kısa RL fine-tune kullanıldı. |
| Baseline | Referans politika | Karşılaştırma için kullanılan sabit, geometrik ve elle yazılmış navigasyon politikasıdır. |
| Pure NCP | Saf NCP | Robotun kararını tamamen NCP çıktısına bırakan deney varyantıdır. Güvenlik açısından en riskli ama model kapasitesini en doğrudan gösteren testtir. |
| Residual NCP | Artık/düzeltici NCP | Sabit politikanın karar skorlarına NCP'nin küçük bir düzeltme sinyali eklediği varyanttır. Gerçek robotik için daha güvenli ve daha savunulabilir bir kurulumdur. |
| Safety supervisor | Güvenlik süpervizörü | Öğrenen politikanın riskli aksiyonlarını sınırlayan güvenlik katmanıdır. Bu prototipte kısa ufuklu çarpışma ve açıklık kontrolleriyle temsil edildi. |
| OOD | Out-of-distribution | Eğitimde görülmeyen veya alışılmıştan farklı harita/engel koşullarıdır. Zorlu haritalar bu fikri basitçe test etmek için kullanıldı. |
| Sim-to-real | Simülasyondan gerçeğe geçiş | Simülasyonda çalışan yöntemin gerçek robota aktarılması problemidir. Bu proje henüz simülasyon düzeyindedir. |

## Ablation Nedir?

Ablation, bir modelin veya sistemin hangi parçasının sonuca ne kadar katkı verdiğini anlamak için yapılan kontrollü çıkarma/değiştirme deneyidir. Yani sistem tek bir bütün olarak değil, farklı bileşenleri değiştirilerek incelenir.

Bu projede ablation üç soruya cevap vermek için kullanıldı:

- **Pure vs residual:** NCP tek başına mı daha iyi çalışıyor, yoksa sabit güvenli politika üzerine düzeltici olarak mı daha kararlı?
- **CfC vs LTC:** MIT `ncps` içindeki iki liquid/NCP katmanı aynı görevde farklı davranıyor mu?
- **Imitation vs RL fine-tune:** Uzman davranışını taklit etmek yeterli mi, yoksa kısa pekiştirmeli öğrenme sonrası performans değişiyor mu?

Bu yüzden ablation burada yalnızca teknik bir tablo değildir; doğrudan Hipotez 6'nın güvenlik kısmını test eden deney tasarımıdır. Saf NCP'nin başarısız olması ve residual NCP'nin daha iyi davranması, öğrenen politikanın güvenlik süpervizörüyle sınırlandırılması gerektiği fikrini güçlendirmiştir.

## Çalışma Akışı

```mermaid
flowchart LR
    A["OCR + literatür okuma"] --> B["Araştırma boşlukları"]
    B --> C["Hipotez 6 seçimi"]
    C --> D["2D robot navigasyon simülasyonu"]
    D --> E["Zorlu haritalar ve labirentler"]
    E --> F["MIT ncps: CfC ve LTC"]
    F --> G["Pure vs residual ablation"]
    G --> H["CfC vs LTC karşılaştırması"]
    H --> I["Bilimsel rapor ve README"]
```

## Yöntem

Simülasyonda robot, 2D kare bir alanda başlangıç noktasından hedef noktasına gitmeye çalışır. Ortamda dikdörtgen engeller vardır. Robotun gözlemi; hedefe göre konum, engellere olan mesafe, yakın çarpışma sinyalleri ve ışın tabanlı basit sensör ölçümlerinden oluşur.

Deneyde üç ana kontrolcü fikri karşılaştırıldı:

| Kontrolcü | Açıklama |
| --- | --- |
| Sabit politika | Geometrik kurallara dayalı, elle yazılmış güvenli başlangıç politikası |
| Saf NCP | Kararı tamamen `CfC` veya `LTC` Neural Circuit Policy çıktısı verir |
| Residual NCP | Sabit politikanın skorlarının üzerine NCP düzeltmesi eklenir |

Bu ayrım önemlidir. Saf NCP, modelin tek başına yeterli olup olmadığını test eder. Residual NCP ise gerçek robotikte daha makul bir güvenlik yaklaşımını temsil eder: önce basit ve güvenli bir kural tabanı çalışır, öğrenen model sadece düzeltme yapar.

## Kullanılan Modeller

Projede MIT'nin resmi `ncps` kütüphanesi kullanıldı:

- `ncps.torch.CfC`
- `ncps.torch.LTC`
- `ncps.wirings.AutoNCP`

`CfC` ve `LTC`, Liquid Neural Network ailesiyle ilişkili sürekli zamanlı veya sürekli zamana yakın nöral dinamikler sunar. Burada amaç, bu mimarilerin küçük bir navigasyon probleminde saf kontrolcü ve residual düzeltici olarak davranışını gözlemlemektir.

## Deneyler

Deney seti üç parçadan oluşur:

1. **2D simülasyon ve görselleştirme:** Robotun engeller arasında hareketi PNG ve GIF olarak kaydedildi.
2. **Zorlu haritalar:** Zigzag koridor, yoğun labirent, aldatıcı U-tuzak, sensör gölgesi ve labirent senaryoları eklendi.
3. **Ablation çalışması:** Resmi `ncps` modelleri imitation learning ile eğitildi, kısa bir RL fine-tune uygulandı, ardından pure-vs-residual ve CfC-vs-LTC karşılaştırmaları yapıldı.

Etkileşimli arayüz de eklendi. Bu arayüzde kullanıcı haritanın adını yazabilir, engelleri kendi yerleştirebilir, başlangıç ve hedef noktalarını değiştirebilir, maksimum simülasyon adımını ayarlayabilir, hazır parametre setup'larını seçebilir ve haritaları kaydedip tekrar yükleyebilir.

## Temel Sonuçlar

Aşağıdaki tablo `results/ncp_ablation_group_summary.csv` dosyasından özetlenmiştir. Değerler küçük ölçekli deneylerin başarı ve çarpışma oranlarını gösterir.

| Denetleyici | Varyant | Default başarı | Default çarpışma | Zorlu harita başarı | Zorlu harita çarpışma |
| --- | --- | ---: | ---: | ---: | ---: |
| Sabit politika | Baseline | 0.750 | 0.000 | 0.500 | 0.500 |
| CfC NCP | Pure + RL | 0.000 | 1.000 | 0.000 | 1.000 |
| LTC NCP | Pure + RL | 0.000 | 1.000 | 0.000 | 1.000 |
| CfC NCP | Residual + RL | 1.000 | 0.000 | 0.600 | 0.400 |
| LTC NCP | Residual + RL | 0.875 | 0.000 | 0.300 | 0.700 |

Bu sonuçlardan çıkan ana yorumlar:

- Saf NCP politikaları bu kısa eğitim düzeninde güvenli davranış öğrenemedi.
- Residual yapı, saf NCP'ye göre belirgin şekilde daha kararlı oldu.
- `CfC + residual + RL fine-tune`, zorlu haritalarda sabit politikadan daha yüksek başarı ve daha düşük çarpışma oranı verdi.
- `LTC + residual` default haritalarda makul çalışsa da zorlu haritalarda CfC kadar iyi sonuç vermedi.
- H6'nın "öğrenen liquid politika adaptasyon sağlayabilir" kısmı sınırlı destek aldı; "güvenlik süpervizörü gerekir" kısmı ise daha güçlü desteklendi.

## Örnek Görseller

Ablation başarı/çarpışma özeti:

![NCP ablation başarı ve çarpışma grafiği](figures/ncp_ablation_success_collision.png)

Labirent simülasyonu:

![Labirent simülasyonu](figures/h6_2d_real_ncp_cfc_labyrinth.gif)

Zorlu harita örnekleri:

![Zorlu harita genel görünümü](figures/h6_hard_map_overview.png)

## Kurulum

Bu proje Python ile çalışır. Windows ortamında bu projede kullanılan Python yolu:

```powershell
C:\ProgramData\miniconda3\python.exe -m pip install -r requirements.txt
```

Gerekli paketler:

```text
numpy
matplotlib
ncps
torch
```

## Google Colab Notebook

Colab üzerinde denemek için repo kökünde bir notebook hazırlandı:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heimdilon/hypothesis-6-lnn-neurosymbolic/blob/main/notebooks/h6_lnn_colab_demo.ipynb)

Notebook şu işleri yapar:

- Private repo için GitHub token ile clone alma
- `requirements.txt` bağımlılıklarını kurma
- Kısa `CfC residual` smoke simülasyonu çalıştırma
- Hazır PNG/GIF sonuçlarını gösterme
- Arayüz sunucusunu Colab iframe içinde açma
- İsteğe bağlı küçük CfC/LTC ablation deneyi çalıştırma

## Etkileşimli Arayüzü Çalıştırma

```powershell
C:\ProgramData\miniconda3\python.exe src\custom_map_server.py --port 8765
```

Sonra tarayıcıda şu adres açılır:

```text
http://127.0.0.1:8765
```

Arayüzde yapılabilenler:

- Harita adı verme
- Engel ekleme, taşıma ve silme
- Start ve goal noktalarını değiştirme
- Maksimum simülasyon adımını ayarlama
- Liquid hücre tipi seçme: `MIT ncps CfC`, `MIT ncps LTC`, `legacy`
- NCP nöron sayısı, sparsity, residual scale ve learning rate gibi parametreleri slider ile değiştirme
- Hazır setup seçme
- Haritayı kaydetme ve tekrar yükleme
- Simülasyon ilerlemesini progress bar ile izleme
- Harita adına göre PNG/GIF/JSON çıktı üretme

## Ablation Deneyini Çalıştırma

```powershell
C:\ProgramData\miniconda3\python.exe src\train_ncp_ablation.py --train-sequences 48 --val-sequences 12 --seq-len 24 --imitation-epochs 5 --rl-episodes 6 --eval-episodes 2 --hidden-dim 24
```

Bu komut şu dosyaları üretir:

- `results/ncp_training_log.csv`
- `results/ncp_ablation_episode_results.csv`
- `results/ncp_ablation_group_summary.csv`
- `results/ncp_residual_vs_pure_summary.csv`
- `results/ncp_cfc_vs_ltc_summary.csv`
- `results/ncp_ablation_scenario_summary.csv`
- `results/ncp_ablation_summary.md`
- `figures/ncp_ablation_success_collision.png`
- `results/models/ncp_cfc_imitation.pt`
- `results/models/ncp_cfc_rl_finetune.pt`
- `results/models/ncp_ltc_imitation.pt`
- `results/models/ncp_ltc_rl_finetune.pt`

## Klasör Yapısı

```text
hypothesis_6_lnn_neurosymbolic/
├── src/
│   ├── run_lnn_experiment.py       # Ana 2D navigasyon simülasyonu
│   ├── train_ncp_ablation.py       # CfC/LTC imitation, RL ve ablation deneyleri
│   ├── custom_map_server.py        # Yerel web arayüzü sunucusu
│   ├── make_2d_gif.py              # GIF üretimi
│   └── plot_hard_maps.py           # Zorlu harita görselleri
├── ui/
│   ├── index.html                  # Harita editörü arayüzü
│   ├── app.js                      # Arayüz mantığı
│   └── styles.css                  # Arayüz stilleri
├── figures/
│   ├── ncp_ablation_success_collision.png
│   ├── h6_2d_real_ncp_cfc_labyrinth.gif
│   └── custom_maps/                # Kullanıcı haritası çıktıları
├── results/
│   ├── ncp_ablation_summary.md
│   ├── ncp_ablation_group_summary.csv
│   ├── h6_ncp_project_report.tex
│   └── h6_ncp_project_report.pdf
├── saved_maps/
│   └── labyrinth_custom.json
├── requirements.txt
└── README.md
```

## Bilimsel Yorum

Bu çalışma bir son ürün robot kontrol sistemi değildir. Daha doğru ifade ile, literatürden çıkarılan H6 hipotezini sınamak için kurulmuş bir **araştırma prototipidir**.

Deneylerin bilimsel katkısı şudur: Liquid Neural Network ailesindeki resmi NCP katmanları, saf politika olarak kısa eğitimde başarısız olurken, güvenli bir sabit politikanın üstüne residual düzeltici olarak eklendiğinde daha anlamlı davranış üretmiştir. Bu, öğrenen kontrolcülerin mobil robot navigasyonunda tek başına kullanılmasından çok, kural tabanlı veya sembolik güvenlik katmanlarıyla birlikte kullanılmasının daha savunulabilir olduğunu gösterir.

Bu sonuç özellikle H6 için önemlidir. Çünkü hipotez yalnızca "LNN daha iyi adapte olur" dememektedir; aynı zamanda "kararlılık riski vardır ve bu risk güvenlik süpervizörüyle sınırlandırılmalıdır" demektedir. Bu projedeki en güçlü bulgu da bu ikinci kısımdadır.

## Sınırlılıklar

- Deney 2D simülasyon düzeyindedir; fiziksel robot validasyonu yapılmamıştır.
- Eğitim bütçesi küçüktür; daha fazla seed, daha uzun RL eğitimi ve daha fazla harita gerekir.
- Sensör modeli basitleştirilmiştir.
- Güvenlik süpervizörü şu anda pratik bir residual/baseline mekanizmasıdır; formel CBF, MPC veya erişilebilirlik analizi eklenmemiştir.
- Sonuçlar hipotez taraması için uygundur; genellenebilir robotik iddiası için daha büyük deney seti gerekir.

## Sonraki Adımlar

- Daha fazla random seed ile istatistiksel güven aralığı raporlamak
- Dinamik engeller ve sensör gürültüsü eklemek
- Residual NCP'yi formel güvenlik filtresiyle birleştirmek
- MPC veya Control Barrier Function tabanlı süpervizör eklemek
- Daha uzun imitation/RL eğitimi yapmak
- Gerçek robot veya daha gerçekçi fizik simülatörüne geçmek

## Sunumda Nasıl Anlatılır?

1. Önce iki derleme makaleyi tanıt: biri öz-öğrenme/LNN tarafını, diğeri engel kaçınma ve yol planlama algoritmalarını sınıflandırıyor.
2. Sonra seçilen araştırma boşluğunu söyle: öğrenen navigasyon politikaları adaptif olabilir ama güvenlik ve kararlılık tarafı zayıf.
3. Hipotez 6'yı açıkla: LNN/NCP adaptasyon sağlayabilir, fakat güvenlik süpervizörü gerektirir.
4. Simülasyonu göster: haritalar, engeller, start-goal, GIF çıktıları ve web arayüzü.
5. Sonuç tablosunu yorumla: saf NCP başarısız, residual NCP daha güvenli, CfC residual + RL zorlu haritalarda en iyi sonuç verdi.

Tek cümlelik kapanış:

> Bu proje, Liquid Neural Network tabanlı navigasyonun potansiyelini gösterirken, mobil robot güvenliği için öğrenen kontrolcünün tek başına değil, güvenli bir residual veya süpervizör yapısıyla birlikte kullanılmasının daha bilimsel ve savunulabilir olduğunu göstermektedir.
