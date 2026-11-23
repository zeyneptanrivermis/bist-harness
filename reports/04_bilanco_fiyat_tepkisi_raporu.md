# 04 – Çeyreklik Bilanço ve Fiyat Tepkisi Raporu (TUPRS)

> **Tüm sayılar örnek/gösterim amaçlıdır.** Temel veri dosyası elle girilmiş (curated) kayıtlardan oluşur, KAP bildirim numarası ya da başka bir kaynak kimliği taşımaz ve yalnızca 5 hisseyi kapsar. Bulgular gerçek şirket verisi olarak sunulamaz. Eğitsel analizdir, yatırım tavsiyesi değildir.

Analiz tarihi 2026-09-10. Sayılar `reports/evidence/fundamental_change_tuprs.json` ve `fundamental_metrics_tuprs.json` dosyalarından gelir (`python scripts/make_report_evidence.py`).

## 1. Soru

2024-01-01 – 2026-09-10 arasında kamuya açıklanan tüm TUPRS çeyreklik tablolarında FAVÖK marjı ve Net Borç/FAVÖK nasıl değişti? Açıklama gününden sonra 1, 5 ve 20 günlük getiri, XU100'e ve sektöre göre ne oldu? İyileşen çeyrekler daha iyi fiyat tepkisi aldı mı?

## 2. Yöntem

- Bir çeyrek, yalnızca `kap_disclosure_date` gününden sonra bilinir. Giriş, açıklama gününü izleyen **işlem günü (t+1)**; getiri bu girişten 1, 5 ve 20 işlem günü sonra ölçülür.
- Anormal getiri (`car_xu100_*`): hissenin aynı pencerede XU100'e göre fazla getirisi.
- `improving` = FAVÖK marjı önceki çeyreğe göre **arttı** ve Net Borç/FAVÖK **azaldı** (ikisi de kesin eşitsizlik).
- Açıklamadan sonra yeterli işlem günü geçmemiş pencereler ortalamalardan çıkarılır (2026Q2'nin 20 günlüğü, 2026-09-10 itibarıyla tamamlanmamış).

## 3. Bulgular (8 çeyrek, 2024Q3 – 2026Q2)

| Çeyrek | Açıklama | FAVÖK marjı | Marj değişimi | ND/FAVÖK | ND/FAVÖK değişimi | İyileşti mi | CAR XU100 1g | 5g | 20g |
|---|---|---|---|---|---|---|---|---|---|
| 2024Q3 | 2024-11-05 | %9,5 | – | −0,12 | – | hayır (önceki çeyrek yok) | %−0,6 | %−3,7 | %−6,0 |
| 2024Q4 | 2025-03-04 | %10,3 | +0,8 puan | −0,13 | −0,01 | evet | %+3,5 | %+7,0 | %+17,6 |
| 2025Q1 | 2025-05-08 | %10,5 | +0,2 | −0,15 | −0,02 | evet | %+0,7 | %+1,4 | %−0,8 |
| 2025Q2 | 2025-08-14 | %10,9 | +0,4 | −0,15 | −0,01 | evet | %−1,5 | %+1,5 | %+11,2 |
| 2025Q3 | 2025-11-06 | %10,8 | −0,1 | −0,16 | −0,01 | hayır | %+0,5 | %+3,9 | %−1,0 |
| 2025Q4 | 2026-03-05 | %10,9 | +0,1 | −0,17 | −0,01 | evet | %−2,3 | %−0,2 | %+1,0 |
| 2026Q1 | 2026-05-07 | %10,9 | 0,0 | −0,18 | −0,01 | hayır (marj sabit) | %−0,1 | %+4,6 | %+1,1 |
| 2026Q2 | 2026-08-13 | %11,0 | +0,1 | −0,19 | −0,01 | evet | %+2,2 | %+10,0 | tamamlanmadı |

Genel görünüm: FAVÖK marjı %9,5'ten %11,0'a yükseldi, Net Borç/FAVÖK negatifte (net nakit pozisyonu) kalıp −0,12'den −0,19'a indi.

**Sektör karşılaştırması yapılamadı:** TUPRS'in veride sektör arkadaşı yok (Petroleum / Refining sektöründe tek hisse), bu yüzden `car_sector_*` alanları boş. Soru "sektöre göre" kısmını cevaplayamıyor.

## 4. "İyileşen çeyrekler daha iyi tepki aldı mı?"

Araç çıktısındaki ortalama CAR (XU100'e göre):

| Grup | Çeyrek | 1g | 5g | 20g |
|---|---|---|---|---|
| İyileşen | 5 | %+0,53 | %+3,95 | %+7,23 (4 çeyrek) |
| Diğer | 3 | %−0,05 | %+1,60 | %−1,94 |

İlk bakışta evet. Ama **bu fark sağlam değil**:
- "Diğer" grubundaki 2024Q3, serinin ilk çeyreği olduğu için önceki çeyrek yok ve "iyileşmedi" diye sayılıyor; gerçekte değerlendirilemeyen bir çeyrek. Onu çıkarınca "diğer" grup 2025Q3 ve 2026Q1'den oluşuyor (1g %+0,22, 5g %+4,25, 20g %+0,09). **5 günde iyileşenler artık daha iyi değil** (%3,95'e karşı %4,25).
- Karşılaştırılan gruplar 5'e 3 (ya da 4'e 2) çeyrek. Bu örneklemde anlamlılık testi yapmak mümkün değil.
- 20 günlük farkı büyük ölçüde tek çeyrek yaratıyor: 2024Q4'ün %+17,6'sı.
- Araç, sonucu **nedensellik değil ilişki** olarak etiketler ("Association only"), biz de öyle okuyoruz.

Araç durumu: `INVESTIGATE`.

## 5. Sınırlamalar

- 8 çeyrek, 1 hisse, elle girilmiş veri, kaynak kimliği yok. Para birimi/ölçek dosyada belirtilmemiş; yalnızca oranlar (marj, ND/FAVÖK) kullanıldı.
- TMS 29 enflasyon düzeltmesi etkisi ayrıca modellenmedi.
- Fiyat tepkisi tek bir gelişmeyi (bilanço) diğerlerinden (makro haberler, endeks hareketi) ayıramaz; CAR yalnızca XU100'ü kontrol eder.
- Temel verisi olan diğer 4 hisse (ASELS, BIMAS, FROTO, THYAO) için aynı analiz bu raporda gösterilmedi. Verisi olmayan 25 hissede model `INVESTIGATE` demelidir.
