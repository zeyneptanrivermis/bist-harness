# 02 – Teknik Dönüş Olayları Raporu (ASELS, Bollinger alt bant teması)

Analiz tarihi 2026-09-10, dönem 2024-01-01 – 2026-09-10. Tüm sayılar `reports/evidence/reversal_asels.json` dosyasından gelir. Yeniden üretmek için:

```bash
python scripts/make_report_evidence.py
```

Eğitsel analizdir, yatırım tavsiyesi değildir.

## 1. Soru

ASELS fiyatı alt Bollinger bandına kaç kez dokundu, kaç olayda fiyat 5 gün içinde yükseldi, ve RSI dibi + yüksek hacimle gelen temaslar, Supertrend düşüş yönünde ve fiyat Ichimoku bulutunun altındayken gelen temaslardan farklı mı davrandı?

## 2. Yöntem

- Olay: `BB_Lower_Touch` (fiyatın alt banda temas ettiği gün t). Giriş **t+1** gününün düzeltilmiş kapanışıdır, çıkış girişten 1, 3, 5 ve 10 işlem günü sonradır. "İsabet" = ilgili ufukta getirinin pozitif olması.
- Tablolardaki getiriler **brüttür**. Aracın ayrıca hesapladığı maliyet sonrası (net) getiri sütunları (`Fwd_<h>d_Net_Return`, gidiş-dönüş komisyon ve kayma düşülmüş) bu özete girmedi.
- Yeterince ileri veri olmayan olaylar ortalamalardan çıkarılır (`excluded_incomplete_5d` = 0).
- Tek koşul (baz) ile birleşik koşullar karşılaştırılır. 10'dan az olay içeren gruplar "anekdot" uyarısı alır.

## 3. Bulgular

**34 alt bant teması** (2024-01-01 – 2026-09-10).

| Ufuk | n | Ortalama | Medyan | İsabet oranı |
|---|---|---|---|---|
| 1 gün | 34 | %0,55 | %0,77 | %55,9 |
| 3 gün | 34 | %1,45 | %1,13 | %55,9 |
| 5 gün | 34 | %2,04 | %1,78 | %67,6 |
| 10 gün | 34 | %1,87 | %0,72 | %55,9 |

**Tek koşul ve birleşik koşullar (5 gün):**

| Koşul | n | 5g isabet | 5g ortalama |
|---|---|---|---|
| 1. Tek olay (baz) | 34 | %67,6 | %2,04 |
| 2. Olay + RSI ≤ 35 | 7 | %57,1 | %0,10 |
| 3. Olay + hacim sıçraması (> 1,2×) | 11 | %54,5 | %0,09 |
| 4. Olay + KAMA düşmüyor | 10 | %60,0 | %1,90 |
| 5. Olay + RSI ≤ 35 + hacim + KAMA sakin | **0** | – | – |
| 6. Olay ama Supertrend düşüşte ve fiyat bulutun altında | 18 | %66,7 | %3,15 |
| 7. Olay + RSI ≤ 35 + hacim sıçraması | **3** | %33,3 | %−0,79 |

## 4. Sorunun cevabı ve tuzak

"RSI dibi + yüksek hacim" (grup 7, n=3, %33,3) ile "Supertrend düşüşte + bulutun altında" (grup 6, n=18, %66,7) karşılaştırması **tersine bir izlenim verir**: kâğıt üstünde daha kötü görünen koşul, daha iyi görünen koşuldan yüksek isabet ve ortalama getiri üretmiş.

Bu bir **tuzaktır**, sonuç olarak okunmamalıdır:
- Grup 7'nin yalnızca 3 olayı var; 3 olayda tek isabet bir "oran" değildir.
- Grup 5 (ideal paket) hiç oluşmamış, yani "ideal" kombinasyon bu hissede test edilemiyor.
- Gruplar iç içe ve birbirini kısmen kapsıyor, bağımsız örnek değiller.
- Yedi grup birlikte denendi; çoklu test düzeltmesi uygulanmadı, bir grubun iyi görünmesi şansla da olabilir.

## 5. İstatistiksel bağlam (bu raporun hesapları)

| Ölçüt | Değer |
|---|---|
| Baz grup, 5g isabet (23/34), eşit olasılığa karşı çift yönlü kesin binom p | 0,058 |
| Grup 6 (12/18) için aynı test | 0,238 |
| Grup 7 (1/3) için aynı test | 1,0 |
| ASELS tüm günler (olay şartsız), aynı kuralla (giriş t+1, çıkış 5 gün sonra, çıkış ≤ 2026-09-10; 669 gün) pozitif getiri oranı | %59,2, ortalama %1,77 |

Yorum: baz grubun %67,6'sı, hissenin genel eğilimi olan %59,2'nin üzerinde ama fark küçük ve örneklem 34. Alt bant teması bu hissede **kanıtlanmış bir dönüş sinyali değildir**. Birleşik koşulların hiçbiri bazdan güvenilir biçimde ayrışmıyor. ASELS 2024-2026'da güçlü yükselmiş bir hisse olduğu için her tür "al" kuralı yüksek isabet gösterebilir (hayatta kalma ve trend yanlılığı).

Sistemin verdiği durum `TECHNICAL_REVERSAL_CANDIDATE`; ancak birleşik grupların çoğu 10 olaydan az olduğu için araç küçük örneklem uyarısı veriyor: *"Small sample: fewer than 10 events in … (n=7) … (n=0) … (n=3). Their hit rates are anecdotal."*

## 6. Eksik kalanlar (bilinçli olarak yapılmadı)

Şartnamedeki olay ailelerinden şunlar uygulanmadı:
- Destek ve direnç temasları (`find_support_resistance` bir gösterge/araç olarak var, olay ailesi olarak yok).
- RSI tepe ve dip noktalarının tespiti (şu an yalnızca RSI ≤ 30 aşırı satım, ≥ 70 aşırı alım eşiği).
- KAMA aşağı kesişimi (yalnızca yukarı kesişim `KAMA_CROSS_UP` var).
- Ichimoku bulut reddi (yalnızca bulut kırılımı `ICHIMOKU_CLOUD_BREAKOUT` var).
- Volatilite rejimleri: hiç uygulanmadı. Bollinger bant genişliği (`BB_Bandwidth`) ve `ATR_14` göstergeleri var ama olayları rejime göre ayırmıyoruz.

## 7. Sınırlamalar

Tek hisse, tek olay ailesi, 34 olay. Sonuçlar diğer hisselere genellenemez. Özet tablodaki getiriler maliyet düşülmeden verilmiştir (bkz. §2); olay çalışması işlem stratejisi değildir.
