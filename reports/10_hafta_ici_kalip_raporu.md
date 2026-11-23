# 10 – Hafta İçi ve Çok Günlü Kalıp Raporu (Senaryo 2, THYAO)

Analiz tarihi **2026-09-10**; bu tarihten sonraki veri kullanılmaz. Eğitsel araştırmadır, yatırım tavsiyesi değildir. Sayılar `reports/evidence/weekday_thyao.json` ve `reports/evidence/weekday_pattern_thyao.json` dosyalarındaki araç çıktısından alınmıştır (`python scripts/make_report_evidence.py` ile yeniden üretilir). PDF sürümü (`weekday_patterns_report.pdf`) ilk taslaktır. Bu belge güncel kanıtla örtüşür ve çoklu test düzeltmesini 32 aday dizi için yapar.

## 1. Soru ve hipotez

Hisselerde tekrarlayan gün ve çok günlü diziler var mı, maliyet sonrası ve görülmemiş dönemde sürüyor mu? (Şartname §12.)

- **H1:** Pazartesi, Salı ve Çarşamba düşen bir haftanın ardından Perşembe ve Cuma yükselir.
- **H0:** Bu haftalardan sonra Perşembe/Cuma getirileri diğer haftalardan farklı değildir.

## 2. Veri ve yöntem

| Öğe | Değer |
|---|---|
| Hisse | THYAO (Ulaştırma / Havayolu), günlük düzeltilmiş kapanış |
| Örnek içi | 2024-01-01 – 2025-12-31 (87 hafta) |
| Örnek dışı | 2026-01-01 – 2026-09-10 (30 hafta) |
| Maliyet | %0,10 komisyon + %0,05 kayma, gidiş-dönüş %0,3 |
| Zamanlama | Sinyal t'de bilinir, işlem t+1 (kalıp Çarşamba kapanışında bilinir, alım Çarşamba kapanışı, satış Cuma kapanışı) |
| Önem testi | Kalıp haftaları ile diğer haftaların Perşembe+Cuma getirisi karşılaştırması |
| Çoklu test | 32 aday haftalık yukarı/aşağı dizisi, Bonferroni α = 0,05 / 32 = 0,0016. Gün tablosu için 5 test, α = 0,01 |

## 3. Bulgular

### 3.1 Çok günlü dizi (ana test)

| Ölçü | Örnek içi | Örnek dışı |
|---|---|---|
| Hafta | 87 | 30 |
| Pzt-Sal-Çar düşen hafta | 19 | 3 |
| Perşembe ve Cuma birlikte yükselen | 6 / 19 (%31,6) | 0 / 3 |
| Perşembe ortalama / medyan | %1,21 / %0,97 | %0,68 / %0,58 |
| Cuma ortalama / medyan | %0,16 / %0,00 | −%0,54 / −%1,28 |
| Koşulsuz Perşembe ortalama | %0,20 | %0,09 |
| Koşulsuz Cuma ortalama | %0,33 | %0,56 |
| Çarşamba kapanışı → Cuma kapanışı, maliyet öncesi ortalama | %1,36 | %0,13 |
| **Aynı, maliyet sonrası ortalama** | **%1,06** | **−%0,17** |
| p (diğer haftalara karşı) | 0,2585 | hesaplanamaz (3 hafta) |

Örnek içinde dizi 19 haftada görüldü ve yalnızca 6'sında iki gün birlikte yükseldi; fark anlamlı değil (p = 0,26, düzeltilmiş eşik 0,0016). Maliyet sonrası %1,06'lık örnek içi ortalama, örnek dışında −%0,17'ye döndü. Örnek dışında yalnızca 3 olay var; bu istatistik değil anekdottur, ama yönü kalıbın tersidir.

### 3.2 Gün bazlı ortalamalar (THYAO)

| Gün | Örnek içi n | Ortalama | p | Maliyet sonrası ort. | Örnek dışı n | Ortalama | p |
|---|---|---|---|---|---|---|---|
| Pzt | 99 | −%0,05 | 0,83 | −%0,35 | 36 | −%0,12 | 0,74 |
| Sal | 98 | −%0,10 | 0,60 | −%0,40 | 35 | %0,22 | 0,51 |
| Çar | 99 | −%0,30 | 0,11 | −%0,60 | 35 | −%0,16 | 0,69 |
| Per | 102 | %0,40 | 0,016 | %0,10 | 35 | −%0,13 | 0,70 |
| Cum | 101 | %0,28 | 0,15 | −%0,02 | 34 | %0,62 | 0,056 |

Koşulsuz günlük ortalama: örnek içi %0,05, örnek dışı %0,07. Perşembe örnek içinde p = 0,016 ile tek başına anlamlı görünür, ama beş günlük tabloda Bonferroni eşiği 0,01'dir; geçmez ve örnek dışında işareti ters döner (−%0,13). Cuma örnek dışında pozitiftir ama örnek içinde değildi (p = 0,15). Bu iki bulgu aynı kalıbı doğrulamaz.

## 4. Çoklu test riski

Yüzlerce gün ve dizi kombinasyonu denendiğinde şans eseri kalıp çıkması beklenir. Ana test 32 aday haftalık diziden biridir, dolayısıyla eşik 0,0016'dır. 0,2585 buna yaklaşmaz. Gün tablosu beş test içerir ve Perşembe'nin p = 0,016 değeri 0,01 eşiğini geçmez.

## 5. Sonuç ve durum

**`REJECT_SIGNAL`:** THYAO'da "Pzt-Sal-Çar düşer, Per-Cum yükselir" kalıbı desteklenmiyor. Örnek içi kanıt zayıf, örnek dışı örneklem çok küçük ve ters yönde, gün tablosundaki tek anlamlı görünen sonuç düzeltmeden sonra geçmiyor. Kalıp işlem kuralı olarak sunulmaz.

## 6. Sınırlamalar

- Tek hisse. Sonuç diğer hisselere ya da sektöre genellenemez; ikinci bir hissede ters kalıp çıkabilir.
- Örnek dışında yalnızca 3 olay var.
- Üç gün üst üste düşüş sonrası toparlanma ve büyük düşüş günü sonrası dönüş (`detect_multiday_patterns`) kod olarak mevcuttur, ama bu raporda THYAO için kanıt dosyası üretilmemiştir; sonuçları bu rapora dahil değildir.
- Piyasa rejimi (XU100 yönü), oynaklık rejimi ve makro olay haftaları üzerinden koşullama bu raporda yok.
- Fiyatlar Yahoo Finance kaynaklıdır; analiz tarihi sabittir.

Ayrımlar: tahmin ≠ kârlı strateji, korelasyon ≠ nedensellik, geçmiş testi ≠ gelecek garantisi, tarihsel kanıt ≠ yatırım tavsiyesi.

## 7. Yeniden üretme

```bash
python scripts/make_report_evidence.py   # reports/evidence/weekday_thyao.json, weekday_pattern_thyao.json
```

Kod: `src/scenarios/weekday_patterns.py`, `src/mcp_tools/research.py` (`test_weekday_pattern_summary`).
