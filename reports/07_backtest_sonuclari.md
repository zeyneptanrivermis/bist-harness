# 07 – Backtest Sonuçları ve Benchmark Karşılaştırması

Analiz tarihi 2026-09-10, dönem 2024-01-02 – 2026-09-10, 30 hisse. Sayılar `reports/evidence/strategy_variants.csv` dosyasından gelir:

```bash
python scripts/run_strategy_variants.py
```

Eğitsel backtest etiketleridir; geçmiş performans gelecek için garanti değildir, yatırım tavsiyesi değildir.

## 1. Yöntem

- **Strateji varyantları (A–E)** (`src/strategies/variants.py`), her biri bir öncekinin üzerine koşul ekler, "ya hep ya hiç" nakit/uzun mantığıyla:
  - **A** teknik: Supertrend yukarı **ve** MACD histogramı > 0.
  - **B** + sektör: A ve hissenin 20 günlük getirisi grup medyanından 5 puandan fazla geride değil.
  - **C** + temel: B ve son açıklanan çeyrekte FAVÖK marjı düşmedi ve kaldıraç artmadı.
  - **D** + makro: C ve son 20 işlem gününde USD/TRY %8'den fazla artmadı.
  - **E** + haber: D ve son 5 işlem gününde hissede olumsuz haber yok.
- **Zamanlama:** t günü sinyali en erken t+1'de uygulanır (ileri bakış yok). Hepsi analiz tarihinde kesilmiş veriyle çalışır.
- **Maliyet:** her işlemde %0,10 komisyon ve %0,05 kayma (varsayılan oranlar, `src/config.py`).
- **Benchmarklar:** XU100 endeksi (aynı dönemde **%+88,8**) ve hissenin kendi **al-tut** getirisi. `A − XU100` sütunu strateji getirisi eksi XU100 getirisidir (alfa tanımımız; risk düzeltmesi yoktur).
- **Sharpe:** yıllık %40 risksiz faiz varsayımıyla hesaplanır (`risk_free_rate=0.40`). Bu yüzden getirisi bu seviyenin altında kalan her strateji **negatif** Sharpe verir; negatif Sharpe tek başına "kayıp" demek değildir.
- **Eksik katmanlar:** bir katman için veri yoksa o katman sessizce geçer ve `flags` ile belirtilir (aşağıdaki tabloda "eksik katman"). Böylece bir varyant, verisi olmadığı hâlde "temel" diye sunulmaz.

## 2. Katmanlar gerçekte ne kadar çalıştı?

| Katman | Verisi olan hisse | Sonuç |
|---|---|---|
| B (sektör) | 12 / 30 (yalnızca sektöründe 2+ hisse olan) | 12 hissede A'dan farklı sinyal üretir |
| C (temel) | 5 / 30 (ASELS, BIMAS, FROTO, THYAO, TUPRS) | B'den farklı sinyal yalnızca 4 hissede: ASELS, FROTO, THYAO, TUPRS (A'dan farklı toplam 14 hisse) |
| D (makro) | 30 / 30 (USD/TRY) | **Hiçbir hissede C'den farklı sinyal üretmedi:** USD/TRY'nin 20 günlük en büyük artışı %4,8, %8 eşiği hiç aşılmadı |
| E (haber) | **0 / 30** (haber dosyası boş) | E, D ile **birebir aynı**; fiilen ayrı bir varyant değil |

Sonuç: **E'nin haber katmanı henüz hiçbir şey ölçmüyor**, D'nin makro filtresi de bu dönemde sinyali değiştirmedi. Pratikte yalnızca A, B ve C farklıdır.

## 3. Varyant özeti (30 hisse)

| Varyant | Medyan getiri | Ortalama getiri | Medyan al-tut | Medyan Sharpe | Medyan azami kayıp | Medyan isabet | Medyan işlem | Al-tut'u geçen | XU100'ü geçen |
|---|---|---|---|---|---|---|---|---|---|
| A | %12,4 | %24,5 | %85,2 | −1,16 | %−27,8 | %39,4 | 19,5 | 4 / 30 | 3 / 30 |
| B | %11,8 | %19,7 | %85,2 | −1,22 | %−26,2 | %41,3 | 20 | 4 / 30 | 2 / 30 |
| C | %11,8 | %18,0 | %85,2 | −1,22 | %−26,2 | %41,7 | 20 | 4 / 30 | 2 / 30 |
| D | %11,8 | %18,0 | %85,2 | −1,22 | %−26,2 | %41,7 | 20 | 4 / 30 | 2 / 30 |
| E | %11,8 | %18,0 | %85,2 | −1,22 | %−26,2 | %41,7 | 20 | 4 / 30 | 2 / 30 |

**Ana sonuç:** hiçbir varyant al-tut'u ya da XU100'ü sistematik biçimde geçmiyor. Medyan strateji getirisi %12 civarında iken medyan al-tut %85, XU100 %88,8. Al-tut'u geçen hisseler: CCOLA, SASA, ARCLK, MAVI (hepsinde al-tut düşük ya da negatif). A'nın XU100'ü geçtiği 3 hisse: ASELS, GUBRF, AKSEN. Ek katmanlar (B–E) A'ya göre belirgin bir iyileşme getirmedi: medyan getiri %12,4'ten %11,8'e, medyan Sharpe −1,16'dan −1,22'ye indi.

**Örnek dışı dilim (2026-01-01 sonrası, 30 hisse medyanı):**

| Varyant | Medyan getiri | Medyan al-tut | Medyan Sharpe | Medyan işlem |
|---|---|---|---|---|
| A | %0,9 | %12,7 | −1,37 | 5 |
| B | %1,8 | %12,7 | −1,37 | 6 |
| C, D, E | %1,8 | %12,7 | −1,39 | 5,5 |

Örnek dışı dilimde de al-tut'un gerisindeler. Bu dilim kısa (yaklaşık 8 ay, hisse başına 5–6 işlem), istatistiksel güç düşüktür.

## 4. Hisse bazında sonuçlar (al-tut getirisine göre sıralı)

| Hisse | A getiri | E getiri | Al-tut | A − XU100 | A Sharpe | A azami kayıp | A işlem | Eksik katman |
|---|---|---|---|---|---|---|---|---|
| ASELS | %294.9 | %253.6 | %752.2 | %206.1 | 0.79 | %-22.1 | 20 | haber yok; sektör yok |
| TUPRS | %44.3 | %32.6 | %278.9 | %-44.5 | -0.83 | %-20.9 | 22 | haber yok; sektör yok |
| EKGYO | %-3.8 | %-3.8 | %207.3 | %-92.6 | -1.09 | %-29.2 | 19 | temel yok; haber yok; sektör yok |
| MPARK | %32.2 | %32.2 | %202.6 | %-56.6 | -0.84 | %-26.0 | 16 | temel yok; haber yok; sektör yok |
| BIMAS | %-9.5 | %-29.7 | %200.1 | %-98.3 | -1.68 | %-43.9 | 26 | haber yok |
| ENKAI | %14.0 | %14.0 | %199.6 | %-74.8 | -1.18 | %-26.8 | 24 | temel yok; haber yok; sektör yok |
| GUBRF | %100.0 | %100.0 | %198.3 | %11.2 | -0.13 | %-28.9 | 19 | temel yok; haber yok; sektör yok |
| ENJSA | %61.6 | %36.2 | %182.4 | %-27.2 | -0.63 | %-17.5 | 17 | temel yok; haber yok |
| AKSEN | %106.4 | %44.9 | %173.7 | %17.6 | -0.14 | %-21.3 | 20 | temel yok; haber yok |
| TURSG | %14.5 | %14.5 | %166.8 | %-74.2 | -1.13 | %-30.0 | 21 | temel yok; haber yok; sektör yok |
| TRALT | %-46.0 | %-46.0 | %163.4 | %-134.8 | -1.70 | %-56.3 | 29 | temel yok; haber yok; sektör yok |
| TTKOM | %-40.6 | %-58.4 | %121.8 | %-129.4 | -2.30 | %-54.4 | 22 | temel yok; haber yok |
| AKBNK | %-3.2 | %-3.2 | %117.2 | %-92.0 | -1.20 | %-37.9 | 20 | temel yok; haber yok; sektör yok |
| EREGL | %27.9 | %7.5 | %91.1 | %-60.9 | -0.97 | %-27.4 | 19 | temel yok; haber yok |
| TCELL | %-15.1 | %-22.0 | %86.0 | %-103.9 | -2.00 | %-42.0 | 22 | temel yok; haber yok |
| KRDMD | %-26.9 | %-23.7 | %84.5 | %-115.6 | -1.72 | %-53.9 | 26 | temel yok; haber yok |
| KCHOL | %17.6 | %17.6 | %77.0 | %-71.2 | -1.25 | %-23.8 | 22 | temel yok; haber yok; sektör yok |
| ECILC | %58.7 | %58.7 | %74.0 | %-30.1 | -0.41 | %-28.3 | 14 | temel yok; haber yok; sektör yok |
| MGROS | %36.5 | %16.2 | %67.0 | %-52.3 | -1.02 | %-22.6 | 24 | temel yok; haber yok |
| TOASO | %-6.2 | %12.8 | %63.9 | %-95.0 | -1.50 | %-25.7 | 19 | temel yok; haber yok |
| OYAKC | %50.6 | %50.6 | %57.1 | %-38.2 | -0.75 | %-23.7 | 16 | temel yok; haber yok; sektör yok |
| ISMEN | %1.5 | %1.5 | %43.1 | %-87.3 | -1.49 | %-25.8 | 18 | temel yok; haber yok; sektör yok |
| MAVI | %44.7 | %44.7 | %37.3 | %-44.1 | -0.87 | %-19.4 | 16 | temel yok; haber yok; sektör yok |
| PETKM | %-44.2 | %-44.2 | %33.7 | %-133.0 | -2.15 | %-52.7 | 20 | temel yok; haber yok; sektör yok |
| THYAO | %9.9 | %7.0 | %29.6 | %-78.9 | -1.70 | %-29.2 | 15 | haber yok |
| FROTO | %-8.4 | %-0.5 | %26.2 | %-97.2 | -1.81 | %-30.5 | 20 | haber yok |
| PGSUS | %-38.9 | %-35.2 | %11.9 | %-127.6 | -2.56 | %-56.9 | 19 | temel yok; haber yok |
| ARCLK | %10.7 | %10.7 | %-26.1 | %-78.1 | -1.59 | %-23.1 | 13 | temel yok; haber yok; sektör yok |
| SASA | %7.6 | %7.6 | %-39.2 | %-81.2 | -0.90 | %-29.1 | 14 | temel yok; haber yok; sektör yok |
| CCOLA | %44.0 | %44.0 | %-84.2 | %-44.8 | -0.89 | %-26.3 | 19 | temel yok; haber yok; sektör yok |

**Okuma:** ASELS tek başına A'nın en iyi sonucunu (%294,9) veriyor, ama al-tut %752,2: strateji nakitte bekleyip yükselişin büyük kısmını kaçırıyor. Yüksek getirili hisselerde (EKGYO, MPARK, BIMAS, ENKAI) strateji ya zararda ya da al-tut'un çok altında. Hiçbir hissede 10'dan az işlem yok (anekdot uyarısı çıkmadı).

## 5. Eksikler ve dürüst sınırlamalar

- **Sektör benchmark'ı kısmen var.** `src/backtest/sector_benchmark.py` sektördeki diğer hisselerin eşit ağırlıklı günlük getirisinden bir sepet kurar ve `Sector_Benchmark_Return` ile `Alpha_vs_Sector` verir (`run_backtest` aracı ve harness). Sektör etiketinin ilk kelimesi gruplama anahtarıdır ("Retail Trade" ve "Retail / Apparel" tek perakende grubudur); bu, ders çalışması için bir yaklaşımdır. Evrende 30 hissenin 17'sinde en az bir akran vardır (perakende, havayolu, çelik, telekom, kimya, otomotiv, sağlık, elektrik). Diğerleri için çıktı `unavailable` der, sayı uydurmaz. Varyant tablosu için `reports/evidence/strategy_variants_sector.csv` aynı 150 satırı sektör alfasıyla verir (`Sector_Benchmark_Return`, `Alpha_vs_Sector`, `oos_Alpha_vs_Sector`; `python scripts/run_strategy_variants.py reports/evidence/strategy_variants_sector.csv`). Akranı olan 17 hissede (85 satır) medyan sektör alfası A–E varyantlarında −%85 ile −%94 puan arasındadır ve yalnızca bir (hisse, varyant) çifti sektör sepetini geçmiştir. Sektör sepeti 2024-2026'da güçlü yükselen akranların al-tut getirisidir; Supertrend uzun/nakit stratejisi onların gerisinde kalır. Bu dosya yerel fiyat önbelleğinden yeniden üretildi; önbellek, ilk tabloyu üreten veriden biraz farklıdır (toplam getiri farkı ortalama 0,7 puan, en çok 9,6 puan), bu yüzden `strategy_variants.csv` ile birebir aynı sayılar beklenmemelidir. Bu rapordaki ana tablo `strategy_variants.csv`'den okunur.
- **Varyantlar ve eşikler** (5 puan, %8, 5 gün) önceden sabit seçildi, bu çalışmada optimize edilmedi; ama başka eşiklerle denenmediği için sağlamlıkları da bilinmiyor.
- **Hayatta kalma ve dönem yanlılığı:** evren 30 hisselik seçilmiş bir BIST örneği ve 2024–2026 güçlü yükselen bir piyasa; al-tut bu yüzden yüksek.
- **Temel veri** yalnızca 5 hisse için ve elle girilmiş; **haber** katmanı boş. C ve E sonuçları bu yüzden yorumlanamaz.
- Hisse başına 13–29 işlem var; tek hisse sonuçları anekdottur, medyanlar daha güvenilirdir.
- Eğitsel bir karşılaştırmadır; hiçbir varyant bir işlem önerisi değildir.
