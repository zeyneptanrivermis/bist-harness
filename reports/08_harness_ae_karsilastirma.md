# 08 – Ajan Harness Deneyi (A–E) Karşılaştırması

> **Önemli açıklama:** Bu raporun A–E tablosundaki sayılar **simüle edilmiş bir sağlayıcıdan** gelir. Hiçbir gerçek model çalıştırılmamıştır. Tablo, deney hattının doğru çalıştığını gösterir, **model kalitesini ölçmez**. Gerçek modelle A–E koşusu bu çalışmada **yapılmadı**; nedenleri ve gerçek model için elimizde olan ölçüm aşağıdadır.

## 1. Deney nedir?

Aynı 12 sabit vaka (4 normal, 8 tuzak) beş mimariden geçirilir; mimariler yeteneklerin kademeli eklenmesiyle ayrılır (`src/experiments/policy.py`):

| Sürüm | Yetenekler |
|---|---|
| A – ham LLM | Araç yok |
| B – araçlar | + onaylı araç çağrıları |
| C – durum makinesi | + zorunlu durum geçişleri (`START → … → CHECK_DATA_QUALITY`) |
| D – kanıt + kalite kapısı | + veri kalitesi kapısı ve kanıt kimlikleri (E1, E2) |
| E – replay + insan onayı | + tekrar oynatılabilir kayıt ve insan onayı |

Vakalar: 4 normal (teknik dönüş, hafta içi, sektör, temel) ve 8 tuzak (ileri bakış, eksik veri, tekrarlayan tarih, kısa geçmiş, bilinmeyen hisse, kanıtsız iddia, doğrudan al/sat isteği, doğrulanmamış kaynak). Puanlama kuralları `src/experiments/evaluator.py` içindedir.

## 2. Simülasyon sonucu (`reports/ae_harness_demo/`)

```bash
python -m src.experiments.cli --output-dir reports/ae_harness_demo
```

Sağlayıcı `simulated-demo` (`DemonstrationProvider`): sürüm A için sabit bir "AL: kesin yükselir" metni, diğer sürümler için kurala göre hazır cevaplar üretir. Sonuçlar bu yüzden **önceden belli**.

| Sürüm | Geçen vaka | Sayısal halüsinasyon | Geçersiz araç | İleri bakış | Doğrudan işlem tavsiyesi |
|---|---|---|---|---|---|
| A – ham LLM | %0,0 | %100 | %0,0 | %8,3 | %100 |
| B – araçlar | %83,3 | %0,0 | %8,3 | %8,3 | %0,0 |
| C – durum makinesi | %91,7 | %0,0 | %0,0 | %8,3 | %0,0 |
| D – kanıt + kapı | %100 | %0,0 | %0,0 | %0,0 | %0,0 |
| E – replay + onay | %100 | %0,0 | %0,0 | %0,0 | %0,0 |

Bu tablo **"daha çok kontrol, daha az hata" diye okunamaz**: A'nın %100 hata vermesi, sağlayıcının A için hep aynı hatalı metni üretmesinden gelir.

## 3. Neden gerçek modelle olduğu gibi çalıştırılamıyor (doğrulanmış engeller)

1. **Sayısal doğruluk ölçütü gerçek çıktıya uymuyor.** 12 vakanın 11'inde `known_numeric_values` boştur. Değerlendirici, cevapta geçen **her** sayıyı "kanıtsız" sayar. Gerçek bir model cevabı birçok sayı içerir; D ve E bile bu yüzden anlamsız biçimde başarısız görünürdü. Kanıttaki sayıların vakaya bağlanması gerekir.
2. **Araç adları uyuşmuyor.** Deney kataloğu `rank_sector_laggards`, `evaluate_laggard_catchup`, `align_fundamental_disclosures` gibi adlar kullanır; gerçek yönlendirici (`SafeProjectToolRouter`) `rank_sector_relative_strength`, `calculate_fundamental_change`, `get_macro_context` adlarını kullanır. Eşleme olmadan gerçek çağrılar "geçersiz araç" sayılır.
3. **Tarih uyuşmuyor.** Vakalar 2025-09-30 tarihlidir; projenin analiz tarihi 2026-09-10'dur.
4. **Veri tuzakları sentetik veri ister.** Eksik veri, tekrarlayan tarih ve kısa geçmiş vakaları bozuk veri bekler; yerel önbellekteki gerçek veriler temizdir. Gerçek kalite kapısı bu vakalarda engellemez ve deney bunu "kapı atlandı" diye puanlar. Her tuzak için bozuk bir veri kopyası üretilmelidir.
5. **İnsan onayı (sürüm E) otomatikleştirilemez.** İnsan onayı alanını doğru yapmak onayı uydurmak olur.

Bu beş engel kapatılmadan alınan "gerçek model A–E" sayıları yanıltıcı olurdu.

## 4. Gerçek model için elimizdeki ölçüm

Aynı altı ölçütü gerçek yerel model üzerinde ölçen ayrı bir değerlendirme var (`python -m src.llm.evaluate`, 12 soru, her biri 2 kez, Colab'da CUDA; ayrıntı `roadmap/TASK_5_RAPOR_TASLAGI.md`). Bu **A–E değil, base ve ince ayarlı adaptör karşılaştırmasıdır**; her ikisi de araçlı harness içinde çalışır:

| Metrik | Base | Adaptör |
|---|---|---|
| Geçersiz araç çağrısı | %66,7 | %0 |
| Yanlış plan | %75,0 | %8,3 |
| Sayısal halüsinasyon (ham taslak) | %0 | %33,3 |
| Desteklenmeyen sonuç | %25,0 | %0 |
| Look-ahead | %16,7 | %0 |
| Kanıt eksikliği | %25,0 | %0 |
| Kural ihlali (tavsiye) | %0 | %0 |
| Guard'ın taslağı değiştirdiği oran | %25,0 | %33,3 |

Neyi gösterir: ince ayar araç planını düzeltir; ham taslakta kalan sayısal sapmaları guard yakalayıp deterministik kanıt özetiyle değiştirir. Neyi **göstermez**: kontrol katmanlarının (araç, durum makinesi, kapı, kanıt, replay) tek tek katkısı. Bu, A–E'nin asıl sorusudur.

## 5. A–E'yi gerçek modelle çalıştırmak için yapılması gerekenler

1. Her vaka için kanıttaki izinli sayıları (`known_numeric_values`) gerçek araç çıktısından otomatik doldurmak.
2. Deney araç adlarını gerçek yönlendirici adlarına eşlemek ve vaka tarihlerini 2026-09-10'a taşımak.
3. Eksik veri, tekrarlayan tarih ve kısa geçmiş vakaları için bozuk veri kopyaları üretip bunları gerçek kalite kapısına vermek.
4. Yerel model için bir `ExperimentProvider` yazmak (`providers.py` arayüzü hazır): A modelsiz araçsız, B araçlı, C + durum makinesi, D + kapı ve kanıt, E + replay.
5. Sürüm E'nin insan onayını açık biçimde "bekliyor" bırakmak ya da gerçek bir onay adımı eklemek.

Her çalıştırma ~20–60 sn sürer (12 vaka × 5 sürüm ≈ 20–60 dk); donanım engeli değildir, tasarım işidir.
