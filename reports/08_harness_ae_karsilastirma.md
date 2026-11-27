# 08 – Ajan Harness Deneyi (A–E) Karşılaştırması

> **Önemli açıklama:** Bu raporda iki ayrı tablo var. **§4.1 gerçek modelle** (Qwen3-4B + adaptör, Colab, 15 soru) alınmış A–E ölçümüdür ve asıl sonuçtur. **§2 simüle edilmiş bir sağlayıcıdan** gelir; yalnızca deney hattının çalıştığını gösteren bir boru hattı testidir ve model kalitesini ölçmez.

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

## 3. Simüle deney gerçek modelle olduğu gibi çalıştırılamıyordu (engeller §5'te kapatıldı)

1. **Sayısal doğruluk ölçütü gerçek çıktıya uymuyor.** 12 vakanın 11'inde `known_numeric_values` boştur. Değerlendirici, cevapta geçen **her** sayıyı "kanıtsız" sayar. Gerçek bir model cevabı birçok sayı içerir; D ve E bile bu yüzden anlamsız biçimde başarısız görünürdü. Kanıttaki sayıların vakaya bağlanması gerekir.
2. **Araç adları uyuşmuyor.** Deney kataloğu `rank_sector_laggards`, `evaluate_laggard_catchup`, `align_fundamental_disclosures` gibi adlar kullanır; gerçek yönlendirici (`SafeProjectToolRouter`) `rank_sector_relative_strength`, `calculate_fundamental_change`, `get_macro_context` adlarını kullanır. Eşleme olmadan gerçek çağrılar "geçersiz araç" sayılır.
3. **Tarih uyuşmuyor.** Vakalar 2025-09-30 tarihlidir; projenin analiz tarihi 2026-09-10'dur.
4. **Veri tuzakları sentetik veri ister.** Eksik veri, tekrarlayan tarih ve kısa geçmiş vakaları bozuk veri bekler; yerel önbellekteki gerçek veriler temizdir. Gerçek kalite kapısı bu vakalarda engellemez ve deney bunu "kapı atlandı" diye puanlar. Her tuzak için bozuk bir veri kopyası üretilmelidir.
5. **İnsan onayı (sürüm E) otomatikleştirilemez.** İnsan onayı alanını doğru yapmak onayı uydurmak olur.

Bu beş engel kapatılmadan alınan "gerçek model A–E" sayıları yanıltıcı olurdu; bu yüzden §4.1 ölçümü engeller kapatıldıktan sonra alındı.

## 4. Gerçek model için ölçüm

### 4.1 Gerçek A–E koşusu (Colab, Qwen3-4B + adaptör)

Simüle edilmiş tablonun yerine, aynı 15 soruyu (12 normal, 3 bozuk veri) gerçek model ile beş sürümde çalıştıran ölçüm geçti. Her soru iki kez soruldu. Ölçüm Colab'da (A100) yeni guard ile alındı (27.11.2026). Sonuç: `reports/evidence/eval_ae_colab_newguard.json` (adaptör SHA-256 `4d62e052…cf96c4d5c`). Eski guard ile alınan önceki ölçüm `reports/evidence/eval_ae_colab.json` içinde duruyor. Komut: `python -m src.llm.evaluate --ae --adapter <dizin>`.

| Ölçüt | A | B | C | D | E |
|---|---|---|---|---|---|
| Yanlış plan | 86,7% | 6,7% | 6,7% | 6,7% | 6,7% |
| Sayısal halüsinasyon (kullanıcıya giden cevap) | 0% | 6,7% | 13,3% | 0% | 0% |
| Desteklenmeyen sonuç | 6,7% | 0% | 0% | 0% | 0% |
| Kural ihlali (tavsiye) | 13,3% | 13,3% | 0% | 0% | 0% |
| Kanıt eksikliği | 0% | 0% | 6,7% | 0% | 0% |
| Guard'ın cevabı şablonla değiştirdiği | 0% | 0% | 0% | 13,3% | 13,3% |
| Kalite kapısı atlandı (bozuk veri) | 20% | 20% | 20% | 0% | 0% |
| Geçersiz araç çağrısı, ileri bakış, çökme, tekrarlanabilirlik hatası | 0% | 0% | 0% | 0% | 0% |

Okuma:
- **Kalite kapısı (D):** Üç bozuk veri sorusunun üçünü de D ve E `ANALYSIS_UNSAFE` ile durdurdu. B ve C bozuk veriyi analiz etti; tekrarlayan tarihli ARCLK sorusunda ikisi de `WATCH` verdi.
- **Guard (D'ye ekleme):** Guard önce taslağı bulgularla birlikte modele geri gönderiyor ve modelin kendisine bir kez yeniden yazdırıyor. Şablon yalnızca yeniden yazım da kuralı ihlal ederse devreye giriyor ve `answer_source: harness_template` ile işaretleniyor. D ve E'de bu, soruların %13,3'ünde oldu; eski guard ile aynı oran %26,7'ydi. Guard'ı olmayan B ve C'de kanıtta olmayan sayılar (%6,7 ve %13,3) ve tavsiye dili (B'de %13,3) kullanıcıya ulaştı.
- **A'nın yanlış planı:** A'da araç planı yok; bu bir hata değil, tasarım gereği. A için kalite kapısı ölçütü de anlamsız.
- **Sınır:** 15 soru ve her sürüm için tek ölçüm. Sonuçlar eğilim gösterir, kanıt değildir. Ayrıca A–E sırasında model yalnızca adaptörle çalıştı; base model A–E'de ölçülmedi.

### 4.2 Base ve adaptör karşılaştırması (`09` §5)

Önceki base-adaptör ölçümü ayrı bir değerlendirmedir (12 soru, araçlı harness içinde, ham taslak üzerinden). Colab ve Mac sonuçları `09` §5'te; Colab'ın sonuçları `reports/evidence/eval_report_colab.json` içinde.

## 5. Yapılanlar

Önceki sürümde listelenen beş engel kapatıldı:

1. Kanıttaki izinli sayılar: gerçek araç çıktısından otomatik alınıyor (`evaluate.py`, `unsupported_numbers`).
2. Araç adları: gerçek yönlendiricinin adları kullanılıyor; sorular 2026-09-10 analiz tarihine göre.
3. Bozuk veri: üç vaka (KCHOL eksik veri, ARCLK tekrarlayan tarih, PGSUS kısa geçmiş) bellekte bozulup gerçek kalite kapısına veriliyor; diskteki dosyalar değişmiyor.
4. Yerel model sürümleri: `run_agent(..., version="A".."E")`. E varsayılandır ve bugünkü ajan ile aynıdır.
5. İnsan onayı: E'de `PENDING` bırakılıyor ve "otomatik doldurulmadı" diye kaydediliyor. Gerçek onay yapılmadı.

Bu sayfadaki eski simüle tablo (§2) yalnızca deney hattının çalıştığını gösteren bir **boru hattı testi**dir; model kalitesi ölçmez.
