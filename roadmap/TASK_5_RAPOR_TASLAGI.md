# Task 5 Rapor Taslağı (hocanın istediği 6 madde)

Hocanın istediği 6 madde, güncel adaptörün (r=32) gerçek sonuçlarıyla doldurulmuştur. Ölçümlerin kaynağı `artifacts/llm/eval_report.json` (Colab) ve `reports/test_sonuclari.md` (Mac).

1. **Temel model ve gerekçe.** Qwen3-[1.7B | 4B]. Gerekçe: 1–8B aralığında, T4 üzerinde QLoRA ile eğitilebilir, chat şablonu ve JSON/araç çıktısı için uygun, açık lisans. Seçilen model: Qwen/Qwen3-4B. (1.7B denenmedi; 4B seçildi çünkü araç planı ve kanıta bağlı özet için daha fazla kapasite beklendi, bu varsayım 1.7B ile ölçülerek doğrulanmadı.)
2. **Veri.** Dört zorunlu senaryodan (sektör geride kalanlar, gün/çok günlük desenler, teknik dönüşler, çeyreklik temel + fiyat tepkisi) ve makro/tavsiye güvenlik sorularından üretilmiş kayıtlar (`python -m src.llm.dataset`): soru -> JSON araç planı, soru + kanıt -> kanıta bağlı İngilizce özet, görülmemiş veri/bilinmeyen hisse için ret.
   - Toplam kayıt (Colab'da `python -m src.llm.dataset --output-dir data/llm/dataset_v3 --per-family 120 --reports-per-family 90`): 1230 (araç planı 660, kanıt raporu 510, ret 60); manifest: `data/llm/dataset_v3/manifest.json`.
   - Hedefler gerçek araç çıktılarından ve `summaries.py` şablonlarından deterministik üretilir; modelin ürettiği metin eğitim hedefi değildir. İnsan incelemesi: **yapılmadı**; 1230 kaydın hepsinde `human_reviewed=false` (satır bazında insan incelemesi yapıldığına dair kayıt yok).
   - 6 hisse eğitimden dışlandı (`HOLDOUT_TICKERS`), değerlendirme bu hisselerle de yapılır. Train/validation: case_id bazlı bölme, sızıntı yok.
3. **Eğitim yöntemi.** QLoRA (4-bit NF4), LoRA r=32, alpha=64, dropout=0.05; 4 epoch, lr 2e-4, etkin batch 8, max_length 4096, yalnızca cevap üzerinde kayıp (`completion_only_loss`), seed 481. Adaptör SHA-256: `4d62e05230eeb7e5855cadf3e7274e1c8b621df4b707de12c12cef5cf96c4d5c` (`eval_report.json`).
4. **Yerel çalıştırma.** `BIST_LLM_OFFLINE=1 python -m src.llm.agent --adapter <dizin>` (analiz tarihi varsayılan 2026-09-10) (Hugging Face Transformers) veya birleştirilmiş model + GGUF + Ollama (localhost). Hiçbir bulut API'si çağrılmaz. Test edilen makine: Apple M5 (MPS), HF yolu. Hızlı yol (isteğe bağlı): birleştirilmiş model Ollama ile `int8` nicemlenir (`ollama create bist-qwen -q int8 -f Modelfile`; Qwen3 için boş `<think>` bloğu `ollama_chat_fn` içinde temizlenir). 5 örnek soruda soru başına 17–23 sn (HF: 31–57 sn); 5 sorunun 4'ünde plan ve ham cevap HF ile birebir aynı, Q4'te Ollama yalnızca `calculate_fundamental_change` çağırdı (HF iki araç çağırıyor). Resmi ölçüm HF yoludur; Ollama yolu 12 soruluk değerlendirmeyle ölçülmedi.
5. **Base vs fine-tune.** `python -m src.llm.evaluate --adapter <dizin>` çıktısındaki tablo (12 soru, 6'sı eğitimde hiç görülmemiş hisselerle; her soru iki kez sorulur):

   | Metrik | Base | Adaptör |
   |---|---|---|
   | Geçersiz araç çağrısı | 0.667 | 0.000 |
   | Yanlış plan (araç/hisse/tarih aralığı) | 0.750 | 0.083 |
   | Sayısal halüsinasyon | 0.000 | 0.333 |
   | Desteklenmeyen sonuç | 0.250 | 0.000 |
   | Look-ahead | 0.167 | 0.000 |
   | Kanıt eksikliği | 0.250 | 0.000 |
   | Kural ihlali (tavsiye) | 0.000 | 0.000 |
   | Tekrarlanabilirlik hatası | 0.000 | 0.000 |
   | Guard'ın cevabı değiştirdiği oran | 0.250 | 0.333 |

   Not: metrikler modelin ham çıktısı üzerinden hesaplanır; guard sonucu gizlemez.
   Ölçüm: Qwen/Qwen3-4B + adaptör (SHA-256 `4d62e052...cf96c4d5c`), Colab'da CUDA GPU üzerinde (`src/llm/infer.py` CUDA'da 4-bit NF4 yükler), 12 soru, her biri iki kez; ham sonuçlar `artifacts/llm/eval_report.json`. Aynı adaptörle Apple M5'te (MPS, bf16) 5 örnek soru iki kez soruldu; planlar ve cevaplar iki turda birebir aynı çıktı.

   **Yorum (dürüst):** Fine-tuning araç planını düzeltti: geçersiz çağrı %66,7'den %0'a, yanlış plan %75'ten %8,3'e (12 sorunun 1'i: sektör sorusunda `sector` argümanıyla plan), desteklenmeyen sonuç, look-ahead ve kanıt eksikliği %0'a indi; tavsiye kuralı ihlali her iki modelde %0. Dikkat: base 12 sorunun 8'inde geçerli plan üretemediği için özet aşamasına en fazla 4 soruda ulaşıyor, bu yüzden base'in %0 sayısal halüsinasyonu adaptörünkiyle adil bir kıyas değil. Adaptörde ham metin 4/12 cevapta kanıttan sapıyor (%33,3; sorular 1, 2, 6, 10); sapmalar çoğunlukla şablon metnini kopyalarken ±0,1 puanlık yuvarlama farkı (örn. Mac'te ASELS sorusunda %1,5 yerine %1,4 ve %67,7 yerine %67,6). Guard bunları yakalayıp deterministik taslakla değiştirdiği için kullanıcıya giden cevaplar kanıtla uyumlu; ancak guard devreye girdiğinde son cevap modelin değil şablonun metnidir (%33). Önceki adaptöre (r=16) göre iyileşme: sayısal halüsinasyon %50→%33, kural ihlali %25→%0, guard devreye girme %58→%33.
6. **Bilinen sınırlamalar.** (a) Sayı denetimi 0–20 arası tam sayıları serbest bırakır. (b) Temel veriler yalnızca 5 hisse için var, elle derlenmiş (curated) ve KAP kaynak kimliği doğrulanmamıştır; diğerlerinde model INVESTIGATE demelidir. (c) TCMB faizi, TÜİK TÜFE ve Fed faizi CSV dosyaları repoda var, ancak kaynakları ve birimleri doğrulanmadı (TÜFE dosyası yıllık oran değil bir endeks düzeyi gibi görünüyor, TCMB dosyası yalnızca 11 satır); USD/TRY ve EUR/TRY Yahoo'dan gelir. Haber katmanı uygulandı ama haber dosyası boş. (d) Sektör denemesinde Banking sektöründe evrende tek hisse (AKBNK) var; karşılaştırma için en az 2 hisse gerekir. (e) Küçük model Türkçe soruyu zayıf anlayabilir; eğitim İngilizce. (f) Değerlendirme seti 12 sorudur, istatistiksel güven sınırlıdır. (g) Guard devredeyken son cevap deterministik taslaktır, modelin değil. (h) Resmi değerlendirme Colab'da 4-bit yüklemeyle yapıldı; Mac'te bf16 ile 5 örnek soru denendi ama 12 soruluk set orada çalıştırılmadı, iki ortamın küçük sayısal farkları olabilir. (i) Ollama int8 yolu 12 soruluk değerlendirmeyle ölçülmedi; hız için isteğe bağlıdır.
