# Task 5 Rapor Taslağı (hocanın istediği 6 madde)

Boşları gerçek sonuçlarla doldurun; `[ ]` işaretli yerler eğitimden sonra gelir.

1. **Temel model ve gerekçe.** Qwen3-[1.7B | 4B]. Gerekçe: 1–8B aralığında, T4 üzerinde QLoRA ile eğitilebilir, chat şablonu ve JSON/araç çıktısı için uygun, açık lisans. Seçilen model: [ ]. (Neden 1.7B değil/evet 4B: [ ])
2. **Veri.** Dört zorunlu senaryodan (sektör geride kalanlar, gün/çok günlük desenler, teknik dönüşler, çeyreklik temel + fiyat tepkisi) ve makro/tavsiye güvenlik sorularından üretilmiş kayıtlar (`python -m src.llm.dataset`): soru -> JSON araç planı, soru + kanıt -> kanıta bağlı İngilizce özet, görülmemiş veri/bilinmeyen hisse için ret.
   - Toplam kayıt (v4 üretimi, `--per-family 120 --reports-per-family 90`): 1230 (araç planı 660, kanıt raporu 510, ret 60); manifest: `data/llm/dataset_v3/manifest.json`.
   - Hedefler gerçek araç çıktılarından ve `summaries.py` şablonlarından deterministik üretilir; modelin ürettiği metin eğitim hedefi değildir. İnsan incelemesi: [yapıldı/yapılmadı, kim].
   - 6 hisse eğitimden dışlandı (`HOLDOUT_TICKERS`), değerlendirme bu hisselerle de yapılır. Train/validation: case_id bazlı bölme, sızıntı yok.
3. **Eğitim yöntemi.** QLoRA (4-bit NF4), LoRA r=32, alpha=64, dropout=0.05; 4 epoch, lr 2e-4, etkin batch 8, max_length 4096, yalnızca cevap üzerinde kayıp (`completion_only_loss`), seed 481. Adaptör SHA-256: [ ] (`eval_report.json`).
4. **Yerel çalıştırma.** `BIST_LLM_OFFLINE=1 python -m src.llm.agent --adapter <dizin>` (analiz tarihi varsayılan 2026-09-10) (Hugging Face Transformers) veya birleştirilmiş model + GGUF + Ollama (localhost). Hiçbir bulut API'si çağrılmaz. Test edilen makine: [ ].
5. **Base vs fine-tune.** `python -m src.llm.evaluate --adapter <dizin>` çıktısındaki tablo (12 soru, 6'sı eğitimde hiç görülmemiş hisselerle; her soru iki kez sorulur):

   | Metrik | Base | Adaptör |
   |---|---|---|
   | Geçersiz araç çağrısı | 0.583 | 0.000 |
   | Yanlış plan (araç/hisse/tarih aralığı) | 0.583 | 0.000 |
   | Sayısal halüsinasyon | 0.000 | 0.500 |
   | Desteklenmeyen sonuç | 0.417 | 0.000 |
   | Look-ahead | 0.167 | 0.000 |
   | Kanıt eksikliği | 0.417 | 0.000 |
   | Kural ihlali (tavsiye) | 0.000 | 0.250 |
   | Tekrarlanabilirlik hatası | 0.000 | 0.000 |
   | Guard'ın cevabı değiştirdiği oran | 0.417 | 0.583 |

   Not: metrikler modelin ham çıktısı üzerinden hesaplanır; guard sonucu gizlemez.
   Ölçüm: Qwen/Qwen3-4B + adaptör (SHA-256 `f361cd0e...bdf6f7e`), Apple Silicon (MPS), 12 soru, her biri iki kez; ham sonuçlar `artifacts/llm/eval_report.json`.

   **Yorum (dürüst):** Fine-tuning araç planını düzeltti (geçersiz çağrı, yanlış plan, look-ahead, kanıt eksikliği ve desteklenmeyen sonuç %17–58'den %0'a). Ama adaptörün ham metni base'e göre daha fazla sayısal sapma içeriyor (%0 → %50) ve 3/12 cevapta tavsiye reddini atlıyor veya tavsiye kuralını ihlal ediyor (%0 → %25). Sapmalar çoğunlukla şablon metnini kopyalarken ±0,1 puanlık yuvarlama farkı (örn. %45,9 yerine %45,8) ve bir sayının yanlış etikete taşınması biçiminde. Guard bunları yakalayıp deterministik taslakla değiştirdiği için kullanıcıya giden son cevaplar kanıtla uyumlu; ancak guard devreye girme oranı %58 olduğundan son cevabın çoğu modelin değil şablonun metnidir.
6. **Bilinen sınırlamalar.** (a) Sayı denetimi 0–20 arası tam sayıları serbest bırakır. (b) Temel veriler yalnızca 5 hisse için var, elle derlenmiş (curated) ve KAP kaynak kimliği doğrulanmamıştır; diğerlerinde model INVESTIGATE demelidir. (c) TCMB faizi, TÜİK TÜFE ve Fed faizi resmi CSV dosyası verilmedikçe kullanılamaz; yalnızca USD/TRY ve EUR/TRY Yahoo'dan gelir. (d) Sektör denemesinde Banking sektöründe evrende tek hisse (AKBNK) var; karşılaştırma için en az 2 hisse gerekir. (d) Küçük model Türkçe soruyu zayıf anlayabilir; eğitim İngilizce. (e) Değerlendirme seti 12 sorudur, istatistiksel güven sınırlıdır. (f) Guard devredeyken son cevap deterministik taslaktır, modelin değil.
