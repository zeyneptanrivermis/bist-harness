# BIST 100 Entegre Ajan Tabanlı Finansal Analiz ve Doğrulama Sistemi: Nihai Teknik Rapor

CSE-481 Mühendislik Ekonomisi · Analiz tarihi **2026-09-10** (bu tarihten sonraki hiçbir veri kullanılmaz)

> Eğitsel araştırma çalışmasıdır. Yatırım tavsiyesi içermez, aracı kurumla bağlantısı yoktur. Geçmiş performans gelecek için garanti değildir. Bu rapordaki her sayı, `reports/evidence/` altındaki araç çıktılarından ya da atıf yapılan alt raporlardan gelir ve komutla yeniden üretilebilir (bkz. Ek B).

## 1. Giriş

Büyük dil modelleri finansal sorularda akıcı ama kanıtsız cevaplar üretebilir: uydurma sayılar, bilinmeyen tarihlerin kullanılması, "kesin yükselir" gibi iddialar. Bu projenin tezi şudur: **dil modeli hesap yapmaz, hesabı deterministik araçlar yapar; model yalnızca bir araç planı seçer ve çıkan kanıtı özetler; bir harness sayıların, tarihlerin ve üslubun kanıta uymasını denetler.** Model yerel çalışır, hiçbir bulut yapay zekâ API'si çağrılmaz.

Bu raporun cevapladığı sorular:
1. Dört zorunlu araştırma senaryosu (sektörde geride kalanlar, hafta içi/çok günlü kalıplar, teknik dönüşler, bilanço ve fiyat tepkisi) veriyle ne söylüyor?
2. Harness (kalite kapısı, durum makinesi, kanıt, replay) ve yerel ince ayarlı model, hocanın altı ölçütünde (sayısal halüsinasyon, geçersiz araç çağrısı, desteklenmeyen sonuç, kanıt eksikliği ve tekrarlanabilirlik, kural ihlali, ileri bakış) nasıl davranıyor?
3. Neyi başaramadık ve neden?

**Kapsam sınırı:** haber ve konuşma katmanları yalnızca zaman damgalı metadata olarak uygulandı (`get_news_events`, `get_speech_features`, `transcribe_finance_video_segment`); iki veri dosyası da boş. Konuşma aracı harness'in isteğe bağlı `LOAD_CONTEXT` adımında izinlidir ama yerel model plan örneklerinde görmediği için onu çağırmaz.

## 2. Sistem mimarisi

```
soru ──▶ yerel model (Qwen3-4B + LoRA) ──▶ JSON araç planı
                                              │
                         SafeProjectToolRouter doğrular (araç adı, evrendeki hisse, tarih ≤ 2026-09-10)
                                              │
        kalite kapısı ─▶ araçlar (indikatör, senaryo, makro, backtest) ─▶ kanıt kümesi (E1, E2, E3…)
                                              │
                 model kanıttan özet yazar ──▶ guard (check_answer)
                                              │         sayı, tarih, üslup, tavsiye, kanıt kimliği denetimi
      ihlal varsa model bulgularla bir kez yeniden yazar; yine ihlal varsa
      deterministik özet "harness özeti" etiketiyle gösterilir
                                              │
                          karar günlüğü (SHA-256) ──▶ replay
```

Bileşenler:
- **Veri katmanı** (`src/data_pipeline.py`, `src/fundamentals.py`, `src/macro.py`, `src/context/news.py`): fiyat, çeyreklik temel veri, makro seriler, haber metadata'sı. Her veri kümesinin kaynağı ve yayın tarihi kuralı `reports/01_veri_sozlugu.md` içindedir.
- **Deterministik araçlar** (`src/mcp_tools/`, `src/scenarios/`, `src/indicators.py`, `src/backtest/`): hesabı yalnızca bunlar yapar. Sinyaller t+1'de uygulanır, her işlemde %0,10 komisyon ve %0,05 kayma uygulanır.
- **Harness** (`src/harness/`): zorunlu durum makinesi (`START → SELECT_UNIVERSE → LOAD_DATA → CHECK_DATA_QUALITY → CALCULATE → RUN_SCENARIO → [LOAD_FUNDAMENTALS] → [LOAD_MACRO_CONTEXT] → [LOAD_CONTEXT (haber)] → PREVIEW_STRATEGIES → RUN_BACKTEST → RISK_GATE → BUILD_EVIDENCE → GENERATE_EXPLANATION → [HUMAN_REVIEW] → LOG_AND_CLOSE`; her durum yalnızca izinli araçları çağırabilir (`src/harness/permissions.py`). Şartnamedeki sıradan tek fark: kanıt, backtest sonucuna bağlı olduğu için `BUILD_EVIDENCE` `RISK_GATE`'ten sonra gelir), veri kalitesi kapısı (`PASS`, `PASS_WITH_WARNINGS`, `HARD_BLOCK`, `ANALYSIS_UNSAFE`), kanıt kümesi ve karar günlüğü.
- **Yerel model** (`src/llm/`): Qwen/Qwen3-4B + LoRA (QLoRA, 4-bit NF4; r=32, alpha=64, dropout 0,05, 4 epoch, lr 2e-4, etkin batch 8, max_length 4096, seed 481, yalnızca cevap üzerinde kayıp, 1230 kayıt: araç planı 660, kanıt raporu 510, ret 60). **Neden Qwen3-4B:** hocanın önerdiği 1–8B aralığında, tek T4 GPU'da QLoRA ile eğitilebiliyor (teslim edilen adaptör ise bf16 ile, yani T4'ten yeni bir Colab GPU'sunda eğitildi: `training_args.bin` `bf16=true`; notebook bf16'yı yalnızca compute capability >= 8 olan GPU'da seçer), açık lisanslı, chat şablonu ve JSON/araç çıktısı için uygun; 4B, araç planı ve kanıta bağlı özet için 1,7B'den fazla kapasite beklendiği için seçildi (1,7B ile karşılaştırma yapılmadı). Eğitim hedefleri araç çıktılarından deterministik şablonla üretilir, modelin ürettiği metin eğitim hedefi değildir. 6 hisse eğitimden dışlanmıştır (ENKAI, FROTO, ISMEN, MPARK, OYAKC, TCELL). Çıkarım Hugging Face Transformers ile yereldedir (Apple M5 MPS ya da CUDA); isteğe bağlı Ollama (int8) yolu vardır.
- **Guard** (`check_answer`): modelin taslağında kanıtta olmayan sayı, analiz tarihinden sonraki tarih, kanıt kimliği eksikliği, doğrudan al/sat tavsiyesi ya da tavsiye isteğine ret eksikliği bulursa ihlal listesini (ör. "şu sayılar kanıtta yok: 77.7") modele geri verir ve model taslağı **bir kez yeniden yazar**. Yeniden yazım da kuralları geçemezse deterministik özet gösterilir, ama modelin cevabı gibi sunulmaz: her cevapta `answer_source` alanı cevabı kimin yazdığını söyler (`model`, `model_retry`, `harness_template`, `harness_refusal`) ve arayüz bunu cevabın üstünde yazar. Ham taslak ve yeniden yazım saklanır, guard sonucu gizlemez. Sayı denetimi her sayıyı kapsar; yalnızca kanıt kimlikleri, ISO tarihler, XU100 ve "t+1" ifadesi denetim dışıdır. Sayı ufukları ("5 günlük") kanıttaki alan adlarından (`5d`, `ret_20d`) tanınır.
- **Karar günlüğü ve replay** (`reports/06_karar_gunlugu_ve_replay.md`): her analiz SHA-256'lı JSON olarak yazılır; tek karakter değişince replay reddeder. Günlük artık modelin ham taslağını ve guard ihlallerini de saklar.

## 3. Veri

30 hisselik sabit evren (24 sektör etiketi, yalnızca 6'sında 2 hisse), günlük fiyatlar 2024-01-02'den başlar (önbellekte 689 işlem günü; analizde yalnızca 2026-09-10'a kadar olanı kullanılır), XU100 endeksi, makro seriler (USD/TRY, EUR/TRY, TCMB faizi, TÜFE, Fed faizi), 5 hisse için çeyreklik temel veri, boş haber dosyası. Ayrıntılı sütun tanımları ve yayın tarihi kuralları: `reports/01_veri_sozlugu.md`.

**RAG kullanılmıyor.** Model belge aramaz ve belgeden bilgi çekmez. Yalnızca hangi deterministik araçların çağrılacağını planlar ve araçların ürettiği kanıtı (E1, E2, ...) özetler. Her sayı bir araçtan gelir, kanıtta olmayan sayıyı guard reddeder. Bu, kanıtsız sayı riskini azaltmak için bilinçli bir tasarım. Şartnamedeki bellek ayrımı (§24) şöyle karşılanıyor:
- **Temporal:** fiyat, makro ve temel veri dosyaları.
- **Episodic:** karar günlükleri.
- **Procedural:** sistem istemi, araç tanımları ve durum-araç izinleri.

Teslimdeki "RAG dosyası" yerine `reports/rag_knowledge_base.json` verilir. Dosya bu kaynakları kaynak adı ve adresi, yayın tarihi kuralı, kapsam (satır, tarih aralığı) ve SHA-256 ile listeler. Modele verilen istemleri ve araç tanımlarını da birebir içerir. Dosyayı `python scripts/make_rag_file.py` koddan üretir; içine elle sayı yazılmaz.

Veri hakkında doğrulanmamış noktalar (alt raporlarda yinelenir):
- Temel veri elle girilmiş, KAP kaynak kimliği yok, yalnızca 5 hisse. Tüm temel veri sayıları "örnek" olarak etiketlenmiştir.
- TÜFE dosyası yıllık oran değil bir endeks düzeyi gibi görünüyor; TCMB dosyası yalnızca 11 satır. Kaynak ve indirme tarihi kayıtlı değil.
- Geliştirme önbelleği 2026-09-30'a kadar uzanıyordu ve yalnızca araçlar analiz tarihinde kesiyordu; bu bir ileri bakış hatasına yol açtı ve düzeltildi (bkz. §4.2). Teslim edilen `bist_data.zip` 2026-09-10'da kesilmiştir; `reports/evidence/` bu kesilmiş veriyle birebir aynı çıkar.

## 4. Araştırma bulguları

| Senaryo | Rapor | Ana sonuç | Sistem durumu |
|---|---|---|---|
| Sektörde geride kalanlar | `03` | Perakende grubunda MAVI en geride (%−3,25); geçmiş benzer gerilikler 10 günde %51,8 kapandı (494 olay, 53 bağımsız dönem). Yakalama hipotezi desteklenmiyor. | `INVESTIGATE` |
| Hafta içi kalıplar | `10` (PDF: ilk taslak) | THYAO'da "Pzt-Sal-Çar düşer, Per-Cum yükselir" örüntüsü desteklenmiyor: örnek içinde 87 haftanın 19'unda görüldü ve bunların yalnızca 6'sında Perşembe ile Cuma birlikte yükseldi; örnek dışında 30 haftanın 3'ünde görüldü, hiçbirinde yükselmedi. Örüntünün bilindiği Çarşamba kapanışında alıp Cuma kapanışında satmanın maliyet sonrası (%0,3 gidiş-dönüş) ortalaması örnek içi %1,1, örnek dışı %−0,2; 32 aday haftalık dizi için Bonferroni α 0,0016 ve örnek içi p 0,259. Kaynak: `reports/evidence/weekday_pattern_thyao.json`. | `REJECT_SIGNAL` |
| Teknik dönüşler | `02` | ASELS'te 34 alt bant teması; 5 günde isabet %67,6 (genel eğilim %59,2). İdeal birleşik koşul hiç oluşmadı (n=0), "RSI dibi + hacim" yalnızca 3 olay. | `TECHNICAL_REVERSAL_CANDIDATE` (küçük örneklem uyarılı) |
| Bilanço ve fiyat tepkisi | `04` | TUPRS'te iyileşen çeyrekler 20 günde daha iyi görünüyor, ama fark tek çeyreğe dayanıyor ve 5 günde ortadan kalkıyor. Tüm sayılar örnek. | `INVESTIGATE` |
| Makro bağlam | `05` | USD/TRY %+67,1; EREGL ile haftalık korelasyon −0,26 (zayıf ve ters). "Kur yükselince ihracatçı yükselir" varsayımı desteklenmiyor. | `INVESTIGATE` |

### 4.1 Backtest ve benchmark (`07`)

A–E strateji varyantları 30 hissede, t+1 zamanlaması ve maliyetlerle çalıştırıldı. **Hiçbir varyant al-tut'u ya da XU100'ü sistematik biçimde geçmedi:** medyan strateji getirisi %12 civarı, medyan al-tut %85,2, XU100 %+88,8; al-tut'u geçen 4/30, XU100'ü geçen 2–3/30. Haber katmanı boş, makro katman hiç sinyal değiştirmedi (USD/TRY'nin 20 günlük en büyük artışı %4,8, eşik %8), bu yüzden D ve E pratikte C ile aynı. Sektör benchmark'ı yalnızca akranı olan 17 hisse için hesaplanır (bkz. `07`).

### 4.2 Bu çalışmada bulunan ve düzeltilen hatalar

- **İleri bakış (Task 1 sektör sayısı):** 746 olay ve %52,4/%53,8/%55,8 isabet, analiz tarihinden sonraki fiyatları içeren önbellekle üretilmişti. Kesilince 743 olay, %52,7/%54,1/%55,9. Belge düzeltildi.
- **Replay günlüğü eksikliği:** günlük yalnızca son cevabı saklıyordu; guard'ın taslağı değiştirip değiştirmediği izlenemiyordu. Ham taslak ve ihlaller eklendi.
- **Ollama çıktısı:** Qwen3'ün boş `<think></think>` bloğu araç planı ayrıştırıcısını bozuyordu; temizlendi.
- **Sektör sorusunda makro kanıt eksikti:** model planı `get_macro_context` çağırmıyordu, bu yüzden yapısal ve geçici gerilik ayrılamıyordu. Harness artık sektör planlarında `LOAD_MACRO_CONTEXT` adımında en geride kalan hisse için makro bağlamı kendisi ekler (zorunlu backtest gibi).
- **Hafta içi kalıbında maliyet ve çoklu test eksikti:** çıktı yalnızca ortalama veriyordu ve "1 test" diyordu. Artık medyanlar, Çarşamba kapanışında alıp Cuma kapanışında satmanın (bileşik) %0,3 gidiş-dönüş maliyeti sonrası getirisi ve 32 olası haftalık yukarı/aşağı dizisi için Bonferroni düzeltmesi raporlanır.

## 5. Yerel model ve harness değerlendirmesi

12 soru (6'sı eğitimde hiç görülmemiş hisselerle), her biri iki kez; metrikler modelin **ham** çıktısı üzerinden hesaplanır. Aynı adaptör (SHA-256 `4d62e052…cf96c4d5c`) iki ortamda ölçüldü: Colab (CUDA, 4-bit NF4, `reports/evidence/eval_report_colab.json`) ve teslim edilen kodla bu Mac (Apple M5, MPS, bf16, `reports/evidence/eval_report_local.json`). Demo Mac yolunu kullanır. Ayrıntı: `roadmap/TASK_5_RAPOR_TASLAGI.md`.

| Metrik | Base (Colab) | Adaptör (Colab) | Base (Mac) | Adaptör (Mac) |
|---|---|---|---|---|
| Geçersiz araç çağrısı | %66,7 | %0 | %58,3 | %16,7 |
| Yanlış plan | %75,0 | %8,3 | %58,3 | %25,0 |
| Sayısal halüsinasyon (ham taslak) | %0 | %33,3 | %0 | %33,3 |
| Desteklenmeyen sonuç | %25,0 | %0 | %41,7 | %0 |
| Look-ahead (plan tarihi) | %16,7 | %0 | %16,7 | %16,7 |
| Kanıt eksikliği | %25,0 | %0 | %41,7 | %0 |
| Kural ihlali (tavsiye) | %0 | %0 | %0 | %0 |
| Tekrarlanabilirlik hatası | %0 | %0 | %0 | %0 |
| Guard'ın taslağı değiştirdiği oran | %25,0 | %33,3 | %41,7 | %33,3 |

Okuma:
- İnce ayar araç planını düzeltti. Base 12 sorunun 8'inde geçerli plan üretemediği için özet aşamasına en fazla 4 soruda ulaşıyor; bu yüzden onun %0 sayısal halüsinasyonu adil bir kıyas değildir.
- Adaptörün ham taslağı 4 cevapta kanıttan sapıyor (çoğunlukla ±0,1 puanlık yuvarlama farkı). Bu ölçümlerde guard bunları yakalayıp deterministik özetle değiştiriyordu; yani kullanıcıya giden cevapların üçte birinde metin modelin değil şablonun. Bu sayılar yeniden yazım ve `answer_source` etiketi eklenmeden önce ölçüldü; yeni guard ile yeniden ölçülmedi.
- **Tekrarlanabilirlik (Mac, HF, bf16, teslim edilen kod):** `scripts/repro_check.py` 5 örnek soruyu ve 3 tuzak soruyu (25.09.2026 tarihi, evrende olmayan TESTX, "kesin yükselir, al, hedef fiyat ver") ikişer kez sordu; 8 sorunun 8'inde plan, kanıt, ham taslak ve son cevap birebir aynı çıktı. Durumlar: Q1 `INVESTIGATE`, Q2 `REJECT_SIGNAL`, Q3 `TECHNICAL_REVERSAL_CANDIDATE`, Q4 `INVESTIGATE`, Q5 `INVESTIGATE`, gelecek tarih ve TESTX `ANALYSIS_UNSAFE`, ISMEN tavsiye isteği `INVESTIGATE`. Q3 ve ISMEN sorusunda guard ham taslağı (kanıtta olmayan sayı) şablonla değiştirdi. Q3'teki "kanıtta olmayan sayılar" aslında tam ortada kalan değerlerin farklı yuvarlanmasıydı: kanıtta 0,0145 ve 0,6765 var, model %1,4 ve %67,6 yazdı (yarımı çifte yuvarlama), guard ise yalnızca %1,5 ve %67,7'yi kabul ediyordu. Guard artık iki yuvarlamayı da kabul ediyor. Bu düzeltmeden sonra beş senaryo yeniden koşuldu (`reports/execution_traces.json`, 26.11.2026). Beşinde de cevabı model yazdı (`answer_source: model`), kural ihlali yok, durumlar yukarıdakiyle aynı. Soru başına 31–57 sn. Ollama int8 yolunda 17–23 sn, 5 sorunun 4'ünde HF ile aynı (`notebooks/README.md`).
- **Colab ve Mac farkı.** 4-bit (Colab) ve bf16 (Mac) sayısal olarak farklı olduğu için greedy çözümleme bazı sorularda başka token seçiyor. Mac'te iki tuzak soruda (25.09.2026 tarihi, evrende olmayan GARAN) model metinle reddetmek yerine bir araç planı yazdı; router planı reddetti ve kullanıcıya `ANALYSIS_UNSAFE` döndü. Bu yüzden Mac'te geçersiz araç çağrısı %16,7 ve look-ahead %16,7 görünüyor: model 2026-09-10 sonrası tarihi istedi, ama hiçbir araç çalışmadı ve analiz tarihinden sonraki veri kullanılmadı. Mac'in yanlış plan oranındaki fark ayrıca perakende sorusundan geliyor: model `tickers` yerine `sector="retail"` yazdı; aynı üç hisseyi seçtiği için sonuç aynı, ama ölçüt birebir argüman beklediği için yanlış sayıldı.
- **Veri anlık görüntüsü.** Colab koşusu daha eski bir fiyat önbelleğiyle yapıldı (ör. perakende 489 olay, ASELS 5 günlük isabet %64,7, THYAO örnek dışı 29 hafta). Mac koşusu teslim edilen `bist_data.zip` ile birebir aynı veriyi kullanır ve rapordaki sayıları (494 olay, %67,6, 30 hafta) verir.
- **Sektör sorularında guard.** Harness artık sektör planlarına makro kanıtı (E3) ekliyor, backtest E4 oldu. Adaptör üç kanıtlı düzende eğitildiği için taslağı E4'ü atlarsa guard `missing_evidence_id` ile şablona geçer; Mac koşusunda perakende taslağı guard'dan geçti, telekom taslağı sayı sapması yüzünden değiştirildi. Yeniden eğitimde `src/llm/dataset.py` aynı `execute_plan` akışını kullandığı için yeni düzen kendiliğinden eğitim verisine girer.
- Önceki adaptöre (r=16) göre sayısal halüsinasyon %50 → %33, kural ihlali %25 → %0, guard devreye girme %58 → %33.

### 5.1 A–E harness deneyi (`08`)

Gerçek model, aynı 15 soruyu (12 normal, 3 bozuk veri) beş kontrol düzeyinde çalıştırdı; her soru iki kez soruldu (Colab). Sonuç `reports/evidence/eval_ae_colab.json` içinde, ayrıntısı `08` §4.1'de. **Sınır:** Bu ölçüm, yeniden yazım adımı ve yuvarlama düzeltmesi eklenmeden önceki guard ile yapıldı. Yeni guard ile A–E yeniden ölçülmedi (Mac'te yaklaşık 1,5 saat sürüyor). Guard'ın cevabı değiştirdiği oran (%26,7), yeni guard ile büyük olasılıkla daha düşük çıkar: beş senaryo koşusunda şablona hiç düşülmedi. Ama bu bir ölçüm değil. Diğer satırlar (kalite kapısı, plan, ileri bakış) guard'dan bağımsızdır.

| Ölçüt | A | B | C | D | E |
|---|---|---|---|---|---|
| Yanlış plan | 86,7% | 6,7% | 6,7% | 6,7% | 6,7% |
| Sayısal halüsinasyon (son cevap) | 0% | 20% | 26,7% | 0% | 0% |
| Kalite kapısı atlandı (bozuk veri) | 20% | 20% | 20% | 0% | 0% |
| Guard'ın cevabı değiştirdiği | 0% | 0% | 0% | 26,7% | 26,7% |

Okuma: kalite kapısı (D) üç bozuk veri sorusunun üçünü de durdurdu; kapı olmadan (B, C) bozuk veri analiz edildi. Guard, D ve E'nin cevaplarının %26,7'sini değiştirdi. A'da araç planı olmadığı için yanlış plan ölçütü anlamsız. 15 soru, sürüm başına tek ölçüm: sonuçlar eğilim gösterir, kanıt değildir. Simüle deney (`08` §2) yalnızca boru hattı testidir.

## 6. Tartışma

**Ne çalışıyor.** Hesabı araçlara, denetimi guard'a bırakmak işe yarıyor: ince ayarlı model geçersiz araç çağrısını Colab'da %66,7'den %0'a, Mac'te %58,3'ten %16,7'ye indirdi (Mac'teki iki hatalı planı router durdurdu), 12 soruluk değerlendirmede tavsiye kuralı ihlali vermedi ve sonuçlar tekrarlanabilir (greedy çözümleme). Sistem "bilmiyorum" demeyi biliyor: küçük örneklem, temel verisi olmayan hisse, boş haber dosyası ve tavsiye istekleri `INVESTIGATE` ya da uyarıyla sonuçlanıyor.

**Ne çalışmıyor ya da kanıtlanmadı.** Dört senaryodan hiçbiri sağlam bir işlem sinyali bulmadı: perakende grubunda yakalama %46–52, hafta içi örüntüsü reddedildi, teknik dönüşün birleşik koşulları anekdot, bilanço tepkisi 8 çeyrekte ayırt edilemiyor. Stratejiler al-tut'u geçemedi. Bu bir başarısızlık değil, dürüst bulgudur: 2024–2026'da güçlü yükselen bir piyasada basit kurallar al-tut'un gerisinde kalıyor ve guard'ın amacı tam olarak bu sonuçların fazla iddialı sunulmamasını sağlamak.

**Sınırlamalar (özet):**
- Evren 30 hisse ve 24 sektör etiketinde yalnızca 6'sında 2 hisse var; sektör karşılaştırması sınırlı; sektör benchmark'ı yalnızca 17 hisse için var ve ilk kelime eşleşmesine dayanır.
- Temel veri elle girilmiş ve 5 hisselik; haber ve konuşma verisi yok (araçlar var, dosyalar boş); TÜFE ve TCMB dosyalarının kaynak sitesi kayıtlı (`01` §4) ama sorgu ve indirme tarihi kayıtlı değil.
- Teknik dönüşte yeni olay aileleri ve volatilite rejimi yalnızca ASELS için raporlandı (`02` §6); küçük hücreler anekdottur ve çoklu test düzeltmesi yoktur. Yerel model bu ailelerle yeniden eğitilmedi.
- Değerlendirme seti 12 soru; istatistiksel güven sınırlı. §5'teki guard oranları, modelin yeniden yazımı ve sıkı sayı denetimi (0–20 arası sayılar da denetlenir) eklenmeden önce ölçüldü.
- Colab ölçümü 4-bit, Mac çalıştırması bf16; iki ortamın küçük sayısal farkları olabilir.
- İz kaydı (`trace`) kodda ve web arayüzünde var, ama repodaki örnek günlük (`example_analysis_log.json`) bu özellikten önce üretildi ve iz kaydı içermez; Ollama yolunda ağırlık özeti kaydedilmez (`06` §1).
- Replay karması kaydın yanında saklandığı için kötü niyetli yeniden yazımı değil, yanlışlıkla bozulmayı yakalar.
- A–E harness deneyi gerçek modelle yalnızca Colab'da (4-bit) yapıldı; 15 soru, sürüm başına tek ölçüm, Mac'te tekrarlanmadı ve base model A–E'de ölçülmedi (§5.1).

## 7. Sonuç

Sistem, "kanıtsız sayı üretme" riskini mimariyle sınırlıyor: deterministik araçlar, zorunlu kalite kapısı, kanıt kimlikleri, guard ve değiştirilemez günlük. Yerel ince ayarlı model araç planlamayı güvenilir yapıyor; metin kalitesindeki kalan sapmaları guard telafi ediyor. Araştırma bulguları ise sistemin kendi vokabülerine uygun biçimde temkinli: dört senaryoda da işlem sinyali olarak savunulabilir bir sonuç çıkmadı. Eksikler (gerçek haber, konuşma ve temel veri; A–E'nin Mac'te tekrarı) §6'da açıkça listelidir.

## Kaynaklar

1. CSE-481 Mühendislik Ekonomisi Dönem Projesi Şartnamesi, 2026 (`roadmap/Engineering Economics Project_2026_CSE-481.pdf`; bölüm 12, 18, 31).
2. Lee, Y. ve ark. *Meta-Harness: End-to-End Optimization of Model Harnesses.* arXiv:2603.28052, 2026.
3. Ning, X. ve ark. *Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems.* arXiv:2605.18747, 2026.
4. Hou, Y. ve ark. *Harness-G: A Graph-Structured Harness for Search Agents.* arXiv:2607.27652, 2026.
5. Dettmers, T. ve ark. *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314, 2023.
6. Hu, E. ve ark. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021.
7. Proje içi belgeler: `reports/01`–`08`, `reports/10_hafta_ici_kalip_raporu.md`, `reports/weekday_patterns_report.pdf`, `roadmap/TASK_1`–`TASK_5`.

## Ek A – Sınıf içi gösterim akışı (yaklaşık 10 dakika)

1. **Testler (1 dk):** `python -m unittest discover -s tests` → `OK`.
2. **Ortam (30 sn):** `BIST_LLM_OFFLINE=1 python -m src.llm.check_local --adapter artifacts/llm/qwen_bist_lora/adapter` → `FAIL` yok.
3. **Arayüz (5 dk):** `python -m src.llm.web --adapter artifacts/llm/qwen_bist_lora/adapter` → `http://127.0.0.1:8765`. Sırayla beş örnek soru; her cevapta durum, araçlar, E1–E3 kanıtı, kontrol listesi. Q3'te guard'ın değiştirdiği taslağı ("Model draft before the guard") göster. Q5'te tavsiye ve fiyat hedefi reddini göster. Cevabın üstündeki "Answer written by" satırı cevabı kimin yazdığını söyler (model, modelin yeniden yazımı ya da harness özeti). Her cevabın altındaki **Run trace** düğmesiyle süreç kaydını aç: model ve adaptör özeti, `temperature=0`/greedy/seed yok, sistem ve kullanıcı istemleri, araç çağrıları (parametre ve tarih aralığı, çıktı), UTC zaman damgaları (`06` §1).
4. **Tuzak sorular (2 dk):** gelecek tarih ("25.09.2026'ya kadar"), evrende olmayan hisse, "kesin yükselir, al, hedef fiyat ver". Hepsi ret ya da `INVESTIGATE` olmalı.
5. **Replay (1 dk):** `python scripts/replay_demo.py` → bozulmamış günlük açılır, değiştirilmiş kopya reddedilir.
6. **Kanıt (30 sn):** `reports/evidence/` ve herhangi bir rapordaki sayıyı araç çıktısıyla eşleştir.

**Canlı soru için analiz tarihi.** Hazır örnekler 2026-09-10 ile üretilir (raporlarla tutarlı). Canlı soru Ekim–Kasım 2026'yı kapsayabilir (hoca: "1 Ocak 2025 – Kasım 2026"), router analiz tarihinden sonrasını reddettiği için sunumdan önce:
1. Fiyatları ve kurları tazele: `python -m src.data_pipeline`, `python -m src.macro --download`. TCMB faizi, TÜFE ve Fed serilerine eksik ayları resmî sitelerden ekle.
2. Arayüzü en son ortak veri günüyle aç: `python -m src.llm.web --adapter artifacts/llm/qwen_bist_lora/adapter --as-of <YYYY-MM-DD>`. Sayfa başlığı bu tarihi gösterir. İleri bakış koruması aynen çalışır: o tarihten sonrasını isteyen soru yine reddedilir.
3. 2026-10 ya da 2026-11 tarihli bir soruyla önceden dene (model eğitimde 2026-06-30, 2026-08-31 ve 2026-09-10 analiz tarihlerini gördü, daha geç bir tarih görmedi).

Süre notu: HF yolunda soru başına 30–60 sn (guard yeniden yazım isterse bir model çağrısı daha); hız gerekirse Ollama int8 (`notebooks/README.md`, Route A) kullanılabilir, resmî sayılar HF yoluyla alınmıştır.

## Ek B – Yeniden üretme komutları

```bash
python scripts/make_report_evidence.py          # reports/evidence/*.json (rapor 02–05)
python scripts/run_strategy_variants.py         # reports/evidence/strategy_variants.csv (rapor 07)
python scripts/replay_demo.py                   # rapor 06
python scripts/make_rag_file.py                 # reports/rag_knowledge_base.json (teslimdeki "RAG dosyası")
python scripts/run_scenarios.py --adapter <dir> # reports/execution_traces.json (beş senaryo, tam iz kaydıyla; model gerekir)
python scripts/make_submission.py --adapter <dir>  # submission.json + reports/fine_tuning_config.json (ayarlar adaptör dosyalarından okunur)
python -m src.experiments.cli                   # A–E simülasyonu (rapor 08)
python -m unittest discover -s tests            # tüm testler
```
