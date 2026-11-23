---
marp: true
paginate: true
title: BIST 100 Ajan Tabanlı Finansal Analiz ve Doğrulama Sistemi
---
<!-- Taslak: Marp ile PDF/PPTX'e çevrilebilir (`marp reports/11_sunum_taslagi.md --pptx`). Her sayı reports/09 ve reports/evidence altındaki araç çıktılarından gelir; sunmadan önce kontrol edin. -->

# BIST 100 Entegre Ajan Tabanlı Finansal Analiz ve Doğrulama Sistemi

CSE-481 Mühendislik Ekonomisi · Analiz tarihi 2026-09-10
Eğitsel araştırma: yatırım tavsiyesi değildir, aracı kurumla bağlantısı yoktur.

---

# Problem

- Dil modelleri finansal sorularda akıcı ama kanıtsız cevap verir: uydurma sayı, bilinmeyen tarih, "kesin yükselir".
- **Tez:** model hesap yapmaz. Hesabı deterministik araçlar yapar, harness süreci yönetir, backtest doğrular, kanıt kaydedilir, son kararı insan verir.

---

# Mimari

soru → yerel model (Qwen3-4B + LoRA) → JSON araç planı → araç yönlendirici doğrular → kalite kapısı → araçlar → kanıt kümesi (E1, E2…) → model özetler → **guard** → karar günlüğü (SHA-256) → replay

- Bulut yapay zekâ API'si yok; model yerelde çalışır.
- Zorunlu durum akışı (START → … → LOG_AND_CLOSE; yükleme adımları isteğe bağlı), her durum yalnızca izinli araçları çağırabilir.

---

# Veri

- 30 hisselik sabit evren, XU100, günlük fiyat (2024-01-02 →, analiz 2026-09-10'a kesik)
- Makro: USD/TRY, EUR/TRY, TCMB faizi, TÜFE, Fed faizi (yayın tarihine göre)
- Çeyreklik temel veri: 5 hisse, elle girilmiş (örnek, kaynak kimliği yok)
- Haber: kod hazır, veri boş

---

# Senaryo 1: Sektörde geride kalanlar

- Perakende: MAVI en geride (−%3,25 sektör medyanına göre)
- Geçmiş benzer gerilikler 10 günde %51,8 kapandı (494 olay, 53 bağımsız dönem)
- **Yakalama hipotezi desteklenmiyor** → `INVESTIGATE`

---

# Senaryo 2: Hafta içi kalıplar (THYAO)

- "Pzt-Sal-Çar düşer, Per-Cum yükselir": örnek içi 19/87 hafta, yalnızca 6'sında Per+Cum birlikte yükseldi (p = 0,26)
- Örnek dışı 3/30 hafta, hiçbiri yükselmedi
- Maliyet sonrası Çarşamba→Cuma: örnek içi %1,06, örnek dışı −%0,17
- 32 aday dizi için Bonferroni α = 0,0016 → **`REJECT_SIGNAL`**

---

# Senaryo 3: Teknik dönüşler (ASELS)

- 34 alt bant teması; 5 günde isabet %67,6, genel eğilim %59,2
- İdeal birleşik koşul hiç oluşmadı (n = 0)
- Tüm olay aileleri tamam: Bollinger, RSI tepe/dip, KAMA, Supertrend, Ichimoku, destek/direnç; piyasa ve oynaklık rejimi kırılımı
- `TECHNICAL_REVERSAL_CANDIDATE` (küçük örneklem uyarılı)

---

# Senaryo 4: Bilanço ve fiyat tepkisi

- TUPRS: iyileşen çeyrekler 20 günde daha iyi görünüyor, ama tek çeyreğe dayanıyor ve 5 günde ortadan kalkıyor
- Anormal getiri XU100'e ve (akranı varsa) sektör sepetine göre
- Tüm sayılar örnek veri → `INVESTIGATE`

---

# Backtest ve benchmark

- t+1 zamanlama, %0,10 komisyon, %0,05 kayma; A–E strateji varyantları, 30 hisse
- **Hiçbir varyant al-tut'u ya da XU100'ü sistematik geçmedi** (al-tut: 4/30, XU100: 2–3/30)
- Sektör sepetine göre alfa (17 hisse): medyan −%85 ile −%94 puan
- Dürüst bulgu: güçlü yükselen piyasada basit kurallar al-tut'un gerisinde

---

# Yerel model ve guard

| Metrik | Base | Adaptör |
|---|---|---|
| Geçersiz araç çağrısı | %66,7 | %0 |
| Desteklenmeyen sonuç | %25,0 | %0 |
| Kural ihlali (tavsiye) | %0 | %0 |
| Sayısal halüsinasyon (ham taslak) | %0* | %33,3 |

\*Base çoğu soruda geçerli plan üretemediği için adil kıyas değil. Guard sapan taslakları deterministik özetle değiştirir. 12 soru: istatistiksel güven sınırlı.

---

# Karar günlüğü ve replay

- Her analiz SHA-256'lı JSON olarak kaydedilir; ham model taslağı ve guard ihlalleri de saklanır
- Tek karakter değişirse replay reddeder
- **Canlı gösterim:** `python scripts/replay_demo.py`

---

# Bu çalışmada bulunup düzeltilen hatalar

- Önbellek analiz tarihinden sonrasını içeriyordu: ileri bakış hatası, sayılar düzeltildi
- Replay günlüğü guard'ın değişikliğini saklamıyordu
- Sektör sorusunda makro kanıt eksikti; harness artık kendisi ekliyor
- Hafta içi kalıbında maliyet ve çoklu test eksikti

---

# Sınırlamalar (açıkça)

- A–E harness deneyi **simüle sağlayıcıyla**; model kalitesini ölçmez
- Temel veri 5 hisse, elle girilmiş; haber verisi boş; TCMB ve TÜFE kaynağı doğrulanmadı
- Değerlendirme seti 12 soru; Colab 4-bit ve Mac bf16 küçük farklar verir
- Konuşma (speech) uygulanmadı (isteğe bağlı)

---

# Canlı gösterim (≈ 10 dk)

1. Testler: `python -m unittest discover -s tests` → `OK`
2. Arayüz: `python -m src.llm.web --adapter artifacts/llm/qwen_bist_lora/adapter`
3. Beş örnek soru + tuzaklar (gelecek tarih, evrende olmayan hisse, "kesin yükselir, al")
4. Replay: bozulmamış günlük açılır, değiştirilmiş kopya reddedilir
5. Kanıt: `reports/evidence/` ile rapordaki sayıyı eşleştir

---

# Sonuç

Sistem "kanıtsız sayı üretme" riskini mimariyle sınırlıyor. Dört senaryoda da işlem sinyali olarak savunulabilir bir sonuç çıkmadı ve sistem bunu söylüyor.

> Tahmin ≠ kârlı strateji · Korelasyon ≠ nedensellik · Backtest ≠ gelecek garantisi · Tarihsel kanıt ≠ yatırım tavsiyesi
