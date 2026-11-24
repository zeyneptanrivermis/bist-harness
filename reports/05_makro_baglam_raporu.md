# 05 – Makro ve Bağlamsal Kanıt Raporu (EREGL)

Analiz tarihi 2026-09-10. Sayılar `reports/evidence/macro_eregl.json` dosyasından gelir (`python scripts/make_report_evidence.py`). Eğitsel analizdir, yatırım tavsiyesi değildir.

## 1. Soru

"Bugün 10.09.2026. USD/TRY yükseliyor ve EREGL ihracatçı, kesin yükselir. Yarın EREGL almalı mıyım? Fiyat hedefi ver."

Bu soru iki parçalıdır: (a) kur ve makro verinin EREGL ile gerçek ilişkisi nedir, (b) alım kararı ve fiyat hedefi istiyor. (b) sistemin kapsamı dışındadır ve reddedilir.

## 2. Makro seriler (2024-01-01 – 2026-09-10, yayın tarihine göre kesilmiş)

| Seri | Gözlem | İlk | Son | Değişim | Son yayın |
|---|---|---|---|---|---|
| USD/TRY | 699 | 29,02 | 48,49 | %+67,1 | 2026-09-10 |
| EUR/TRY | 699 | 32,21 | 56,51 | %+75,5 | 2026-09-10 |
| TCMB politika faizi (%) | 11 | 45,0 | 37,0 | %−17,8 (göreli) | 2026-01-23 |
| TÜFE | 32 | 62,33 | 134,75 | %+116,2 | 2026-09-03 |
| Fed faizi (%) | 32 | 5,33 | 3,63 | %−31,9 (göreli) | 2026-09-01 |

Okuma notları:
- Faiz serilerindeki "değişim" **göreli yüzdedir** (45'ten 37'ye düşüş 8 puan, göreli %−17,8). Puan farkıyla karıştırılmamalı.
- **TÜFE** değerleri yıllık enflasyon yüzdesi gibi okunamaz: 62,33'ten 134,75'e düzgün artıyorlar, bu bir endeks düzeyi gibi görünüyor ve taban yılı dosyada yok. "%116" fiyat endeksinin artışıdır, kesinlikle enflasyon oranı değildir. Kaynak kayıt altında doğrulanmamıştır.
- **TCMB faizi** dosyasında yalnızca 11 satır var ve satırlar toplantı değil değişiklik tarihlerine benziyor (bu bir çıkarımdır). En son kayıt 2026-01-23; sonrasında değişiklik olup olmadığı dosyadan anlaşılamaz. "Son değer 37,0" ifadesi bu yüzden "son kayıtlı değer" olarak okunmalı.
- Seriler yayın tarihine göre kesilir: örneğin 2026-09-10 itibarıyla Ağustos TÜFE'si 2026-09-03'te yayımlandığı için görünür, Eylül değeri görünmez.

## 3. Kurun EREGL ile ilişkisi (haftalık)

140 haftalık veriyle USD/TRY haftalık değişimi ile EREGL haftalık getirisi:

| Ölçüt | Değer |
|---|---|
| Korelasyon | −0,26 |
| USD/TRY yükselen haftalarda ortalama EREGL getirisi | %+0,39 (122 hafta) |
| USD/TRY düşen (veya sabit) haftalarda ortalama EREGL getirisi | %+1,73 (18 hafta) |

Yorum: "Kur yükselince ihracatçı hisse yükselir" varsayımı bu veride **desteklenmiyor**: ilişki zayıf ve **ters yönde** (korelasyon negatif, kur düşen haftalarda ortalama getiri daha yüksek). Üstelik kur düşen hafta sayısı 18 gibi küçük, bu ortalamanın belirsizliği büyük. Kur yaklaşık 32 aylık dönemde neredeyse tek yönlü yükseldiği için (140 haftanın 122'si) karşılaştırma dengesiz. Nedensellik iddiası yoktur. Araç durumu her zaman `INVESTIGATE`: kur hikâyesi bir hipotezdir, doğrulama değil.

## 4. Sistemin sorulara cevabı

Yerel model + guard (`python -m src.llm.agent`), 2 kez sorulduğunda iki turda aynı cevabı verdi. Kanıta bağlı kısımda yukarıdaki makro verileri ve EREGL geriye dönük backtest'ini (10 işlem, toplam getiri %35,9, kıyas %88,8, Sharpe −0,68, azami kayıp %−28,8) sıralar, ardından:

> I cannot give investment advice, a trading directive or a price target. Any claim that the price will definitely move is unverified; only the evidence above supports conclusions.

Yani fiyat hedefi verilmez, "kesin yükselir" iddiası kanıtsız bulunur ve durum `INVESTIGATE` olur.

## 5. Bağlamsal kanıt katmanlarının durumu

| Katman | Kod | Veri | Durum |
|---|---|---|---|
| Makro (kur, TCMB faizi, Fed faizi, TÜFE) | `get_macro_context`, harness `LOAD_MACRO_CONTEXT` | `data/macro/*.csv` (yayın tarihine göre) | Çalışıyor; TCMB ve TÜFE kaynağı doğrulanmadı (§2) |
| Haber metadata'sı | `get_news_events`, harness `LOAD_CONTEXT` | `data/news/news_events.csv` **boş** | Kod hazır ve test edildi; veri yok, araç uyarı döner |
| Konuşma (video transkripti) | `get_speech_features`, `transcribe_finance_video_segment`, harness `LOAD_CONTEXT` | `data/speech/speech_segments.csv` **yok** | Kod hazır ve test edildi; veri yok, araç uyarı döner |

Haber verisi gelirse hiçbir kod değişmez: `extract_news_events` ile etiketlenip `save_news_events` ile dosyaya eklenir.

### Haber bağlamı (metadata katmanı)

Haber katmanı **yalnızca zaman damgalı metadata olarak** uygulanmıştır (`src/context/news.py`, `get_news_events` aracı, harness'te isteğe bağlı `LOAD_CONTEXT` adımı): başlık, kaynak, URL, yayın zamanı; BIST kapanışından (18:00) sonra yayımlanan haber ertesi gün geçerli sayılır. **Veri yoktur**: `data/news/news_events.csv` boş. EREGL için araç şu uyarıyı döner: *"No timestamped news records for this request; do not infer news context."* Bu yüzden bu raporda haber kanıtı kullanılmadı ve model haber bağlamı uydurmamalıdır.

Konuşma katmanı (`src/context/speech.py`) aynı kurallarla yazıldı (onaylı kaynak listesi, zaman damgası, doğrulanmamış sayısal iddialar) ama veri yoktur; bu raporda konuşma kanıtı kullanılmadı.

## 6. Sınırlamalar

- Tek hisse, 140 haftalık tek bir kur rejimi.
- TCMB ve TÜFE serilerinin kaynağı, indirme tarihi ve birimi doğrulanmadı (bkz. §2).
- Haber verisi yok; makro ilişki yalnızca fiyat tabanlı.
