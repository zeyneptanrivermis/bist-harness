# 01 – Veri Sözlüğü (Data Dictionary)

Analiz tarihi: **2026-09-10** (bu tarihten sonraki hiçbir veri kullanılmaz). Her veri kümesi için kaynak, sütunlar ve **yayın tarihi kuralı** aşağıdadır. Sütun adları kodla birebir aynıdır; `tests/test_report_docs.py` her sütunun burada geçtiğini denetler.

## 1. Evren (`src/universe.py`)

30 hisse + XU100 endeksi. Sütunlar: `ticker` (BIST kodu), `yahoo_symbol` (Yahoo Finance sembolü, `.IS` eki), `sector` (sektör etiketi, elle atanmış).

| Sektör | Hisseler |
|---|---|
| Technology / Defense | ASELS |
| Petroleum / Refining | TUPRS |
| Retail Trade | BIMAS, MGROS |
| Retail / Apparel | MAVI |
| Transportation / Airlines | THYAO, PGSUS |
| Banking | AKBNK |
| Holding / Investment | KCHOL |
| Basic Metals / Steel | EREGL, KRDMD |
| Telecommunications | TCELL, TTKOM |
| Food & Beverage | CCOLA |
| Chemicals / Plastics | SASA |
| Mining | TRALT |
| Automotive | FROTO, TOASO |
| Construction | ENKAI |
| Healthcare | MPARK |
| Real Estate Investment Trust | EKGYO |
| Chemicals / Fertilizer | GUBRF |
| Insurance | TURSG |
| Electric Utilities | ENJSA, AKSEN |
| Cement | OYAKC |
| Brokerage / Capital Markets | ISMEN |
| Durables / Machinery | ARCLK |
| Healthcare / Pharmaceuticals | ECILC |
| Petrochemicals | PETKM |

**Sınırlama:** 24 sektör etiketinin yalnızca 6'sında 2 hisse vardır; diğerlerinde tek hisse olduğu için sektör karşılaştırması yapılamaz (örneğin Banking yalnızca AKBNK). Sektör karşılaştırması için en az 2 hisse gerekir.

## 2. Fiyat verisi (`data/raw/<TICKER>.csv` ve `.parquet`)

- **Kaynak:** Yahoo Finance (`yfinance`), günlük OHLCV, 2024-01-01 başlangıçlı. Veri 689 işlem günü (ASELS örneği).
- **Yayın tarihi kuralı:** Bir günün çubuğu o günün kapanışında bilinir. Sinyaller en erken **t+1** günü işleme alınır (`signal_shift=1`).

| Sütun | Anlam |
|---|---|
| `Date` | İşlem günü (indeks) |
| `Open`, `High`, `Low`, `Close` | Günlük açılış, en yüksek, en düşük, kapanış (TRY) |
| `Adj Close` | Temettü/bölünme düzeltilmiş kapanış |
| `Volume` | İşlem adedi |
| `Daily_Return` | `Adj Close` günlük yüzde değişimi |
| `Turnover` | `Close × Volume` (TRY cinsinden işlem hacmi) |

## 3. Teknik göstergeler (`src/indicators.py`, `calculate_all_indicators`)

Fiyat sütunlarından deterministik hesaplanır, saklanmaz. Her gösterge yalnızca o güne kadarki veriyi kullanır.

| Sütun | Anlam |
|---|---|
| `SMA_20`, `SMA_50`, `SMA_200` | Basit hareketli ortalamalar |
| `EMA_20` | Üssel hareketli ortalama |
| `KAMA_10` | Kaufman uyarlanabilir hareketli ortalama |
| `KAMA_Slope`, `KAMA_Slope_State`, `KAMA_Position` | KAMA eğimi, eğim durumu, fiyatın KAMA'ya göre konumu |
| `RSI_14` | 14 günlük RSI |
| `MACD`, `MACD_Signal`, `MACD_Hist` | MACD, sinyal çizgisi, histogram |
| `ATR_14` | 14 günlük ortalama gerçek aralık |
| `BB_Middle`, `BB_Upper`, `BB_Lower` | Bollinger orta, üst, alt bant |
| `BB_Bandwidth`, `BB_PctB` | Bant genişliği, %B |
| `BB_Lower_Touch`, `BB_Upper_Touch`, `BB_Middle_Test` | Bant teması bayrakları |
| `Supertrend`, `Supertrend_Direction`, `Supertrend_State` | Supertrend düzeyi, yönü, durumu |
| `Tenkan_Sen`, `Kijun_Sen`, `Senkou_Span_A`, `Senkou_Span_B` | Ichimoku çizgileri |
| `Cloud_Top`, `Cloud_Bottom`, `Ichimoku_State`, `Chikou_vs_Past` | Bulut sınırları, durum, Chikou karşılaştırması |
| `Resistance_Level`, `Support_Level` | Pencere bazlı direnç ve destek düzeyi |
| `Dist_Resistance_Pct`, `Dist_Support_Pct` | Fiyatın direnç/destek düzeyine yüzde uzaklığı |

## 4. Makro serileri (`data/macro/*.csv`)

| Dosya | İçerik | Kaynak (dosya başlığına göre) | Yayın tarihi kuralı |
|---|---|---|---|
| `USDTRY.csv`, `EURTRY.csv` | Günlük kur (sütunlar `date`, `value`) | Yahoo Finance | Gün sonu değeri, aynı gün |
| `TCMB_POLICY_RATE.csv` | Bir haftalık repo (politika) faizi, % | TCMB EVDS | `published_date` = karar günü |
| `TUIK_CPI.csv` | TÜFE | TÜİK / TCMB EVDS | `published_date` dönemden sonraki ay (ör. Ocak değeri Şubat başında) |
| `FED_FUNDS.csv` | Fed faiz oranı, % | FRED `FEDFUNDS` | `published_date` dönemden sonraki ay başı |

Resmi üç dosyanın sütunları: `date` (değerin ait olduğu dönem), `value`, `published_date` (kamuya açıklandığı gün). Yükleyici seriyi `published_date` ile indeksler; `get_macro_context` analiz tarihinden sonra yayımlanan satırları keser. `published_date` sütunu yoksa `date` kullanılır ve bu sızıntı riskidir.

**Doğrulanmamış noktalar:** `TUIK_CPI.csv` değerleri 62,33'ten 132,31'e düzgün artıyor; bu yıllık enflasyon yüzdesi değil, bir **endeks düzeyi** gibi görünüyor ve taban yılı dosyada belirtilmemiş. Dosyalar repoya "Macro Finance Data" commit'iyle eklenmiş, kaynak sorgusu/indirme tarihi kayıtlı değil.

## 5. Çeyreklik temel veriler (`data/fundamentals/bist_quarterly_fundamentals.csv`)

- **Kaynak:** `src/fundamentals.py` içindeki elle girilmiş (curated) kayıtlar. Kod yorumu KAP açıklama zamanlarını anıyor ama **KAP bildirim numarası/kaynak kimliği yoktur**, bu yüzden doğrulanamaz. Tüm sayılar **örnek/gösterim amaçlı** kabul edilmelidir.
- **Kapsam:** yalnızca 5 hisse (ASELS, BIMAS, FROTO, THYAO, TUPRS), 40 satır.
- **Yayın tarihi kuralı:** bir çeyrek, yalnızca `kap_disclosure_date` tarihinden sonra kullanılabilir (point-in-time).

| Sütun | Anlam |
|---|---|
| `ticker` | Hisse kodu |
| `period` | Çeyrek (ör. `2025Q3`) |
| `kap_disclosure_date` | Çeyreğin kamuya açıklandığı gün |
| `revenue` | Hasılat (para birimi/ölçek dosyada belirtilmemiş) |
| `ebitda` | FAVÖK |
| `net_income` | Net kâr |
| `net_debt` | Net borç (negatif = net nakit) |
| `roe` | Özkaynak kârlılığı |
| `ebitda_margin` | FAVÖK / hasılat |

## 6. Haber metadata'sı (`data/news/news_events.csv`)

- **Kaynak:** takımın doldurduğu dosya ya da `extract_news_events` ile etiketlenen derlenmiş başlıklar. Yalnızca başlık, kaynak, URL ve yayın zamanı saklanır; haber gövdesi saklanmaz, hiçbir şey kazınmaz.
- **Durum:** dosya şu an **boş** (yalnızca başlık satırı, 0 kayıt). Boş dosya "haber kanıtı yok" anlamına gelir, uyarı verir, olay uydurulmaz.
- **Yayın tarihi kuralı:** `published_at` saati 18:00 ve sonrasıysa (BIST kapanışı) haber ancak ertesi gün kullanılabilir. `effective_date` bunu uygular.

| Sütun | Anlam |
|---|---|
| `event_id` | Kaynak, zaman ve başlıktan türetilen kimlik |
| `published_at` | Yayın zamanı |
| `effective_date` | Kararda kullanılabileceği ilk gün |
| `ticker` | Haberde geçen evren hissesi (boş = piyasa geneli) |
| `sector` | Hissenin sektörü |
| `topics` | Anahtar kelime kurallarıyla etiketlenen konular (`;` ile ayrılmış) |
| `stance` | `positive`, `negative`, `mixed`, `neutral` (bağlam etiketi, gerçek değil) |
| `headline` | Başlık |
| `source` | Kaynak |
| `url` | Bağlantı |

**Konuşma (speech-to-text) kapsam dışıdır**, uygulanmadı.

## 7. Türetilmiş özellikler

Hepsi araç çıktılarından hesaplanır ve `reports/evidence/*.json` içinde saklanır.

| Alan | Anlam |
|---|---|
| `Sector_Return_Lookback`, `Stock_Return_Lookback` | 20 günlük sektör medyanı ve hisse getirisi |
| `Lag_Score` | `hisse getirisi − sektör getirisi` (negatif = geride) |
| `Entry_Date`, `Entry_Price` | Sinyalden bir sonraki gün giriş (t+1) |
| `Fwd_5d_Stock_Return`, `Fwd_10d_Stock_Return`, `Fwd_20d_Stock_Return` | Girişten sonraki 5/10/20 gün hisse getirisi |
| `Fwd_5d_Sector_Return`, `Fwd_10d_Sector_Return`, `Fwd_20d_Sector_Return` | Aynı pencerede sektör getirisi |
| `Fwd_5d_Outperformance`, `Fwd_10d_Outperformance`, `Fwd_20d_Outperformance` | Hisse eksi sektör getirisi |
| `Fwd_5d_CatchUp_Hit`, `Fwd_10d_CatchUp_Hit`, `Fwd_20d_CatchUp_Hit` | Fark kapandıysa 1 |
| `Volume_Ratio` | O günkü hacim / 20 günlük hacim ortalaması (sıçrama eşiği 1,2) |
| `Volume_Regime` | `HIGH` (oran ≥ 1,2), `LOW` (< 0,8), aksi halde `NORMAL` |
| `hit_rate_5d` | Olaydan sonra 5 gün içinde fiyatın yükseldiği olay oranı |
| `ebitda_margin_change_qoq` | FAVÖK marjının çeyreklik değişimi |
| `net_debt_to_annualized_ebitda` | Net borç / yıllıklandırılmış FAVÖK |
| `nd_ebitda_change_qoq` | Bu oranın çeyreklik değişimi |
| `improving` | Marj arttı ve kaldıraç azaldıysa doğru |
| `ret_1d`, `ret_5d`, `ret_20d` | Açıklamadan sonra 1/5/20 gün getiri |
| `car_xu100_1d`, `car_xu100_5d`, `car_xu100_20d` | XU100'e göre anormal getiri |
| `car_sector_1d`, `car_sector_5d`, `car_sector_20d` | Sektöre göre anormal getiri (sektör sepeti yoksa boş) |
| `Total_Return`, `Benchmark_Total_Return`, `Buy_and_Hold_Return` | Strateji, XU100 ve al-tut toplam getirisi |
| `Annualized_Return_CAGR`, `Sharpe_Ratio`, `Max_Drawdown`, `Win_Rate`, `Total_Trades`, `Signal_Days` | Backtest metrikleri |
| `Alpha_vs_Benchmark` | Strateji getirisi eksi XU100 getirisi |
