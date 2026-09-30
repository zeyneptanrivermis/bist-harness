# TASK 1: PROJE ALTYAPISI VE TEMEL VERİ KATMANI TAMAMLANDI
## CSE-481 Mühendislik Ekonomisi — BIST 100 Entegre Ajan Analiz Sistemi
**Durum:** Tamamlandı ve Test Edildi  
**Hedef Kitle:** Proje Ekip Arkadaşı (Sıfır teknik önbilgi ile adım adım anlaşılabilir formatta)

---

## 1. GENEL BAKIŞ: BU AŞAMADA NE YAPTIK?

Task 1 kapsamında projenin temel çalışma omurgasını sıfırdan kurduk:
1. Python 3.12 ile yalıtılmış sanal çalışma ortamı (`.venv`) oluşturuldu ve gerekli kütüphaneler yüklendi.
2. Hocanın şartnamesindeki 30 BIST 100 hissesi ve sektörleri kodlandı.
3. Yahoo Finance üzerinden 30 hissenin ve BIST 100 (`XU100.IS`) endeksinin son 2 yıllık günlük fiyat verileri otomatik indirildi, temizlendi ve yerel diske kaydedildi.
4. Deterministik (sayısal/matematiksel) teknik analiz göstergeleri (Bollinger, RSI, KAMA, Supertrend, Ichimoku) sıfırdan hatasız kodlandı ve `THYAO` hissesi üzerinde test edildi.
5. 1. Zorunlu Araştırma Senaryosu olan "Sektör Gerisinde Kalanlar (Sector Laggard / Catch-Up)" analiz motoru yazıldı ve çalıştırıldı (746 olay tespit edildi, %51.47 başarı oranı ölçüldü).

---

## 2. OLUŞTURULAN KLASÖR YAPISI

Proje dizininde modüler ve endüstri standardı bir mimari kuruldu:

```
economics/
├── .venv/                   # Python 3.12 sanal ortamı (tüm kütüphaneler burada)
├── data/
│   ├── raw/                 # İnternetten inen ham veriler (31 adet .parquet ve .csv)
│   ├── processed/           # Temizlenmiş, getirisi hesaplanmış veriler (31 adet)
│   └── macro/               # TCMB faiz, döviz ve enflasyon serileri için ayrılan alan
├── notebooks/               # Jupyter analiz defterleri için ayrılan alan
├── reports/                 # Rapor ve grafik çıktılarının kaydedileceği klasör
├── src/
│   ├── __init__.py          # Python paket tanımlayıcısı
│   ├── config.py            # Ayarlar, tarih aralıkları, komisyon oranları ve sabitler
│   ├── universe.py          # 30 hisselik BIST çalışma evreni ve sektör fonksiyonları
│   ├── data_pipeline.py     # Yahoo Finance veri çekme, temizleme ve önbellekleme
│   ├── indicators.py        # Deterministik Teknik İndikatörler Motoru
│   ├── scenarios/
│   │   └── sector_laggard.py# 1. Zorunlu Senaryo: Sektör Gerisinde Kalanlar Motoru
│   ├── backtest/            # Backtest simülatörü için ayrılan modül
│   ├── mcp_tools/           # Yapay zeka MCP araçları için ayrılan modül
│   └── harness/             # Stateful Harness ve Veri Kalite Kapısı için ayrılan modül
├── requirements.txt         # Yüklenen kütüphanelerin listesi
├── PROJE_REHBERI_VE_YOL_HARITASI.md # Büyük resim, felsefe ve ürünleştirme rehberi
├── TASK_1_YAPILANLAR.md     # (Bu Belge) Task 1 detaylı çalışma raporu
└── .gitignore               # Git'e gereksiz/büyük dosyaların gitmesini engelleyen kural dosyası
```

---

## 3. OLUŞTURULAN DOSYALAR VE ÇALIŞMA MANTIKLARI

Aşağıdaki 6 ana dosya oluşturulmuş ve her biri tek tek doğrulanmıştır:

### 1. `requirements.txt` (Kütüphane Listesi)
* **Amacı:** Projenin ihtiyaç duyduğu araç takımlarını belirler.
* **Yüklenenler:**
  * `pandas`, `numpy`, `scipy`: Veri işleme, zaman serisi tabloları ve istatistiksel matematik.
  * `yfinance`: Yahoo Finance üzerinden Borsa İstanbul verilerini çekme.
  * `fastmcp`: İlerleyen aşamalarda Python fonksiyonlarımızı yapay zekaya (LLM) birer MCP aracı olarak tanıtma.
  * `matplotlib`, `seaborn`: Finansal grafik ve getiri eğrileri çizimi.
  * `pyarrow`: Verileri diskte çok hızlı okunan `.parquet` formatında sıkıştırma.

### 2. `src/config.py` (Merkezi Ayarlar ve Kurallar)
* **Amacı:** Kodun içine tarih veya oran gömmek yerine tüm kuralları tek merkezden yönetmek.
* **İçerdiği Önemli Kurallar:**
  * `WARMUP_START_DATE = "2024-01-01"`: İndikatörlerin (200 günlük ortalama vb.) ısınması için veri başlangıç tarihi.
  * `ANALYSIS_START_DATE = "2025-01-01"`: Hocanın şartnamesinde belirttiği TMS 29 enflasyon muhasebesi sonrası tutarlı analiz dönemi.
  * `DEFAULT_COMMISSION_RATE = 0.001`: Borsa işlemlerinde binde 1 aracı kurum komisyonu.
  * `DEFAULT_SLIPPAGE_RATE = 0.0005`: Binde 0.5 fiyat kayma payı (slippage).
  * `SIGNAL_SHIFT_DAYS = 1`: **Look-ahead bias (geleceğe bakma hatası)** olmasın diye; bugün kapanışta oluşan sinyale göre işlemi ancak yarın sabah ($t+1$) gerçekleştirebilme kuralı.
  * `ALLOWED_ACTIONS`: Sistemin izin verdiği sonlu karar durumları (`HOLD`, `WATCH`, `POTENTIAL_CATCH_UP_CANDIDATE`, `TECHNICAL_REVERSAL_CANDIDATE`, `ANALYSIS_UNSAFE` vb.).

### 3. `src/universe.py` (30 BIST 100 Hissesi ve Sektörleri)
* **Amacı:** Şartnamenin 5. bölümündeki çalışma evrenini resmi olarak tanımlamak.
* **İçeriği:**
  * 30 hissenin Borsa kodu, Yahoo kodu ve sektörü (Örn: ASELS $\to$ ASELS.IS / Teknoloji, THYAO $\to$ THYAO.IS / Havacılık, AKBNK $\to$ AKBNK.IS / Bankacılık...).
  * Gösterge Endeks: `XU100.IS` (BIST 100 Endeksi).
  * **Fonksiyonlar:** 
    * `get_all_tickers()`: 30 hissenin kod listesini döner.
    * `get_sector_map()`: Hangi hissenin hangi sektörde olduğunu döner.
    * `get_stocks_in_sector("Banking")`: Bankacılık hisselerini döner.
    * `get_peer_group("THYAO")`: THYAO'nun sektördaşlarını (PGSUS vb.) döner.

### 4. `src/data_pipeline.py` (Veri Çekme, Temizleme ve Önbellekleme)
* **Amacı:** İnternete bağlanıp fiyatları çeker, temizler ve diske kaydeder.
* **Nasıl Çalışır?**
  1. `fetch_single_ticker()`: Yahoo Finance'tan her hissenin Open, High, Low, Close, Volume (OHLCV) verisini çeker.
  2. `clean_and_preprocess_ohlcv()`: Eksik günleri kontrol eder, ileriye doğru doldurur, günlük yüzde getirisini (`Daily_Return`) ve Türk Lirası işlem hacmini (`Turnover = Fiyat x Hacim`) hesaplar.
  3. `download_entire_universe()`: 30 hisse + XU100 endeksini indirip `data/raw/` ve `data/processed/` klasörlerine hem `.parquet` (yüksek hız) hem de `.csv` (Excel ile incelenebilir) olarak kaydeder.
* **Sonuç:** `data/processed/` altında 31 dosya başarıyla oluşturuldu.

### 5. `src/indicators.py` (Deterministik Teknik İndikatörler Motoru)
* **Amacı:** Şartnamede istenen göstergeleri yapay zekanın insafına bırakmadan, saf matematiksel formüllerle hesaplamak.
* **Hesaplanan Göstergeler:**
  1. **SMA & EMA:** 20, 50, 200 günlük hareketli ortalamalar.
  2. **KAMA (Kaufman Adaptive Moving Average):** Piyasa oynaklığına göre hassasiyetini kendisi ayarlayan uyarlanabilir ortalama ve bunun eğimi (`KAMA_Slope`).
  3. **RSI (14 günlük Wilder):** 0-100 arası aşırı alım/aşırı satım göstergesi.
  4. **MACD:** 12-26-9 parametreli momentum ve histogram göstergesi.
  5. **Bollinger Bantları:** Üst bant, orta bant, alt bant, %B ve Bant Genişliği (Bandwidth).
  6. **ATR (14 günlük):** Volatilite (gerçek aralık ortalaması).
  7. **Supertrend (10, 3.0):** Trend yönü (+1 Boğa, -1 Ayı).
  8. **Ichimoku Cloud:** Tenkan-sen, Kijun-sen, Senkou Span A/B, Bulut Üstü (`ABOVE_CLOUD`), Bulut Altı (`BELOW_CLOUD`) tespiti.
* **Test Sonucu:** `THYAO` hissesi üzerinde denendi; 689 günlük veriye 34 sütunluk analitik göstergeler başarıyla eklendi.

### 6. `src/scenarios/sector_laggard.py` (1. Araştırma Senaryosu: Catch-Up)
* **Amacı:** *"Bir sektör yükselirken geride kalan hisse sonradan arayı kapatır mı?"* hipotezini test etmek.
* **Nasıl Çalışır?**
  1. `calculate_sector_returns()`: Her hissenin ve her sektörün 20 günlük medyan getirisini hesaplar.
  2. `calculate_sector_lag_scores()`: `Hisse Getirisi - Sektör Getirisi` farkını bulur (Eksi ise hisse geridedir).
  3. `evaluate_laggard_catchup()`: Sektörün güçlü olduğu (+%5 üzeri) ama hissenin geride kaldığı (-%3'ten fazla) tüm geçmiş günleri bulur ve bu hissenin sonraki 5, 10 ve 20 günde sektöründen daha çok getiri sağlayıp sağlamadığını (`Outperformance`) ölçer.
* **Test Sonucu:** 
  * Tespit edilen toplam vaka: **746 olay**.
  * 5 günlük arayı kapatma başarı oranı: **%51.47**. (Yani tek başına sektör geriliği bir alım garantisi değildir; temel analiz ve makro kanıtlarla filtrelenmesi gerekir!)

---

## 4. KODLARI TEST ETMEK VE ÇALIŞTIRMAK İÇİN KOMUTLAR

Herhangi bir arkadaşın terminalde şu komutları çalıştırarak yapılanları anında kendi gözüyle görebilir:

```bash
# 1. Proje ana klasörüne git
cd /Users/zeyneptanrivermis/Desktop/projects/economics

# 2. İndirilen 31 veri dosyasını listele
ls -lh data/processed/

# 3. İndikatör hesaplamasını THYAO üzerinde test et
PYTHONPATH=. .venv/bin/python -c "
from src.data_pipeline import load_ticker_data
from src.indicators import calculate_all_indicators

df = load_ticker_data('THYAO')
enriched = calculate_all_indicators(df)
print(enriched[['Close', 'RSI_14', 'Supertrend_Direction', 'Ichimoku_State']].tail(5))
"

# 4. Sektör Gerisinde Kalanlar (Catch-Up) motorunu çalıştır
PYTHONPATH=. .venv/bin/python -c "
from src.data_pipeline import load_ticker_data
from src.universe import get_all_tickers
from src.scenarios.sector_laggard import calculate_sector_lag_scores, evaluate_laggard_catchup

tickers = get_all_tickers()
prices = {t: load_ticker_data(t) for t in tickers}
events = evaluate_laggard_catchup(prices)
print('Tespit edilen olay sayısı:', len(events))
print('5 Günlük Başarı Oranı:', f\"{events['Fwd_5d_CatchUp_Hit'].mean():.2%}\")
"
```

---

## 5. SIRADAKİ ADIMLAR (TASK 2)

Bir sonraki aşamada yapılacaklar:
1. **Diğer 3 Araştırma Senaryosunun Yazılması:**
   * Senaryo 2: Gün ve Çok Günlük Kalıplar (`weekday_patterns.py`)
   * Senaryo 3: Teknik Dönüş Sinyalleri (`technical_reversals.py`)
   * Senaryo 4: Bilanço Açıklama Reaksiyonu (`fundamental_reaction.py`)
2. **Backtesting Motorunun Kurulması:**
   * $t+1$ gecikmeli al-sat simülasyonu, komisyon, Sharpe oranı ve Max Drawdown hesaplayıcı.
3. **MCP Tool ve Kalite Kapısı Katmanı:**
   * Bu fonksiyonların LLM'e araç (tool) olarak sunulması.

---
*Bu doküman, projede Task 1'de yapılan tüm adımları ve çalışma mantığını belgelemektedir.*
