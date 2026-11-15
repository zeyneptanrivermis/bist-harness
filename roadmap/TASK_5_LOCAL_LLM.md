# TASK 5: Yerel, Grup-Özel LLM ve İngilizce Araştırma Özeti

Bu katman fiyat tahmin motorunun yerine geçmez. Mevcut Python araçları
hesaplama, quality gate, point-in-time kesiti ve backtest sorumluluğunu korur.
Fine-tune edilen açık kaynak model yalnızca:

- gerekli olduğunda izinli bir aracı JSON biçiminde istemeyi,
- araç/harness sonucu verildiğinde yalnızca committed evidence'e bağlı İngilizce
  bir araştırma özeti yazmayı,
- veri güvenli değilse `ANALYSIS_UNSAFE` veya `HARD_BLOCK` sonucunu korumayı

öğrenir.

## Bileşenler

- `src/llm/training_data.py`: Bulut LLM kullanmadan sürümlenmiş A-E katalogdan
  review edilebilir başlangıç verisi ve sızıntısız split üretir.
- `src/llm/tool_loop.py`: Modelin JSON araç isteğini yalnızca mevcut yerel,
  cutoff-aware wrapper'lara bağlayan küçük güvenlik yönlendiricisi.
- `src/llm/build_corpus.py`: Açık OHLCV verisini isteğe bağlı indirip mevcut
  harness ile unreviewed İngilizce corpus adayları üretir.
- `notebooks/05_local_llm_finetuning.ipynb`: Mac/CPU smoke-test ve Colab CUDA
  QLoRA eğitim profillerini içerir.
- `requirements-llm.txt`: Opsiyonel eğitim bağımlılıkları.

## Deney protokolü

1. Grup, seed kayıtlarını gerçek deterministik araç çıktılarıyla çoğaltır ve
   İngilizce hedef metinleri kendisi gözden geçirir.
2. `human_reviewed=true` olmayan örnekle adapter eğitilmez.
3. Aynı `case_id` eğitim ve testte birlikte bulunmaz.
4. Base model ve LoRA adapter, A-E held-out vakalarında aynı promptlarla
   karşılaştırılır: desteklenmeyen iddia, doğrudan işlem tavsiyesi, ileri tarih
   ve izin dışı tool çağrısı ihlalleri raporlanır.

Model ve adapter sürümü, seed, veri manifesti ve adapter hash'i rapora
yazılmalıdır. Checkpoint ve veri Git'e eklenmez.

## İlk veri üretimi

Boş bir veri klasörü için aşağıdaki komut, 30 hisselik evren ile XU100'ün açık
OHLCV verisini indirir; ortak beş cutoff tarihinde senaryoları çalıştırır ve
inceleme dosyalarını üretir:

```powershell
python -m src.llm.build_corpus --download --dates-per-ticker 3
```

Çıktılar `data/llm/candidates/` altında `candidates.jsonl`, ekip incelemesi
için `candidates_review.csv` ve hata/atlama listesi içeren `manifest.json`
olarak oluşur. Bunlar doğrudan eğitim verisi değildir.

Küçük bir gerçek smoke test için yalnızca iki hisse, teknik ve takvim
senaryoları seçilebilir. Bu komut sadece gerekli iki hisse ile XU100 verisini
indirir:

```powershell
python -m src.llm.build_corpus --download --tickers ASELS THYAO --scenarios technical_reversal weekday_pattern --dates-per-ticker 2 --output-dir data/llm/smoke_test
```

## CSV incelemesini eğitim JSONL'sine dönüştürme

`candidates_review.csv` dosyası Excel veya LibreOffice'te açılır. Onaylanacak
satırlarda `human_reviewed` alanı `true` yapılır; `reviewer` doldurulur ve
grubun düzenlediği İngilizce metin `reviewed_summary` alanına yazılır. Sonra:

```powershell
python -m src.llm.prepare_reviewed_records
```

Komut, yalnızca eksiksiz onaylı satırları `data/llm/reviewed_records.jsonl`
dosyasına aktarır; `BUY/SELL` gibi yasak ifade içeren veya eksik satırlarla
eğitim başlatmaz.

## Gözden geçirme kuralları (adım 2)

Üretimde isteğe bağlı olarak `--reviewers Ad1 Ad2 Ad3` verilirse satırlar
`assigned_to` sütununa dağıtılır. `reviewed_summary` taslakla dolu gelir;
`human_reviewed` her zaman `false` başlar. Her gözden geçiren yalnızca kendi
satırlarını düzenler:

1. Taslaktaki her rakamı `user_prompt` içindeki evidence JSON'u ile karşılaştırın.
2. Evidence'ta olmayan hiçbir sayı, neden veya gelecek bilgisi eklemeyin.
3. `review_notes` içindeki otomatik uyarıları ele alın (veri kalitesi, durum/skor
   çelişkisi, benchmark'ı geçemeyen backtest). Çözemiyorsanız satırı `false` bırakın.
4. BUY/SELL, hedef fiyat veya getiri garantisi kullanmayın; eğitim amaçlı
   sınırlama cümlesini koruyun.
5. Onaylayınca `human_reviewed=true` ve `reviewer` doldurun, sonra
   `python -m src.llm.prepare_reviewed_records` ile doğrulayın.
