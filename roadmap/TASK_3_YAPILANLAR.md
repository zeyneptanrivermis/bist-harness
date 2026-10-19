# TASK 3: MCP, Kalite Kapısı ve Stateful Harness

Task 3, hesap motorlarını yeniden yazmadan onları denetlenebilir bir ajan akışına bağlar.

## Eklenen bileşenler

- `src/mcp_tools/`: 18 JSON-serileştirilebilir MCP wrapper'ı ve FastMCP sunucu fabrikası. Varsayılan veri kaynağı yerel önbellektir; araç çağrısı sessizce yeni piyasa verisi indirmez.
- `src/harness/quality.py`: Zorunlu OHLCV sütunları, veri eksikliği, geçersiz OHLC aralığı, minimum geçmiş, aşırı hareket ve point-in-time kesiti denetimi.
- `src/harness/harness.py`: `START → SELECT_UNIVERSE → LOAD_DATA → CHECK_DATA_QUALITY → CALCULATE → RUN_SCENARIO → RUN_BACKTEST → BUILD_EVIDENCE → HUMAN_REVIEW → LOG_AND_CLOSE` durum makinesi.
- `src/harness/evidence.py`: Yalnızca kalite kapısı ve deterministik araç çıktılarından oluşturulan committed evidence seti.
- `src/harness/replay.py`: SHA-256 doğrulamalı JSON analiz kaydı ve replay bütünlük denetimi.
- `tests/test_task3_harness.py`: Yeni akış ve mevcut indikatör/senaryo/backtest/fundamental motorları için ağdan bağımsız regresyon testleri.

## Çalıştırma

```powershell
python -m unittest discover -s tests -v
python -m src.mcp_tools.server
```

MCP araçları `as_of_date` alır. Harness, bu tarihten sonraki satırları hesaplamadan önce çıkarır; `uses_forward_data=True` veya `signal_shift < 1` bildirilen bir istek `HARD_BLOCK`/`ANALYSIS_UNSAFE` sonucuyla durur.

Temel finansal kayıtlar hâlâ yerel curated veri kümesindedir. KAP kaynağı/URL kimlikleri eklenene kadar bu kayıtlar araç yanıtında ve kalite raporunda kaynak uyarısıyla taşınır.
