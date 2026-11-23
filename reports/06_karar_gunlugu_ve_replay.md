# 06 – Karar Günlüğü ve Replay Gösterimi

Amaç: bir analizin sonradan **değiştirilemediğini ve neyin ne zaman yapıldığını gösterebildiğimizi** kanıtlamak. Eğitsel çalışmadır.

## 1. Bir analiz ve günlüğü

Soru (THYAO, hafta içi deseni), yerel model + guard ile sorulur:

```bash
BIST_LLM_OFFLINE=1 python -m src.llm.agent --adapter artifacts/llm/qwen_bist_lora/adapter "<soru>"
```

Her cevap `data/analysis_logs/analysis_<kimlik>.json` olarak yazılır. Kimlik, `(soru, analiz tarihi, araç planı)` karmasından türer: **aynı soru aynı günlüğe yazar**. Bu bilinçlidir, tekrarlanabilirliği (aynı soru, aynı cevap) gösterir ama günlüğün üzerine yazar.

Örnek günlük, bu repoda saklanır: `reports/evidence/example_analysis_log.json`. İçeriği:

| Alan | Anlam |
|---|---|
| `analysis_id` | Soru + tarih + plandan türeyen kimlik |
| `request` | Soru ve analiz tarihi (`2026-09-10`) |
| `state_history` | Harness durum makinesinin geçtiği adımlar: `START → SELECT_UNIVERSE → LOAD_DATA → CHECK_DATA_QUALITY → CALCULATE → RUN_SCENARIO → RUN_BACKTEST → BUILD_EVIDENCE → LOG_AND_CLOSE` |
| `plan` | Modelin seçtiği ve router'ın doğruladığı araç çağrıları (burada `test_weekday_pattern`) |
| `evidence` | Taahhüt edilen kanıt kümesi E1–E3 (kalite kapısı, senaryo, backtest) |
| `action` | Sonuç durumu (`REJECT_SIGNAL`) |
| `answer` | Kullanıcıya giden cevap |
| `raw_answer` | Modelin guard'dan önceki ham taslağı |
| `violations` | Guard'ın taslakta bulduğu ihlaller (burada boş) |
| `replay_sha256` | Kayıt içeriğinin SHA-256 özeti |

`raw_answer` ve `violations` alanları bu çalışmayla eklendi: önceki sürümde günlük yalnızca son cevabı saklıyordu, guard'ın modelin taslağını değiştirip değiştirmediği günlükten anlaşılamıyordu.

## 2. Replay: bozulmamış günlük açılır, tek karakter değişince reddedilir

```bash
python scripts/replay_demo.py reports/evidence/example_analysis_log.json
```

Çıktı:

```
1. Untouched log replays: action=REJECT_SIGNAL, tools=['test_weekday_pattern'], evidence=['E1', 'E2', 'E3']
   states: START -> SELECT_UNIVERSE -> LOAD_DATA -> CHECK_DATA_QUALITY -> CALCULATE -> RUN_SCENARIO -> RUN_BACKTEST -> BUILD_EVIDENCE -> LOG_AND_CLOSE
   guard violations: [], raw draft stored: True, final answer == raw draft: True
   hash:   892a294162558415237bf433f581ca1ca6194fc9f84b4a6152abe686ba308de8
2. Tampered copy (one edit) is rejected: Replay log integrity check failed: content hash does not match.
```

Betik günlüğün bir **kopyasında** tek bir ifadeyi (`REJECT_SIGNAL` → `WATCH`) değiştirir; `replay_analysis` karma uyuşmadığı için `ValueError` verir. Orijinal dosya değişmez.

## 3. Bu ne garanti eder, ne etmez

- **Eder:** kayıt oluşturulduktan sonra içeriğinin (soru, plan, kanıt, cevap) değiştirilmediği. Karma, kaydın kendisinden hesaplanır (`sort_keys`, kanonik JSON).
- **Etmez:** karmanın kaydın yanında saklanması yüzünden, **hem içeriği hem karmayı birlikte yeniden yazan** biri bunu fark edilmeden yapabilir. Karma, dışarıda (ör. teslim anında ayrı bir yerde) saklanmadıkça kötü niyetli değişikliği değil, yanlışlıkla bozulmayı yakalar.
- **Etmez:** veriyi yeniden hesaplamaz. Replay bütünlüğü doğrular; aracı yeniden çalıştırıp aynı sayıları üretmek için ayrıca soruyu tekrar sormak gerekir (aynı soru, aynı çıktı: 5 örnek soru iki turda birebir aynı çıkmıştır).

## 4. Hocanın "kanıt ve tekrarlanabilirlik" ölçütü için kontrol listesi

1. Soruyu sor, `data/analysis_logs/` içindeki dosyayı göster.
2. Dosyada `plan`, `evidence`, `state_history`, `raw_answer`, `violations` alanlarını göster.
3. `python scripts/replay_demo.py <dosya>` ile bozulmamış kaydın açıldığını ve değiştirilmiş kopyanın reddedildiğini göster.
4. Aynı soruyu tekrar sor: aynı `analysis_id`, aynı cevap.
