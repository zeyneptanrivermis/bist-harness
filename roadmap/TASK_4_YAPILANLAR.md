# TASK 4: A–E Harness Karşılaştırma Deneyleri

Task 4, aynı finansal soru setinde mimari kontrollerin güvenilirliğe etkisini ölçen tekrar üretilebilir deney çerçevesidir. Bu katman **finansal getiri tahmini yapmaz**; araç, quality gate, evidence, replay ve human review kontrollerinin hata önleme etkisini değerlendirir.

## Uygulama

- `src/experiments/catalog.py`: 12 sabit normal ve adversarial vaka.
- `src/experiments/policy.py`: A–E mimarilerinin araç ve kontrol sınırları.
- `src/experiments/providers.py`: Sağlayıcıdan bağımsız arayüz, kayıtlı cevap ve simülasyon sağlayıcısı.
- `src/experiments/evaluator.py`: Sayısal halüsinasyon, geçersiz araç, look-ahead, evidence eksikliği, doğrudan işlem tavsiyesi, state/quality/replay/human-review ihlali puanlama kuralları.
- `src/experiments/reporting.py`: JSONL, ham CSV, karşılaştırma CSV'si ve başarı oranı grafiği.
- `src/experiments/cli.py`: Ağ veya API anahtarı gerektirmeyen simülasyon koşusu.

## Çalıştırma

```powershell
python -m unittest discover -s tests -v
python -m src.experiments.cli --output-dir reports/ae_harness_demo
```

Bu CLI yalnızca deney hattının doğru işlediğini göstermek için `simulated-demo` sağlayıcısını kullanır. Akademik A–E karşılaştırması için sonraki adımda **tek bir sabit gerçek LLM modeli**, sıcaklık ayarı, sağlayıcı adaptörü ve API yetkilendirmesi seçilmelidir. Aynı vaka kataloğu ve prompt tüm sürümlerde değiştirilmeden korunmalıdır.
