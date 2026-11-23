# 03 – Sektörde Geride Kalanlar / Yakalama (Catch-Up) Raporu

Analiz tarihi 2026-09-10, dönem 2024-01-01 – 2026-09-10. Eğitsel analizdir, yatırım tavsiyesi değildir.

## 1. Soru

Perakende grubunda (BIMAS, MGROS, MAVI) son 20 işlem gününde sektör medyanından en çok kim geride kaldı? 2024-01-01 – 2026-09-10 arasında benzer gerilikler 5, 10 ve 20 günde kapandı mı? Hisse `POTENTIAL_CATCH_UP_CANDIDATE` olarak etiketlenebilir mi?

## 2. Hangi sayı seti kullanılır (karar)

Repoda iki farklı hesap vardı ve birbirini yalanlıyor gibi görünüyordu: Task 1 belgesindeki **746 olay** ve örnek cevaplardaki **494 olay**. Bunlar **farklı tanımlardır**, biri diğerinin hatası değildir:

| | Task 1 motoru (`evaluate_laggard_catchup`) | Araç (`rank_sector_relative_strength`) |
|---|---|---|
| Kapsam | Tüm evren, her sektör kendi içinde | Soruda verilen hisse grubu |
| Geride kalma kuralı | Sabit: hisse −%3'ten fazla geride **ve** sektör ≥ +%5 | Uyarlanır: şu anki gerilimin yarısı kadar, en çok −%3 (burada %1,62) |
| Karşılaştırma | Sektör getirisi | Gruptaki diğerlerinin medyanı |
| Analiz tarihinde kesme | **Yok** (önbellekteki tüm veriyi kullanır) | Var |
| Bağımsız dönem sayımı | Yok | Var (20 işlem günü ara) |

**Bu raporun ana sayı seti araçtan gelir** (`reports/evidence/sector_retail.json`), çünkü: (1) soruya doğrudan cevap verir, (2) analiz tarihinde kesilmiştir, (3) bağımsız dönem sayısı verir, (4) modelin ve harness'in kullandığı yoldur.

Task 1 motoru yalnızca **tutarlılık kontrolü** olarak kullanılır (§5). Önemli düzeltme: Task 1 belgesindeki **746 olay ve %52,41 / %53,75 / %55,83** sayıları, fiyat önbelleğinin 2026-09-30'a kadar uzanan hâliyle üretilmişti; son sinyal günü 2026-09-11'e kadar çıkıyor. Yani **analiz tarihinden sonraki fiyatlar** kullanılmıştı (ileri bakış). Veri 2026-09-10'da kesilince 743 olay ve %52,7 / %54,1 / %55,9 çıkıyor.

## 3. Mevcut durum (2026-09-10, 20 günlük pencere)

| Hisse | 20g getiri | Gruptan fark (medyana göre) | Sıra |
|---|---|---|---|
| MAVI | %−5,06 | %−3,25 | 1 (en geride) |
| MGROS | %−1,81 | %0,00 | 2 |
| BIMAS | %+14,21 | %+16,02 | 3 |

Grup medyanı %−1,81. En geride kalan **MAVI**.

Not: evren sektör etiketinde MAVI "Retail / Apparel", BIMAS ve MGROS "Retail Trade" olarak ayrıdır. Bu üçlü, sorunun tanımladığı elle seçilmiş bir gruptur, evrenin resmi bir sektörü değildir.

## 4. Geçmişte benzer gerilikler kapandı mı?

Benzer geriliğin tanımı: gruptan en az %1,62 geride kalma. Giriş t+1'de, çıkış girişten 5/10/20 işlem günü sonra. "İsabet" = hissenin gruptaki diğerlerinin medyanını geçmesi.

| Ufuk | Olay | Tamamlanmayan (hariç) | Bağımsız dönem | İsabet | Yanlış pozitif | Medyan fazla getiri |
|---|---|---|---|---|---|---|
| 5 gün | 499 | 5 | 53 | %49,7 | %50,3 | %−0,02 |
| 10 gün | 494 | 10 | 53 | %51,8 | %48,2 | %+0,21 |
| 20 gün | 484 | 20 | 53 | %45,9 | %54,1 | %−0,60 |

**Dikkat:** 494 sayısı **10 günlük ufkun** olay sayısıdır, 5 günde 499, 20 günde 484 olay vardır.

Olaylar ardışık günlerde tekrar eder, bu yüzden bağımsız bilgi 53 dönemdir (olaylar değil). 53 dönemde bile isabet oranları yazı-tura düzeyinde (%46–52) ve ufka göre işaret değiştiriyor (20 günde %45,9 < %50).

## 5. Tutarlılık kontrolü: tüm evrende Task 1 motoru (2026-09-10'da kesilmiş)

| | Değer |
|---|---|
| Olay | 743 (12 hisse, 6 sektör) |
| Bağımsız dönem (28 takvim günü ara, hisse bazında) | 91 |
| 5 gün isabet (n=738) | %52,7, medyan fazla getiri %+0,18 |
| 10 gün isabet (n=737) | %54,1, medyan fazla getiri %+0,53 |
| 20 gün isabet (n=737) | %55,9, medyan fazla getiri %+1,24 |

Olaylar yalnızca 2 veya daha fazla hissesi olan 6 sektörde oluşur (Electric Utilities 167, Automotive 144, Basic Metals / Steel 144, Transportation / Airlines 108, Retail Trade 106, Telecommunications 74). Bu daha sıkı tanımla (sektör güçlü + hisse geride) isabet oranları biraz daha yüksek (%53–56) ama yine zayıf; 91 bağımsız dönemin istatistiksel gücü sınırlı.

## 6. Etiket kararı

**`POTENTIAL_CATCH_UP_CANDIDATE` verilemez.** Araç durumu `INVESTIGATE` verir. Gerekçeler:
- Etiket için gerilik en az −%3 olmalı: MAVI −%3,25 ile eşiğin hemen üstünde.
- 10 günlük isabet en az %55 olmalı: **%51,8**.
- Bağımsız dönem en az 10 olmalı: 53 (karşılanıyor).
- Hissenin temel verisi olmalı: **MAVI için temel veri kaydı yok**, bu yüzden "yapısal gerilik geçici gerilikten ayrılamıyor" uyarısı çıkıyor (temel veri yalnızca 5 hisse için var).

Makro açıklama: aracın uyarısı, FX/faiz etkisini görmek için `get_macro_context` çağrılması gerektiğini söyler; bu raporda yapılmadı.

## 7. Bankacılık sınırlaması

Banking sektöründe evrende yalnızca AKBNK var. Araç "en az 2 hisse gerekir" uyarısıyla `available: false` döner (`reports/evidence/sector_banking.json`). Bankacılık sektörü bu yöntemle **sıralanamaz**.

## 8. Sınırlamalar

- Evrende 24 sektör etiketinin 18'inde tek hisse var; sektör karşılaştırması yalnızca 6 sektörde mümkün.
- Önbellekte 2026-09-30'a kadar veri bulunur; analiz tarihinden sonrasını yalnızca araçlar keser. Tarihte kesmeyen yeni bir kod yazılırsa aynı ileri bakış hatası tekrarlanır.
- Sonuç "geride kalan hisse arayı kapatır" hipotezini **desteklemiyor**; destek varsa da zayıf ve ufka duyarlı.

Yeniden üretmek için: `python scripts/make_report_evidence.py` (araç sayıları) ve §5 için `evaluate_laggard_catchup` çağrısı, fiyat sözlüğü `.loc[:"2026-09-10"]` ile kesilmiş olarak.
