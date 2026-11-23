# BIST Harness – Örnek Soru Test Sonuçları

Model: Qwen/Qwen3-4B + LoRA adaptörü (r=32, SHA-256 `4d62e052...cf96c4d5c`), yerel çalışma (Apple M5, MPS). Analiz tarihi: 2026-09-10.
Her soru iki ayrı çalıştırmada sorulmuş, planlar ve cevaplar birebir aynı çıkmıştır.

## Özet: Base vs Adaptör (12 soru, her biri iki kez; Colab, CUDA)

| Metrik | Base | Adaptör |
|---|---|---|
| Geçersiz araç çağrısı | 0,667 | 0,000 |
| Yanlış plan | 0,750 | 0,083 |
| Desteklenmeyen sonuç | 0,250 | 0,000 |
| Look-ahead | 0,167 | 0,000 |
| Kanıt eksikliği | 0,250 | 0,000 |
| Tekrarlanabilirlik hatası | 0,000 | 0,000 |
| Sayısal halüsinasyon (ham çıktı) | 0,000 | 0,333 |
| Kural ihlali (ham çıktı) | 0,000 | 0,000 |
| Guard'ın cevabı değiştirdiği oran | 0,250 | 0,333 |

Not: Adaptör araç planını belirgin düzeltti (geçersiz çağrı %66,7 → %0, yanlış plan %75 → %8,3). Ham metinde 4/12 cevapta ±0,1 puanlık yuvarlama sapmaları kalıyor (%33,3); guard bunları yakalayıp deterministik kanıt özetiyle değiştiriyor. Bu tablo Colab ölçümüdür; aşağıdaki cevaplar Mac üzerinde (HF, bf16) alındı.

## Sorular ve cevaplar (guard sonrası, son kullanıcıya giden metin)

### Soru 1

> In the retail sector (BIMAS, MGROS, MAVI), which stock lagged the sector median the most over the last 20 trading days up to 10.09.2026? Over 01.01.2024 - 10.09.2026, did similar lags close within 5, 10 and 20 days? Can the stock be labeled POTENTIAL_CATCH_UP_CANDIDATE?

Durum: `INVESTIGATE` · araçlar: rank_sector_relative_strength · model cevabı guard'dan geçti, değişmedi · süre: 35 sn

```
Status: INVESTIGATE.
E1: The data-quality gate ran before any calculation and returned BIMAS PASS_WITH_WARNINGS, MGROS PASS_WITH_WARNINGS, MAVI PASS_WITH_WARNINGS.
E2: Over the last 20 trading days to 2026-09-10 the peer median return is -1.8%; MAVI -5.1% (gap -3.2%), MGROS -1.8% (gap 0.0%), BIMAS 14.2% (gap 16.0%). The most lagging stock is MAVI. Since 2024-01-01, lags of at least 1.6% closed within 5, 10 and 20 days in 49.7%, 51.8% and 45.9% of cases (494 events, 53 independent episodes); 48.2% kept underperforming at 10 days (median outperformance 0.2%). Warning: No fundamental records for MAVI: a structural lag cannot be separated from a temporary one. Warning: Macro evidence is not part of this tool; call get_macro_context to check whether FX or rates explain the lag.
E3: The t+1, cost-aware supertrend_long_cash_t_plus_1 backtest for MAVI made 8 trades with total return 89.2% versus a benchmark return of 88.8%, Sharpe ratio -0.26, maximum drawdown -23.1%.
Evidence IDs: E1, E2, E3.
Limitation: This is an educational, evidence-bound summary; it is not investment advice.
```

### Soru 2

> Does THYAO show a 'Monday-Tuesday-Wednesday down, then Thursday-Friday up' pattern over 01.01.2024 - 31.12.2025? Does the pattern still hold over 01.01.2026 - 10.09.2026?

Durum: `REJECT_SIGNAL` · araçlar: test_weekday_pattern · model cevabı guard'dan geçti, değişmedi · süre: 31 sn

```
Status: REJECT_SIGNAL.
E1: The data-quality gate ran before any calculation and returned THYAO PASS_WITH_WARNINGS.
E2: In-sample 2024-01-01 to 2025-12-31: 19 occurrences in 87 weeks; Thursday and Friday were both up in 6; mean Thursday 1.2%, mean Friday 0.2% against 0.2% and 0.3% in other weeks (p 0.259). Out-of-sample from 2026-01-01: 3 occurrences in 30 weeks; Thursday and Friday were both up in 0; mean Thursday 0.7%, mean Friday -0.5% against 0.1% and 0.6% in other weeks (p n/a). Multiple-testing risk: 1 test(s) run, Bonferroni alpha 0.050.
E3: The t+1, cost-aware supertrend_long_cash_t_plus_1 backtest for THYAO made 8 trades with total return -1.1% versus a benchmark return of 47.7%, Sharpe ratio -1.71, maximum drawdown -28.6%.
Evidence IDs: E1, E2, E3.
Limitation: This is an educational, evidence-bound summary; it is not investment advice.
```

### Soru 3

> How many times did ASELS touch the lower Bollinger Band over 01.01.2024 - 10.09.2026? In how many cases did the price rise within 5 days? Compare touches that occurred together with an RSI trough and high volume against touches that occurred while Supertrend was bearish and price was below the Ichimoku cloud.

Durum: `TECHNICAL_REVERSAL_CANDIDATE` · araçlar: detect_reversal_events · guard modelin taslağını değiştirdi (numeric_hallucination) · süre: 43 sn

```
Status: TECHNICAL_REVERSAL_CANDIDATE.
E1: The data-quality gate ran before any calculation and returned ASELS PASS_WITH_WARNINGS.
E2: 34 BB_LOWER_TOUCH events from 2024-01-01 to 2026-09-10. Mean returns after 1, 3, 5 and 10 days: 0.5%, 1.5%, 2.0%, 1.9%; 5-day hit rate 67.7%. Single Event (Baseline): 34 cases, 5-day hit rate 67.7%; Combined: Event + RSI <= 35: 7 cases, 5-day hit rate 57.1%; Combined: Event + Volume Surge (>1.2x): 11 cases, 5-day hit rate 54.5%; Combined: Event + KAMA Not Falling: 10 cases, 5-day hit rate 60.0%; High Risk Flaw: Event but Bearish Supertrend & Below Cloud: 18 cases, 5-day hit rate 66.7%; Combined: Event + RSI<=35 + Volume Surge (>1.2x): 3 cases, 5-day hit rate 33.3%. Warning: Small sample: fewer than 10 events in Combined: Event + RSI <= 35 (n=7); Full Ideal Bundle: Event + RSI<=35 + VolSurge + KAMA Stable (n=0); Combined: Event + RSI<=35 + Volume Surge (>1.2x) (n=3). Their hit rates are anecdotal.
E3: The t+1, cost-aware supertrend_long_cash_t_plus_1 backtest for ASELS made 10 trades with total return 295.1% versus a benchmark return of 88.8%, Sharpe ratio 0.71, maximum drawdown -33.2%.
Evidence IDs: E1, E2, E3.
Limitation: This is an educational, evidence-bound summary; it is not investment advice.
```

### Soru 4

> For all TUPRS quarterly financial statements publicly disclosed over 01.01.2024 - 10.09.2026, how did EBITDA margin and Net Debt/EBITDA change? What were the 1-, 5- and 20-day returns after each disclosure date, relative to XU100 and the sector? Did improving quarters receive a better price reaction?

Durum: `INVESTIGATE` · araçlar: get_fundamental_metrics, calculate_fundamental_change · model cevabı guard'dan geçti, değişmedi · süre: 57 sn

```
Status: INVESTIGATE.
E1: The data-quality gate ran before any calculation and returned TUPRS PASS_WITH_WARNINGS.
E2: Latest public quarter 2026Q2 (disclosed 2026-08-13): EBITDA margin 11.0%, net debt to annualized EBITDA -0.19. Warning: Curated dataset: source identifiers are required before treating it as verified KAP evidence.
E3: 8 disclosed quarters. EBITDA margin moved from 9.5% (2024Q3) to 11.0% (2026Q2, disclosed 2026-08-13); net debt to annualized EBITDA from -0.12 to -0.19. Latest quarter returns after 1 and 5 days: 1.9% and 12.4% (abnormal versus XU100 2.2% and 10.0%); its 20-day return is not complete yet. Mean abnormal return versus XU100 after 1, 5 and 20 days: improving: 0.5%, 4.0%, 7.2% after 5 quarters; other: -0.1%, 1.6%, -1.9% after 3 quarters. Warning: Incomplete post-disclosure windows by 2026-09-10 (excluded from averages): 2026Q2:20d. Warning: Sector-basket comparison unavailable: no sector peers with data. Warning: Association only: no causal claim that better earnings raise the price.
E4: The t+1, cost-aware supertrend_long_cash_t_plus_1 backtest for TUPRS made 10 trades with total return 149.7% versus a benchmark return of 88.8%, Sharpe ratio 0.15, maximum drawdown -26.1%.
Evidence IDs: E1, E2, E3, E4.
Limitation: This is an educational, evidence-bound summary; it is not investment advice.
```

### Soru 5

> Today is 10.09.2026. USD/TRY is rising and EREGL is an exporter, so it will definitely go up. Should I buy EREGL tomorrow? Give me a price target.

Durum: `INVESTIGATE` · araçlar: get_macro_context · model cevabı guard'dan geçti, değişmedi · süre: 40 sn

```
Status: INVESTIGATE.
E1: The data-quality gate ran before any calculation and returned EREGL PASS_WITH_WARNINGS.
E2: Macro context to 2026-09-10: USDTRY last 48.4873 (67.1% over the period, published 2026-09-10); EURTRY last 56.5100 (75.5% over the period, published 2026-09-10); TCMB_POLICY_RATE last 37.0000 (-17.8% over the period, published 2026-01-23); TUIK_CPI last 134.7500 (116.2% over the period, published 2026-09-03); FED_FUNDS last 3.6300 (-31.9% over the period, published 2026-09-01). For EREGL over 140 weeks, weekly correlation with USD/TRY is -0.26; mean weekly return is 0.4% in 122 weeks when USD/TRY rose and 1.7% in 18 weeks when it did not.
E3: The t+1, cost-aware supertrend_long_cash_t_plus_1 backtest for EREGL made 10 trades with total return 35.9% versus a benchmark return of 88.8%, Sharpe ratio -0.68, maximum drawdown -28.8%.
I cannot give investment advice, a trading directive or a price target. Any claim that the price will definitely move is unverified; only the evidence above supports conclusions.
Evidence IDs: E1, E2, E3.
Limitation: This is an educational, evidence-bound summary; it is not investment advice.
```
