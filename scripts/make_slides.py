"""Build reports/BIST_harness_sunum.pptx. Every number comes from reports/09 and reports/08 (check before presenting).

    pip install python-pptx && python scripts/make_slides.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parents[1] / "reports" / "BIST_harness_sunum.pptx"
NAVY, TEAL, GREY, LIGHT = RGBColor(0x14, 0x2B, 0x4D), RGBColor(0x1F, 0x8A, 0x8A), RGBColor(0x55, 0x5B, 0x66), RGBColor(0xEA, 0xF2, 0xF4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, left, top, width, height, lines, size=20, color=GREY, bold=False):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame
    box.word_wrap = True
    for i, line in enumerate(lines):
        para = box.paragraphs[0] if i == 0 else box.add_paragraph()
        para.text, para.space_after = line, Pt(8)
        para.font.size, para.font.color.rgb, para.font.bold = Pt(size), color, bold


def slide(title, bullets=(), note=None, size=22):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.15))
    bar.fill.solid(), setattr(bar.fill.fore_color, "rgb", NAVY), bar.line.fill.background()
    text(s, 0.6, 0.18, 12.1, 0.9, [title], size=32, color=WHITE, bold=True)
    if bullets:
        text(s, 0.7, 1.5, 11.9, 4.9, [f"•  {b}" for b in bullets], size=size)
    if note:
        text(s, 0.7, 6.55, 11.9, 0.7, [note], size=14, color=TEAL)
    return s


def table(s, rows, top=1.6, col_widths=None, size=16):
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(0.7), Inches(top), Inches(11.9), Inches(0.5 * len(rows)))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = shape.table.cell(r, c)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (LIGHT if r % 2 else WHITE)
            for p in cell.text_frame.paragraphs:
                p.font.size, p.font.bold = Pt(size), r == 0
                p.font.color.rgb = WHITE if r == 0 else GREY
    for i, w in enumerate(col_widths or []):
        shape.table.columns[i].width = Inches(w)


# 1
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(), setattr(bg.fill.fore_color, "rgb", NAVY), bg.line.fill.background()
text(s, 0.9, 2.0, 11.5, 2.0, ["BIST 100 Entegre Ajan Tabanlı Finansal Analiz ve Doğrulama Sistemi"], size=40, color=WHITE, bold=True)
text(s, 0.9, 4.2, 11.5, 1.6, ["CSE-481 Mühendislik Ekonomisi  ·  Analiz tarihi 2026-09-10",
                              "Eğitsel araştırma: yatırım tavsiyesi değildir, aracı kurumla bağlantısı yoktur."], size=20, color=RGBColor(0xBF, 0xD9, 0xDD))

slide("Problem ve tez", [
    "Dil modelleri finansal sorularda akıcı ama kanıtsız cevap verir: uydurma sayı, bilinmeyen tarih, \"kesin yükselir\".",
    "Tez: model hesap yapmaz. Hesabı deterministik araçlar yapar, harness süreci yönetir, backtest doğrular, kanıt kaydedilir, son kararı insan verir.",
    "Sistem AL/SAT demez; kanıta bağlı analitik durum üretir (INVESTIGATE, REJECT_SIGNAL, TECHNICAL_REVERSAL_CANDIDATE ...).",
])

slide("Mimari", [
    "soru → yerel model (Qwen3-4B + LoRA) → JSON araç planı → yönlendirici doğrular",
    "kalite kapısı → deterministik araçlar → kanıt kümesi (E1, E2, E3 ...)",
    "model kanıttan özet yazar → guard: sayı, tarih, üslup, tavsiye, kanıt kimliği denetimi",
    "karar günlüğü (SHA-256) → replay",
    "Zorunlu durum makinesi: her durum yalnızca izinli araçları çağırabilir. Bulut yapay zekâ API'si yok, model yerelde çalışır.",
], size=20)

slide("Veri", [
    "30 hisselik sabit evren, XU100, günlük fiyat (2024-01-02 →, analiz 2026-09-10'a kesik)",
    "Makro: USD/TRY, EUR/TRY, TCMB faizi, TÜFE, Fed faizi (yayın tarihine göre). Kaynak siteleri: evds2.tcmb.gov.tr, tuik.gov.tr, fred.stlouisfed.org",
    "Çeyreklik temel veri: 5 hisse, elle girilmiş örnek (KAP kaynak kimliği yok)",
    "Haber ve konuşma: araçlar hazır (get_news_events, get_speech_features), veri dosyaları boş; araç uyarı verir, içerik uydurmaz",
], size=20)

slide("Senaryo 1: Sektörde geride kalanlar", [
    "Perakende grubunda MAVI en geride (sektör medyanına göre %−3,25).",
    "Geçmiş benzer gerilikler 10 günde %51,8 kapandı (494 olay, 53 bağımsız dönem).",
    "Yakalama hipotezi desteklenmiyor → INVESTIGATE.",
])

slide("Senaryo 2: Hafta içi kalıbı (THYAO)", [
    "\"Pzt-Sal-Çar düşer, Per-Cum yükselir\": örnek içi 87 haftanın 19'unda görüldü, yalnızca 6'sında Per+Cum birlikte yükseldi (p = 0,26).",
    "Örnek dışı: 30 haftanın 3'ünde görüldü, hiçbirinde yükselmedi.",
    "Maliyet sonrası Çarşamba → Cuma: örnek içi %1,1, örnek dışı %−0,2.",
    "32 aday dizi için Bonferroni α = 0,0016 → REJECT_SIGNAL.",
])

slide("Senaryo 3: Teknik dönüşler (ASELS)", [
    "34 alt bant teması; 5 günde isabet %67,6 (genel eğilim %59,2).",
    "İdeal birleşik koşul hiç oluşmadı (n = 0); \"RSI dibi + hacim\" yalnızca 3 olay.",
    "Olay aileleri: Bollinger, RSI tepe/dip, KAMA, Supertrend, Ichimoku, destek/direnç; piyasa ve oynaklık rejimi kırılımı.",
    "TECHNICAL_REVERSAL_CANDIDATE, küçük örneklem uyarılı.",
])

slide("Senaryo 4 ve makro bağlam", [
    "TUPRS: iyileşen çeyrekler 20 günde daha iyi görünüyor, ama fark tek çeyreğe dayanıyor ve 5 günde ortadan kalkıyor. Tüm sayılar örnek veri → INVESTIGATE.",
    "Anormal getiri XU100'e ve (akranı varsa) sektör sepetine göre.",
    "Makro: USD/TRY %+67,1; EREGL ile haftalık korelasyon −0,26 (zayıf ve ters). \"Kur artınca ihracatçı yükselir\" desteklenmiyor.",
])

slide("Backtest ve benchmark", [
    "t+1 zamanlama, %0,10 komisyon, %0,05 kayma; A–E strateji varyantları, 30 hisse.",
    "Hiçbir varyant al-tut'u ya da XU100'ü sistematik geçmedi: medyan strateji getirisi ~%12, medyan al-tut %85,2, XU100 %+88,8.",
    "Al-tut'u geçen 4/30, XU100'ü geçen 2–3/30.",
    "Dürüst bulgu: güçlü yükselen piyasada basit kurallar al-tut'un gerisinde.",
])

s = slide("Yerel model ve guard (12 soru, ham taslak, Colab)")
table(s, [["Metrik", "Base", "Adaptör"], ["Geçersiz araç çağrısı", "%66,7", "%0"], ["Yanlış plan", "%75,0", "%8,3"],
          ["Desteklenmeyen sonuç", "%25,0", "%0"], ["Kural ihlali (tavsiye)", "%0", "%0"], ["Sayısal halüsinasyon (ham taslak)", "%0*", "%33,3"]],
      col_widths=[6.2, 2.85, 2.85])
text(s, 0.7, 5.0, 11.9, 1.5, ["*Base çoğu soruda geçerli plan üretemediği için bu adil bir kıyas değil.",
                              "Guard sapan taslakları deterministik özetle değiştirir. 12 soru: istatistiksel güven sınırlı."], size=16)

s = slide("A–E harness deneyi: gerçek model (15 soru, Colab T4)")
table(s, [["Ölçüt", "A", "B", "C", "D", "E"],
          ["Yanlış plan", "86,7%", "6,7%", "6,7%", "6,7%", "6,7%"],
          ["Sayısal halüsinasyon (son cevap)", "0%", "20%", "26,7%", "0%", "0%"],
          ["Kalite kapısı atlandı (bozuk veri)", "20%", "20%", "20%", "0%", "0%"],
          ["Guard'ın cevabı değiştirdiği", "0%", "0%", "0%", "26,7%", "26,7%"]],
      col_widths=[5.4, 1.3, 1.3, 1.3, 1.3, 1.3])
text(s, 0.7, 4.4, 11.9, 2.2, ["Kalite kapısı (D) üç bozuk veri sorusunun üçünü de durdurdu; kapısız B ve C bozuk veriyi analiz etti.",
                              "A'da araç planı yok, yanlış plan ölçütü anlamsız. 15 soru, sürüm başına tek ölçüm: eğilim gösterir, kanıt değildir."], size=18)

slide("Karar günlüğü ve replay", [
    "Her analiz SHA-256'lı JSON olarak kaydedilir; ham model taslağı ve guard ihlalleri de saklanır.",
    "Tek karakter değişirse replay reddeder.",
    "Tekrarlanabilirlik (Mac, 8 soru, ikişer kez): plan, kanıt, ham taslak ve son cevap 8/8 birebir aynı.",
    "Canlı gösterim: python scripts/replay_demo.py",
])

slide("Bu çalışmada bulunup düzeltilen hatalar", [
    "Önbellek analiz tarihinden sonrasını içeriyordu: ileri bakış hatası, sayılar düzeltildi (746 → 743 olay).",
    "Replay günlüğü guard'ın değişikliğini saklamıyordu.",
    "Sektör sorusunda makro kanıt eksikti; harness artık kendisi ekliyor.",
    "Hafta içi kalıbında maliyet ve çoklu test eksikti.",
])

slide("Sınırlamalar", [
    "Temel veri 5 hisse, elle girilmiş; haber ve konuşma dosyaları boş; TÜFE dosyası endeks düzeyi gibi görünüyor.",
    "Değerlendirme seti küçük (12 ve 15 soru); Colab 4-bit ve Mac bf16 küçük farklar verir.",
    "A–E yalnızca Colab'da koşuldu; base model A–E'de ölçülmedi.",
    "Teknik dönüşte yeni olay aileleri yalnızca ASELS için raporlandı; çoklu test düzeltmesi yok.",
], size=21)

slide("Sonuç", [
    "Kanıtsız sayı riski mimariyle sınırlanıyor: deterministik araçlar, zorunlu kalite kapısı, kanıt kimlikleri, guard ve değiştirilemez günlük.",
    "İnce ayarlı model araç planlamayı güvenilir yapıyor; metindeki sapmaları guard telafi ediyor.",
    "Dört senaryoda da savunulabilir bir işlem sinyali çıkmadı. Bu başarısızlık değil, dürüst bulgu.",
    "Tahmin ≠ kârlı strateji. Geçmiş kanıt ≠ yatırım tavsiyesi. Son karar insanındır.",
])

prs.save(OUT)
print(OUT)
