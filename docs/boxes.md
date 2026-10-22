# Ramki słów w soc_ocr: stan na 2026-10-22 (commit db827bb + zmiany niezacommitowane)

Dokument wejściowy dla sesji, która ma zaprojektować **kalibrację umieszczania ramek** przez
nasz system. Opisuje, skąd biorą się wszystkie ramki w projekcie, jak je dziś mierzymy, co już
zmierzono i jakie są ograniczenia. Najpierw przeczytaj `CLAUDE.md`, bo obowiązują jego zasady
i decyzje.

---

## 1. Po co są ramki

1. **Zaczernianie (cel docelowy, poza obecnym zakresem).** Pozycje słów z kontraktu `ocr/v1`
   (`bbox_px`, `bbox_pt`) posłużą anonimizatorowi w repo `soc` do zamalowania danych na obrazie.
   Ramka, która nie obejmuje całego słowa, zostawia widoczne fragmenty liter. Ramka za duża
   zasłania sąsiednie słowa, co jest mniej groźne, ale psuje czytelność.
2. **Konsensus.** Drugi silnik (Tesseract) czyta wycinek wyznaczony ramką pierwszego (Paddle).
   Ucięta ramka oznacza ucięty wycinek, a z niego gorszy głos.
3. **Ewaluacja.** Ramki decydują, który odczyt porównujemy z którym słowem wzorca (grupowanie)
   i ile tekstu pokrywa ramka (miara pokrycia).
4. **Przegląd.** Wycinki PNG w kolejce przeglądu powstają z ramek.

**Cel kalibracji:** ramka każdego słowa obejmuje wszystkie piksele jego liter (≈100% „pełnych
ramek”) i nie zachodzi na litery sąsiednich słów. Ma to działać w czasie pracy systemu na
prawdziwych dokumentach, czyli **bez** wzorca, bez SVG i bez znajomości koloru tekstu.

---

## 2. Układ współrzędnych

- **Typ ramki:** `Box = tuple[float, float, float, float]` = `(x0, y0, x1, y1)` w pikselach
  obrazu (`ocr/engines/base.py:9`). Początek w lewym górnym rogu, y rośnie w dół, prostokąt
  równoległy do osi.
- **Ramki są zawsze prostokątami równoległymi do osi.** Tekst obrócony też dostaje taki
  prostokąt, opisany na obróconym słowie. Czworokąty silnika zamieniamy na prostokąty
  opisane (`_quad_box`, `ocr/engines/paddle_onnx.py:23`).
- **Wszystkie ramki w projekcie są w pikselach tego samego obrazu:** wzorca, Paddle,
  Tesseracta, ręcznych etykiet i rozstrzygnięć.
- **PDF (`ocr/pdf_regions.py`):** region ma `bbox_pt` w punktach strony (początek w lewym
  górnym rogu widocznego pola strony, y w dół) i jest renderowany z `render.dpi = 300`.
  - Przeliczenie: `bbox_px = bbox_pt * dpi / 72`, plus przesunięcie regionu na stronie.
  - Kontrakt `ocr/v1` ma mieć dla słowa i `bbox_px`, i `bbox_pt`.
  - `ocr/schema.py` jeszcze nie istnieje.

---

## 3. Ramki tekstu wzorcowego (SVG → Chromium)

Kod: `eval/svg_ground_truth.py`. Dane: `artifacts/datasets/gt/svg/` (714 SVG, 42 253 słowa,
pary JSON + PNG). Zbiory: `eval/dev_sets/svg_dev_v1.txt` (dev) i
`eval/dev_sets/svg_test_v1.txt` (test, nie stroić na nim).

### Jak powstaje ramka

1. **Render:** Playwright + Chromium, sieć zablokowana, fonty z `config/fonts.conf`
   (`FONTCONFIG_FILE`, np. Selawik zamiast Segoe UI). Skala: dłuższy bok ≈ 2000 px
   (`svg_gt.target_long_side_px`), w granicach `min_scale = 0.5` i `max_scale = 4.0`.
2. **Rozwinięcie `<use>`** w prawdziwe węzły (`_EXPAND_USE_JS`, linia 50), żeby tekst
   w symbolach dało się zmierzyć.
3. **Pomiar słowa** (`_WORDS_JS`, linia 101, okolice 186–200):
   - dla każdego ciągu `\S+` w węźle tekstowym tworzony jest `Range`,
   - ramką jest `Range.getBoundingClientRect()`.
   - **To ramka typograficzna, a nie atramentowa:**
     - w poziomie: od początku pierwszego znaku do końca ostatniego (szerokość przesunięcia,
       razem z marginesami bocznymi glifów),
     - w pionie: pełna wysokość linii fontu, czyli górna i dolna linia fontu, a nie piksele
       liter.
   - Przy okazji zapisywane są: `font` (px), `rotated`, `clipped` (wystaje poza
     `foreignObject`), `paint` (kolor liter), `occluded` (zasłonięte przez nieprzezroczysty
     kształt; test `elementsFromPoint`) oraz flagi sklejania.
4. **Sklejanie kawałków jednego słowa:**
   - `merge_fragments` (linia 266) skleja kolejne kawałki w pliku,
   - `merge_by_position` (linia 326) skleja kawałki porozrzucane po pliku, ale stykające się
     na rysunku (PowerPoint),
   - ramka sklejonego słowa to suma ramek kawałków,
   - progi: `glue_gap_em = 0.12` i `glue_gap_no_space_em = 0.3`.
5. **`drop_duplicates`** (linia 301): usuwa ten sam tekst w tym samym miejscu.
6. **Przeliczenie na piksele obrazu:** × skala, przycięcie do obrazu. Słowo z mniej niż
   `min_visible_fraction = 0.5` ramki w obrazie wypada. Słowo przycięte dostaje
   `clipped = True`.
7. **Widoczność (`is_visible`, linia 374):** kolor `paint` różni się od otoczenia ramki
   o więcej niż `svg_gt.ink_threshold = 48`. Bez jednolitego koloru liczy się ≥
   `ink_min_pixels` pikseli tuszu.

### Pola zapisane w JSON wzorca (`Word`, linia 216)

`text`, `bbox_px`, `font_px`, `rotated`, `legible` (font ≥ 8 px), `clipped`, `group` (ten
sam element `<text>` lub blok HTML).

- **Koloru `paint` nie zapisujemy w JSON**, choć jest dostępny w czasie renderu.
- `ignore_regions_px` to ramki obrazów rastrowych osadzonych w SVG. Odczyty w tych obszarach
  nie są oceniane.

### Słowa wzorca, które nie są oceniane

- `GtWord.ignore = not legible or clipped` (`eval/run_eval.py:181`).
- Słowa krótsze niż 2 znaki po normalizacji (`eval.min_scored_chars`, interpunkcja
  brzegowa obcinana).

### Zmierzone na dev: ramka typograficzna minus tusz

Jednostka: wysokość tuszu słowa. Wartość dodatnia oznacza, że ramka wzorca wystaje poza tusz.
Tusz to piksele różniące się o > 60 od mediany koloru brzegu ramki. 10 705 słów.

| bok | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| lewy | −0.03 | 0.00 | +0.03 | +0.07 | +0.11 |
| prawy | −0.04 | −0.01 | +0.02 | +0.04 | +0.09 |
| góra | −0.02 | +0.19 | +0.26 | +0.33 | +0.49 |
| dół | −0.02 | +0.02 | +0.23 | +0.30 | +0.39 |

Wnioski:
- W poziomie ramka wzorca jest prawie ciasna.
- W pionie ma około ¼ wysokości tuszu zapasu z góry i z dołu: interlinia oraz miejsce na
  wydłużenia i akcenty, których dane słowo nie ma.
- Wartości ujemne (p5) to przypadki, w których tusz wystaje poza ramkę typograficzną
  (kursywa, glify szersze niż przesunięcie) albo próg tuszu złapał obcą grafikę.

### Możliwość niewykorzystana: dokładna maska pikseli słowa we wzorcu

Wymaga dwóch zrzutów z Chromium: normalnego oraz takiego, w którym ukryto wszystko poza
tekstem.

- Maska słowa = piksele tekstu z drugiego zrzutu, leżące w ramce typograficznej słowa
  i widoczne na pierwszym zrzucie.
- Daje ciasną „ramkę atramentową” co do piksela, bez krawędzi kształtów, strzałek i teł.
- Rozwiązuje też problem progu tuszu, który dziś łapie obcą grafikę.

**Nie zaimplementowane.** To kandydat na „prawdę” do kalibracji i do miary.

---

## 4. Ramki wzorca z ręcznych etykiet i rozstrzygnięć (przybliżone)

### Ręczne etykiety (`eval/label_regions.py`)

Dotyczą tekstu, którego SVG nie zna: ikony, obrazy rastrowe w SVG.

- **Ramka regionu** to suma ramek odczytów silników, które zaproponowały dany tekst
  (`collect_regions`, linia 62).
- Wpisy trafiają do `artifacts/datasets/gt/manual/svg_regions.jsonl`, osobno dla każdego
  zbioru: dev `svg_regions`, test `svg_regions_test`.
- Przy wczytaniu (`load_gt`, `eval/run_eval.py:176`) tekst regionu jest dzielony na słowa,
  a ramka cięta wzdłuż x proporcjonalnie do liczby znaków (`_split_box`, linia 153).
- Słowo nie jest dodawane, jeśli wzorzec ma już ten sam tekst w pobliżu (`_near_box`,
  linia 145).

### Rozstrzygnięcia (`eval/adjudicate.py` → `artifacts/datasets/gt/manual/svg_corrections.jsonl`)

- Rekord grupowy (`_apply_group`, `eval/run_eval.py:123`): jeśli liczba słów się zgadza,
  zmieniany jest tylko tekst i ramki wzorca zostają. W przeciwnym razie nowe słowa dostają
  ramkę grupy pociętą `_split_box`.
- Status `unreadable` dodaje ramkę do regionów wyłączonych z oceny.

**Dla kalibracji:** ramki z tych dwóch źródeł są geometrycznie przybliżone, bo pochodzą
z silnika albo z cięcia proporcjonalnego. **Nie używaj ich jako prawdy o położeniu.** Słowa
z SVG da się od nich odróżnić tylko pośrednio: nie mają `font_px`, bo powstają jako `GtWord`
w `load_gt`. Jeśli kalibracja potrzebuje tego rozróżnienia, trzeba dodać znacznik źródła do
`GtWord`.

---

## 5. Ramki PaddleOCR (silnik główny; to one trafiają do wyniku)

Kod: `ocr/engines/paddle_onnx.py` (`paddle_openvino.py` zmienia tylko backend). Biblioteka:
RapidOCR 3.10, PP-OCRv6 medium, `.venv/lib/python3.11/site-packages/rapidocr/`. Konfiguracja:
`[paddle]` w `config/default.toml`.

### Ścieżka `read(img)` (detekcja + rozpoznanie), używana w konsensusie

1. **Detekcja (DB):**
   - obraz o krótszym boku skalowanym do ≥ `det_limit_side_len = 736` (`limit_type "min"`),
     dłuższy bok ≤ `max_side_len = 2000`,
   - mapa prawdopodobieństwa progowana `det_thresh = 0.3`, kontury ze średnim wynikiem ≥
     `det_box_thresh = 0.5`,
   - każdy kontur → minimalny obrócony prostokąt → **poszerzenie `unclip`**
     (`ch_ppocr_det/utils.py:262`):
     `distance = pole · det_unclip_ratio / obwód`, przy `det_unclip_ratio = 1.6`.
   - Wynik: **czworokąt całej linii tekstu**, nie słowa.
2. **Wycinek linii** (`utils/process_img.py:72`, `get_rotate_crop_image`):
   - `warpPerspective` czworokąta do prostokąta,
   - jeśli wysokość/szerokość ≥ 1.5, wycinek jest obracany `np.rot90` (tekst pionowy).
3. **Klasyfikator kierunku** (`use_cls = true`): może obrócić wycinek o 180°.
4. **Rozpoznanie (CTC):** ciąg kolumn. Dla każdego znaku RapidOCR zapamiętuje numer kolumny,
   w której znak „wypalił” (`word_cols`), i pewność.
5. **Ramki znaków** (`cal_rec_boxes/main.py`, `cal_ocr_word_box` linia 76, `calc_box` linia
   145), z `return_single_char_box=True`:
   - `avg_col_width = szerokość_ramki_linii / line_txt_len`,
   - `avg_char_width` = średnia po słowach z `(ostatnia_kolumna − pierwsza_kolumna) ·
     avg_col_width / (liczba_znaków − 1)`,
   - **ramka znaku = środek jego kolumny ± avg_char_width/2**, na pełną wysokość linii,
   - ramki posortowane po x, nakładki rozdzielone po połowie (`adjust_box_overlap`).
   - **Nie mierzy się tu żadnych pikseli.** Pozycja wynika z kolumny CTC, a szerokość jest
     średnią dla linii. Kolumna CTC to miejsce rozpoznania znaku, a nie środek glifu.
6. **Powrót do współrzędnych obrazu** (`reverse_rotate_crop_image`, linia 202):
   - odwrotna perspektywa,
   - dla linii pionowych obrót o −90°.
   - **Obrót o 180° z klasyfikatora nie jest tu uwzględniany.** To hipoteza do sprawdzenia
     przy zamianach ramek, patrz §9.
7. **Nasz podział na słowa** (`line_to_words`, `paddle_onnx.py:32`):
   - znaki linii parowane z tekstem po kolei,
   - spacje w tekście wyznaczają granice słów,
   - **ramka słowa = suma ramek jego znaków**,
   - pewność słowa = najsłabszy znak,
   - jeśli znaki nie zgadzają się z tekstem, cała linia staje się jednym słowem z sumą ramek.

### Ścieżka `recognize(img, boxes)` (bez detekcji, na zadanych ramkach)

Kod: linia 156. Wycinek ramki + `recognize_pad_px = 4`.

- **Pozycje znaków liczone są inaczej niż w `read`:**
  `kolumna · (szerokość_wycinka / line_txt_len)`, liniowo, bez `avg_char_width`.
- W konsensusie używana tylko do ponownego odczytu (sam tekst, ramki pomijane).

### Wniosek

Ramka słowa Paddle to **oszacowanie z kolumn CTC**. Detektor zna dobrze tylko zasięg całej
linii, a i to z poszerzeniem `unclip`.

---

## 6. Ramki Tesseracta (silnik drugi)

Kod: `ocr/engines/tesseract.py`.

- **`read(img)`** (hOCR, `psm 11`): ramki słów Tesseracta są **ciasne do pikseli liter**
  (`parse_hocr`, linia 20; `hocr_char_boxes=1` daje też ramki znaków). W konsensusie ta
  ścieżka działa tylko przy `secondary_detection = true`, a domyślnie jest wyłączona.
- **`recognize_each(img, boxes)`** (konsensus):
  - wycinek ramki Paddle + `recognize_pad_px = 4` (`_variants`, linia 81), `psm 7`,
    równolegle w 6 procesach,
  - ramki z hOCR przeliczane z powrotem na współrzędne obrazu (`_read_box`, linia 104),
  - przy wariancie obróconym (dziś wyłączony, `rotate_min_aspect = 0`) ramką jest ramka
    wejściowa.
- **Konsensus używa z wyniku Tesseracta tylko tekstu i pewności** (`_joined`). Jego ciasne
  ramki są wyrzucane. To wolne źródło informacji o tuszu w obrębie ramki Paddle ± 4 px. Nie
  sięga dalej niż 4 px poza ramkę Paddle.

---

## 7. Konsensus (`ocr/consensus.py`)

- **Ramka w wyniku = ramka Paddle** (`p.bbox_px` we wszystkich gałęziach `_vote`, linia
  ~147). Tesseract i ponowny odczyt wpływają tylko na tekst i status.
- **Ponowny odczyt** (`_reread`, linia 130): wycinek ramki Paddle + `reread_pad_px = 6`,
  powiększony 2×, a oba silniki czytają go jako całość.
- **Margines 4 px i 6 px częściowo ratuje odczyt przy uciętej ramce, ale nie ratuje ramki
  w wyniku.**
- `secondary_edge_junk` istnieje, bo wycinek z marginesem łapie krawędzie kształtów
  („Azure |”). Każde poszerzanie ramek zwiększy to zjawisko. Kalibracja ramek do wyniku może
  wymagać osobnej ramki do czytania i osobnej do wyniku.

---

## 8. Jak ewaluacja używa ramek (`eval/metrics.py`, tryb `matching = "align"`)

### Grupowanie (`_groups`, linia 291)

- Słowo wzorca i odczyt są w jednej grupie, gdy `przecięcie ≥ group_min_overlap (0.2) ·
  pole mniejszej ramki` (`_linked`, linia 265).
- Sąsiednie słowa wzorca jednej linii są łączone (`_same_line`, linia 270):
  - nakładają się w poprzek linii o ≥ ½ mniejszego wymiaru,
  - odstęp ≤ `group_line_gap (0.8)` × ten wymiar,
  - linie poziome tylko dla słów szerszych niż wysokie, pionowe tylko gdy oba są wyższe
    niż szerokie.
- Teksty w grupie są porównywane znak po znaku. Kolejność czytania wybierana jest osobno dla
  wzorca i odczytu spośród 6 wariantów (`_reading_order`, linia 320), bo Paddle zamienia
  ramki na liniach obróconych.
- **Skutek: przesunięcie ramki nie liczy się jako błąd odczytu.** Ramki mierzy osobna miara.

### Miara pokrycia (`_ink_coverage`, linia 355)

Dla każdego ocenianego, wykrytego słowa wzorca:
1. W ramce wzorca liczymy **tusz**: piksele różniące się od mediany koloru brzegu ramki
   o > `box_ink_threshold = 60` w dowolnym kanale.
2. `box_coverage` = odsetek tuszu leżący w sumie ramek odczytów, które złożyły się na to
   słowo (`readers` z wyrównania).
3. „Pełna” ramka: `box_coverage ≥ box_full_coverage (0.99)`.
4. Raport: `box_full_coverage` (odsetek pełnych) i `box_mean_coverage`.

**Ograniczenia miary:**
- **Tusz z progu może zawierać obcą grafikę:** krawędź kształtu, strzałkę, sąsiednią linię
  tekstu w zapasie pionowym ramki wzorca. Zaniża to pokrycie bez winy silnika. Dokładne
  maski z §3 by to usunęły.
- **Nie mierzymy nadmiaru:** ile ramka silnika zachodzi na litery sąsiednich słów lub
  wychodzi poza słowo. Kalibracja, która tylko poszerza, wygra na tej mierze za darmo.
  **Trzeba dodać miarę nadmiaru** (np. odsetek tuszu innych słów wzorca wewnątrz ramki
  odczytu, albo IoU z ramką atramentową).
- Słowo czytane przez kilka odczytów dostaje sumę ich ramek. To poprawne dla zaczerniania,
  ale nie ocenia ramek pojedynczo.
- Miara liczy się tylko dla słów wykrytych. Słowa pominięte mierzy kompletność detekcji.

### Pozostałe użycia ramek w ewaluacji

- Krótki pojedynczy token odczytu liczy się tylko, gdy leży w ≥ 50% w ramce ocenianego słowa.
- Odczyt bez wzorca w pobliżu jest liczony jako „extra”, chyba że leży w ≥ 50%
  (`match_min_overlap`) w regionie ignorowanym.

---

## 9. Zmierzone

### Test (agregat; zgodnie z zasadą bez analizy pojedynczych słów testu)

Przebieg `*consensus_test1-first`, wzorzec przed rozstrzyganiem testu:
- pełne ramki **32.6%**,
- średnie pokrycie **94.8%**.

### Dev, szczegółowo

Przebieg `*consensus_v2-edgejunk-dev1`, zbiór `svg_dev_v1`, 10 705 słów z pomiarem pokrycia.

Jednostka: wysokość tuszu słowa. Wartość + oznacza, że ramka silnika **ucina** tusz z tej
strony, a − oznacza zapas.

**Słowa poziome (n = 10 484):** pełne 31.2%, średnie pokrycie 0.949. Słowo z kilku odczytów:
1.3%.

| bok | p5 | p25 | p50 | p75 | p95 | ucina (> 0.02) |
|---|---|---|---|---|---|---|
| lewy | −0.12 | 0.00 | **+0.08** | +0.17 | +0.31 | **70.3%** |
| prawy | −0.29 | −0.15 | −0.07 | 0.00 | +0.12 | 22.4% |
| góra | −0.56 | −0.33 | −0.25 | −0.18 | +0.07 | 7.9% |
| dół | −0.50 | −0.33 | −0.23 | −0.14 | −0.03 | 2.4% |

Wśród niepełnych najgorszy bok to: lewy 85%, prawy 10%, góra 4%, dół 1%.

**Ten sam pomiar w szerokościach znaku**, dla słów poziomych z jednym odczytem, według
pozycji słowa w linii Paddle (linia = odczyty o identycznych y0 i y1):

| pozycja | n | lewy p5 / p50 / p95 | prawy p5 / p50 / p95 | mediana px lewy / prawy | mediana wys. tuszu |
|---|---|---|---|---|---|
| jedyne w linii | 4551 | −0.24 / **+0.12** / +0.49 | −0.47 / −0.12 / +0.20 | +1 / −1 | 21 px |
| pierwsze | 2268 | −0.21 / **+0.16** / +0.50 | −0.43 / −0.08 / +0.20 | +2 / −1 | 19 px |
| środkowe | 1371 | −0.19 / **+0.17** / +0.53 | −0.57 / −0.11 / +0.22 | +2 / −1 | 18 px |
| ostatnie | 2153 | −0.19 / **+0.17** / +0.49 | −0.46 / −0.13 / +0.23 | +2 / −1 | 19 px |

**Słowa obrócone lub pionowe (n = 221):** pełne 27.6%, średnie pokrycie **0.639**. Słowo
z kilku odczytów: 8.1%.
- W poprzek linii (lewy i prawy) ramki mają zwykle zapas.
- Wzdłuż linii (góra i dół): p5/p95 to około **±7–8 wysokości tuszu**. Ramka bywa przesunięta
  o całe słowa, czyli zamieniona z sąsiednim słowem linii.
- Wśród niepełnych najgorszy bok to: góra 48%, dół 40%.

### Interpretacja

1. **Systematyczne przesunięcie w prawo:** ramki słów poziomych są przesunięte o około
   0.12–0.17 szerokości znaku (1–2 px przy tekście ~20 px). Lewa krawędź ucina pierwszą
   literę w 70% słów, a po prawej zostaje zapas. Wygląda to na stałe przesunięcie kolumn CTC
   względem glifów (§5, krok 5). **Pojedyncza poprawka stała naprawi medianę, ale nie
   rozrzut:** p5–p95 to około ±0.35 szerokości znaku, czyli kilka px.
2. **W pionie ramki Paddle mają zapas,** bo mają wysokość linii z detektora po `unclip`.
   Ucinanie góry i dołu jest rzadkie.
3. **Słowo jedyne w linii też jest ucięte z lewej,** choć detektor zna zasięg całej linii.
   Ramka słowa to suma komórek znaków, a nie ramka detektora. Dla takich słów ramka linii z
   detektora jest od ręki lepszym kandydatem, do sprawdzenia.
4. **Tekst obrócony** to osobny problem: przypisanie ramek do słów wzdłuż linii jest błędne
   (zamiany). Najpierw trzeba to naprawić, dopiero potem kalibrować. Hipotezy:
   - obrót 180° z klasyfikatora nieuwzględniony w `reverse_rotate_crop_image`,
   - kierunek obrotu `rot90` przy tekście czytanym z dołu do góry kontra z góry do dołu.

Tekst obrócony to około 2% słów dev.

---

## 10. Dane i narzędzia do pracy

- **Zapisane predykcje** z ramkami: `artifacts/eval_runs/<run>/predictions.jsonl`. Jeden
  wiersz na obraz: `json`, `seconds`, `words[]` z `text`, `bbox_px`, `conf`, `char_conf`,
  `status`, `reason` i `votes`.
  - Raporty: `eval/reports/<run>.json`.
  - Dev: `*consensus_v2-edgejunk-dev1`. Test: `*consensus_test1-first` (tylko agregaty).
- **Ponowna ocena bez uruchamiania silników:**
  ```
  .venv/bin/python -m eval.run_eval --from-run <run> --images eval/dev_sets/svg_dev_v1.txt
  ```
  `ReplayEngine` w `eval/run_eval.py:198`. Pozwala testować przekształcenia ramek na
  zapisanych predykcjach w sekundy. Potrzebny jest wtedy wariant replay z funkcją
  poprawiającą ramki, a obraz jest dostępny w `load_gt`.
- **Pełny przebieg silników na dev:** kilka minut na laptopie (Paddle OpenVINO + Tesseract
  ×6 procesów). Zapisuj temperaturę i throttling w raporcie, jeśli przebieg jest długi.
- **Skrypt statystyk z §9:** w Załączniku. Uruchamiać z `PYTHONPATH=.`.
- **Testy:** `tests/` (172 testy, pytest). Każda nowa funkcja ma mieć test.

---

## 11. Ograniczenia i zasady dla projektu kalibracji

- **Musi działać bez wzorca:** na samym obrazie i wynikach silników, lokalnie, bez sieci.
  Kolor tekstu nie jest znany. Na prawdziwych dokumentach tło bywa gradientem, zrzutem
  ciemnym albo zdjęciem.
- **Parametry w `config/default.toml`, nie w kodzie.** Determinizm.
- **Stroić tylko na dev (`svg_dev_v1`).** Test (`svg_test_v1`) wyłącznie do końcowego
  pomiaru, i tylko agregaty. Nie oglądać pojedynczych błędów testu.
- **Nie pogarszać odczytu.** Jeśli poprawione ramki mają też zasilać Tesseracta
  i ponowny odczyt, zmierzyć wpływ na CER i odsetek przeglądu. Szersza ramka oznacza więcej
  śmieci na krawędziach. Możliwy wariant: poprawiać tylko ramkę w wyniku.
- **Raportować obie strony:** pokrycie (niedomiar) i nadmiar (wejście na sąsiednie słowa).
  Osobno dla tekstu poziomego i obróconego.
- **Zbiór SVG to głównie diagramy wektorowe** (Azure Architecture Center, Wikimedia itp.),
  wyrenderowane czysto. Skany, JPEG i zrzuty ekranu z rozmyciem nie są tu dobrze
  reprezentowane. Metoda oparta na tuszu musi być odporna na szum i kompresję. Do
  sprawdzenia później na danych z `synth/` (jeszcze nie istnieje).

## 12. Kierunki do rozważenia (nie są decyzjami)

1. **Dokładne maski tuszu we wzorcu** (§3): uczciwa prawda do kalibracji i do miary nadmiaru.
2. **Miara nadmiaru** w `eval/metrics.py` obok pokrycia.
3. **Dociąganie ramki do tuszu w czasie pracy** (snap-to-ink):
   - lokalnie ustalić tło (brzeg ramki linii z detektora) i tusz,
   - przyciąć puste brzegi,
   - rozszerzać krawędź, dopóki przecina ją spójna plama tuszu,
   - ograniczenia: przerwa większa niż odstęp międzyliterowy, granice sąsiednich słów
     (środek przerwy), ramka linii z detektora jako górna granica.
4. **Prosta poprawka stała** (przesunięcie lewej krawędzi o k·szerokość znaku) jako punkt
   odniesienia dla metod bardziej złożonych.
5. **Słowa jedyne w linii:** ramka linii z detektora, ścieśniona do tuszu.
6. **Ramki Tesseracta z hOCR** (wycinek ramki Paddle ± 4 px) jako dodatkowy sygnał
   o zasięgu tuszu.
7. **Tekst obrócony:** najpierw naprawić mapowanie ramek wzdłuż linii (§9, punkt 4), na
   przykład własne mapowanie kolumn CTC z uwzględnieniem obrotu z klasyfikatora.
8. **Tekst natywny PDF** (osobny temat): w diagramach wektorowych z zachowanym tekstem pdfium
   podaje pozycje znaków wprost, bez OCR. Te ramki też będą typograficzne, a nie
   atramentowe (§3).

---

## Załącznik: skrypt statystyk ramek (§9)

Zapisz np. jako `scratch/box_stats.py` i uruchom:
`PYTHONPATH=. .venv/bin/python scratch/box_stats.py eval/dev_sets/svg_dev_v1.txt consensus_v2-edgejunk-dev1`

```python
"""Aggregate box geometry on the dev set: GT typographic box vs ink, engine boxes vs ink."""
import glob, sys
from pathlib import Path
import numpy as np
from eval.metrics import PredWord, evaluate_image
from eval.run_eval import ReplayEngine, load_gt, load_index, images_from
from ocr.config import load_config

cfg = load_config().eval
gt_dir = Path("artifacts/datasets/gt/svg")
rows = images_from(Path(sys.argv[1]), load_index(gt_dir))
eng = ReplayEngine(Path(glob.glob(f"artifacts/eval_runs/*{sys.argv[2]}")[0]), Path("eval/reports"), False)

def ink_mask(a, box, thr):
    h, w = a.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1])); x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x1 - x0 < 2 or y1 - y0 < 2: return None, None
    c = a[y0:y1, x0:x1].astype(int)
    border = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    m = np.abs(c - np.median(border, axis=0)).max(axis=2) > thr
    return m, (x0, y0)

S = {k: [] for k in ("hor", "rot")}
gt_margin = []
POS = {}
nwords = 0
for row in rows:
    if row["json"] not in eng.preds: continue
    img, gt, regions = load_gt(gt_dir, row)
    a = np.asarray(img)
    words, _ = eng.read_row(row)
    pred = [PredWord(w.text, w.bbox_px, w.conf, w.status) for w in words]
    res = evaluate_image(gt, pred, cfg, regions, image=a)
    for w in res.words:
        if w["box_coverage"] is None: continue
        g = gt[w["gt"]]
        m, (ox, oy) = ink_mask(a, g.bbox, cfg.box_ink_threshold)
        if m is None or not m.any(): continue
        ys, xs = np.nonzero(m)
        ix0, iy0, ix1, iy1 = xs.min() + ox, ys.min() + oy, xs.max() + 1 + ox, ys.max() + 1 + oy
        ih = iy1 - iy0; iw = ix1 - ix0
        vertical = (g.bbox[3]-g.bbox[1]) > 1.5*(g.bbox[2]-g.bbox[0])
        L = min(ih, iw)
        gt_margin.append(((ix0-g.bbox[0])/L, (g.bbox[2]-ix1)/L, (iy0-g.bbox[1])/L, (g.bbox[3]-iy1)/L))
        rb = [pred[i].bbox for i in w["readers"]]
        ux0, uy0 = min(b[0] for b in rb), min(b[1] for b in rb); ux1, uy1 = max(b[2] for b in rb), max(b[3] for b in rb)
        # positive = engine box cuts into ink on that side; negative = slack
        d = ((ux0-ix0)/L, (ix1-ux1)/L, (uy0-iy0)/L, (iy1-uy1)/L)
        S["rot" if (g.rotated or vertical) else "hor"].append((w["box_coverage"], len(rb), *d))
        if not (g.rotated or vertical) and len(rb) == 1:
            p = pred[w["readers"][0]]
            line = sorted((q for q in pred if q.bbox[1] == p.bbox[1] and q.bbox[3] == p.bbox[3]), key=lambda q: q.bbox[0])
            pos = "single" if len(line) == 1 else "first" if line[0] is p else "last" if line[-1] is p else "middle"
            cw = iw / max(1, len(w["text"]))
            POS.setdefault(pos, []).append((d[0]*L/cw, d[1]*L/cw, (ux0-ix0), (ix1-ux1), ih))
        nwords += 1

def pct(v, ps=(5, 25, 50, 75, 95)): return " ".join(f"{np.percentile(v,p):+.2f}" for p in ps)
print("words", nwords)
gm = np.array(gt_margin)
print("GT typographic box minus ink (in ink-height units; p5 p25 p50 p75 p95):")
for k, n in enumerate(("left", "right", "top", "bottom")): print(f"  {n:6} {pct(gm[:,k])}")
for key, v in S.items():
    if not v: continue
    v = np.array(v)
    print(f"[{key}] n={len(v)} full>=0.99: {np.mean(v[:,0]>=0.99):.1%}  mean cov {v[:,0].mean():.3f}  multi-reader {np.mean(v[:,1]>1):.1%}")
    print("  engine box vs ink (+ = cuts ink, - = slack), p5 p25 p50 p75 p95:")
    for k, n in enumerate(("left", "right", "top", "bottom")):
        print(f"    {n:6} {pct(v[:,2+k])}   cuts>0.5px-equiv: {np.mean(v[:,2+k]>0.02):.1%}")
    low = v[v[:,0] < 0.99]
    if len(low):
        sides = np.argmax(low[:, 2:], axis=1)
        print("  among not-full: worst side", {n: f"{np.mean(sides==k):.0%}" for k, n in enumerate(("left","right","top","bottom"))})
print("horizontal, one reader, by word position in Paddle line; in char widths (cut + / slack -), and px:")
for k, v in POS.items():
    v = np.array(v)
    print(f"  {k:7} n={len(v):5} left {pct(v[:,0])} | right {pct(v[:,1])} | left px med {np.median(v[:,2]):+.1f} right px med {np.median(v[:,3]):+.1f} ink h med {np.median(v[:,4]):.0f}")
```

Uwaga: „tusz” w skrypcie liczony jest progiem względem brzegu ramki wzorca, więc dla słów
z obcą grafiką w ramce wyniki są zaszumione (§8, ograniczenia).
