# soc_ocr — moduł OCR dla pipeline'u `soc` (instrukcje dla Claude Code)

Ten plik opisuje cel, ustalone decyzje i podział pracy. Czytaj go na początku każdej sesji.
Na pytania „dlaczego tak” odpowiada sekcja *Decyzje*. Nie zmieniaj tych decyzji bez zgody użytkownika.

## Cel

Wyłuskać **tekst z obrazów i diagramów w plikach PDF** (zrzuty konsol, diagramy architektury,
skany, zdjęcia), razem z **pozycją każdego słowa i pewnością odczytu**. Wynik trafi później do
istniejącego silnika anonimizacji (pakiet `anonymizer/` w repo `soc`).

Repo: https://github.com/miszkaga/soc_ocr (osobne od `soc`; integracja przez kontrakt JSON).

Poza zakresem na tym etapie:
- **anonimizacja.** Nie dotykaj repo `soc` ani pakietu `anonymizer/`,
- zaczernianie obrazów,
- obsługa DOCX/PPTX. Moduł ma się do niej nadawać później, ale teraz robimy tylko PDF.

Człowiek jest w pętli na końcu: odczyty niepewne nie są zgadywane, tylko trafiają do kolejki przeglądu.

## Decyzje (ustalone z użytkownikiem)

1. **Główny silnik to klasyczna detekcja + rozpoznawanie** z pewnością dla każdego znaku.
   VLM nie jest źródłem prawdy, bo halucynuje i nie podaje pozycji słów. Na późniejszym etapie
   może być co najwyżej weryfikatorem regionów niepewnych.
2. **Wszystko działa lokalnie.** Żadnych zewnętrznych API, żadnych chmurowych usług OCR.
   Prawdziwe dokumenty nigdy nie opuszczają laptopa.
3. **Dwa niezależne silniki i konsensus.** Zgoda + wysoka pewność → `accepted`.
   Rozbieżność → ponowny odczyt (inna skala lub obróbka). Dalej rozbieżne → `uncertain` → przegląd.
4. **Silniki za wymiennym interfejsem.** Kandydaci do pomiaru: Tesseract 5 (`pol+eng`),
   PaddleOCR przez ONNX Runtime, PaddleOCR przez OpenVINO, docTR. Wybór po pomiarach, nie z góry.
5. **Dane treningowe i testowe:** obrazy publiczne z internetu (z zapisaną licencją) oraz
   generowane zrzuty z danymi syntetycznymi. Prawdziwe dokumenty nie trafiają do treningu.
   Zestaw musi zawierać trudne ciągi: e-maile, hosty, IP, identyfikatory, liczby i polskie znaki
   (l/1/I, O/0, rn/m, ą/a).
6. **Douczanie tylko tam, gdzie pomiar pokaże zysk.** Liczy się udział typu obrazu × liczba
   błędów gotowego modelu na tym typie. Jeden model rozpoznający douczany na mieszance
   ważonej rzeczywistymi proporcjami typów.

## Maszyny i podział pracy

Sprawdź, na której maszynie jesteś (`hostname`, system), zanim cokolwiek uruchomisz.

| | Laptop (praca główna) | Desktop (GPU) |
|---|---|---|
| Nazwa | `majkelo-aspire-a14-52m` | `desktop-os6vi2p` |
| System | Linux | Windows 11 (PowerShell) |
| CPU | Intel Core Ultra 7 256V, 8 rdzeni, AVX2 + AVX-VNNI, NPU + Arc 140V | Intel i7-7700K, 4 rdzenie / 8 wątków |
| RAM | 16 GB (wlutowane, zwykle ~6 GB wolne) | 16 GB |
| GPU | brak NVIDIA, możliwe OpenVINO (CPU / iGPU / NPU) | GTX 1050 Ti 4 GB (Pascal, sm_61), sterownik 582.28 |
| Zadania | inwentaryzacja PDF, wycinanie regionów, generowanie danych, pomiary na CPU / OpenVINO, ewaluacja, integracja, **wszystko z prawdziwymi dokumentami** | **douczanie modeli**, pomiary wnioskowania na GPU, ciężkie przebiegi generowania danych |

Zasady:
- **Zadania oznaczone `[GPU]` uruchamiaj tylko na desktopie.** Na laptopie przygotuj
  konfigurację i skrypt, a uruchomienie zostaw desktopowi.
- **Prawdziwe dokumenty tylko na laptopie.** Na desktop trafiają wyłącznie dane publiczne lub syntetyczne.
- **Laptop:** przy długich przebiegach zasilanie podłączone, ciężkie aplikacje zamknięte
  (system wchodzi w swap). Przy długim obciążeniu procesor zwalnia z powodu temperatury,
  zapisuj to w raportach.
- **Desktop:** przed treningiem zamknij Teams, Discorda, Steama i Edge, bo zajmują pamięć karty
  (około 0,6 GB z 4 GB). Trenuj w **FP32**, bo FP16 jest na Pascalu bardzo wolne.

## Środowisko

Python 3.11 na obu maszynach. Wersje przypinamy w `ocr/requirements-*.txt` i nie aktualizujemy bez pomiaru.

**Laptop (Linux)**
```bash
sudo apt install tesseract-ocr tesseract-ocr-pol tesseract-ocr-eng poppler-utils
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r ocr/requirements-cpu.txt     # pypdfium2, pdfplumber, opencv-python-headless, pytesseract,
                                            # onnxruntime, openvino, rapidfuzz, playwright, pytest
python -m playwright install chromium       # renderowanie syntetycznych zrzutów
```

**Desktop (Windows)**
- **Środowisko na krótkiej ścieżce**, np. `C:\ocr\.venv`, bo pip wcześniej wywalił się na zbyt
  długiej ścieżce. Ustawień systemu (LongPathsEnabled) nie zmieniaj bez zgody użytkownika.
- Tesseract z instalatora UB Mannheim, z pakietem języka polskiego. Ścieżka w `TESSERACT_CMD`.
- **CUDA:** sterownik pokazuje „CUDA 13.0”, ale CUDA 13 nie wspiera Pascala. Instaluj biblioteki
  w wariancie **CUDA 12.x**, np. PyTorch `cu126`. Po instalacji sprawdź:
  ```python
  import torch
  print(torch.cuda.is_available(), torch.cuda.get_arch_list())   # musi zawierać 'sm_61'
  ```
  Jeśli `sm_61` brak, zejdź do wariantu `cu118`. Linia sterowników 580 jest ostatnia z obsługą
  Pascala, więc sterownika nie aktualizuj.
- PaddlePaddle w wersji GPU na Windowsie i Pascalu jest ryzykowne. Modele PaddleOCR uruchamiaj
  przez ONNX. Douczanie rób w PyTorchu (docTR lub PARSeq), chyba że pomiar wskaże inaczej.

## Struktura repo `soc_ocr` (do utworzenia)

```
ocr/
  pdf_regions.py      # inwentaryzacja strony i klasyfikacja regionów
                      #   (tekst natywny / skan / obraz osadzony / diagram wektorowy), wycięcie przez pypdfium2
  preprocess.py       # typ obrazu -> ścieżka obróbki (dark mode, skala, binaryzacja tylko dla skanów)
  engines/
    base.py           # interfejs: detect(img) -> boxes, recognize(img, boxes) -> words z pewnością znaków
    tesseract.py  paddle_onnx.py  paddle_openvino.py  doctr_engine.py
  consensus.py        # porównanie silników, ponowny odczyt, statusy accepted / uncertain
  layout.py           # słowa -> linie -> bloki; w diagramach etykiety zostają osobno
  schema.py           # kontrakt wyjściowy v1 (niżej) + walidacja
  review.py           # kolejka przeglądu: lista niepewnych regionów + wycinki PNG
  cli.py              # python -m ocr extract file.pdf -o out/
synth/                # generator zrzutów: szablony HTML (konsola AWS/Okta/Jira, tabele, diagramy) + Playwright
eval/
  metrics.py          # kompletność detekcji, CER/WER, CER dla trudnych ciągów, kalibracja (ECE), odsetek odrzuceń
  run_eval.py         # raport -> eval/reports/<data>_<maszyna>_<commit>.json
training/             # [GPU] konfiguracje i skrypty douczania
tests/
artifacts/            # POZA gitem: datasets/, models/, cache/ + MANIFEST.json (sumy SHA-256, źródła, licencje)
```

`artifacts/` jest w `.gitignore`. Kod synchronizujemy przez git (repo `soc_ocr` na GitHubie),
a modele i zbiory kopiujemy między maszynami osobno i weryfikujemy po sumach z `MANIFEST.json`.

## Kontrakt wyjściowy v1 (`ocr/schema.py`)

```json
{
  "schema": "ocr/v1", "source": "plik.pdf", "engine_versions": {"tesseract": "5.x", "...": "..."},
  "pages": [{
    "page": 1, "width_pt": 595.0, "height_pt": 842.0,
    "regions": [{
      "id": "p1r3", "kind": "embedded_image | scanned_page | vector_diagram | logo",
      "bbox_pt": [x0, y0, x1, y1], "render_dpi": 300, "status": "accepted | uncertain | unreadable",
      "lines": [{"words": [{
        "text": "j.kowalski@example.com", "conf": 0.993, "char_conf": [0.99, "..."],
        "bbox_px": [x0, y0, x1, y1], "bbox_pt": [x0, y0, x1, y1],
        "votes": {"tesseract": "j.kowalski@example.com", "paddle": "j.kowalski@example.com"},
        "status": "accepted"
      }]}],
      "text": "tekst regionu w kolejności czytania"
    }]
  }],
  "review": [{"region": "p1r5", "reason": "engines_disagree", "crop": "review/p1r5.png"}]
}
```

Wersjonuj schemat. Każda niezgodna zmiana podnosi wersję.

## Miary i cel

Najważniejsza jest **kompletność detekcji**, bo pominięty fragment tekstu nie dostaje żadnej flagi.
Dalej: CER i WER w odczytach `accepted`, osobno dla trudnych ciągów; kalibracja pewności;
odsetek odrzuceń (praca człowieka).

Cel zapisujemy jako: *CER < 0,1% w odczytach zaakceptowanych przy maks. X% skierowanych do przeglądu*.
X ustalamy po pierwszym pomiarze. Żeby wykazać 99,9% poprawnych słów (95% ufności),
potrzeba kilku tysięcy słów testowych.

Zestaw testowy nigdy nie trafia do treningu. Każdy raport zawiera commit, maszynę i wersje silników.

## Etapy

**Etap 1: pomiar gotowych silników (laptop, CPU / OpenVINO)**
1. `pdf_regions.py` + testy na kilku publicznych PDF-ach z obrazami.
2. Zbiór rozwojowy (około 200 obrazów):
   - publiczne zrzuty i diagramy, źródła i licencje w `artifacts/datasets/SOURCES.md`,
   - tekst wzorcowy: ręczny opis albo zrzuty z `synth/`, gdzie tekst znamy z góry.
3. Cztery warianty silników za wspólnym interfejsem. Pomiar jakości i czasu na obraz.
4. Raport z podziałem na typ obrazu i analizą błędów, a na jego podstawie decyzja o douczaniu.

**Etap 2: dane syntetyczne i douczanie**
1. `synth/`: tysiące zrzutów z tekstem wzorcowym (fonty UI, skale, tryb ciemny, polskie znaki).
   Uruchamiane na laptopie, większe przebiegi na desktopie.
2. `[GPU]` Douczenie modelu rozpoznającego na desktopie. FP32, mały batch, zapisy co epokę.
3. Kalibracja pewności i progi `accepted` / `uncertain` na zbiorze walidacyjnym.
4. Ponowny pomiar na laptopie (model w ONNX, wnioskowanie na CPU / OpenVINO).

**Etap 3 (później, po decyzji użytkownika):** weryfikator VLM dla regionów niepewnych.
4 GB pamięci karty na desktopie to za mało na większość VLM-ów. Do tego tabele.

## Punkt integracji z anonimizacją (później)

W repo `soc` plik `anonymizer/adapters/pdf.py` dziś robi OCR tylko dla całych stron skanowanych. Docelowo
odczyta kontrakt `ocr/v1` i zamieni tekst regionów na segmenty `kind="ocr"`. Pozycje słów
posłużą później do zaczerniania. **Nie implementuj tego teraz.**

## Zasady pracy

- **Bez sieci dla prawdziwych dokumentów.** Do internetu wolno sięgać tylko po pakiety, modele
  i publiczne zbiory danych.
- **Nie commituj danych, modeli ani wyników OCR prawdziwych dokumentów** (`artifacts/`, `out/`).
- **Determinizm:** stałe ziarna losowe, przypięte wersje, DPI i parametry silników w konfiguracji, a nie w kodzie.
- **Każda nowa funkcja ma test.** Testy nie mogą wymagać GPU: oznacz takie `@pytest.mark.gpu`
  i pomijaj je na laptopie.
- **Przy wątpliwości co do decyzji z sekcji *Decyzje* zapytaj** zamiast zmieniać kierunek.
- **Odpowiadaj użytkownikowi po polsku.** Kod, nazwy i komentarze po angielsku.
