# Inštalačná príručka — AI Documentation & Code Explorer

Aplikácia na automatické generovanie dokumentácie, UML diagramov a rozpoznávanie architektúry/dizajnových vzorov pre Python repozitáre. Využíva lokálny LLM cez Ollama (model `llama3.2`).

---

## 1. Systémové požiadavky

| Komponent | Verzia / popis |
|-----------|----------------|
| Python    | 3.10 alebo vyššie |
| Java JRE  | 8 alebo vyššie (potrebné pre PlantUML) |
| Ollama    | aktuálna verzia (lokálny LLM runtime) |
| Git       | aktuálna verzia |
| RAM       | minimálne 8 GB (odporúčané 16 GB pre väčšie repozitáre) |
| OS        | Windows 10/11, Linux, macOS |

---

## 2. Inštalácia základných nástrojov

### 2.1 Python

Stiahni a nainštaluj Python 3.10+ z [python.org](https://www.python.org/downloads/).

Overenie:
```bash
python --version
```

### 2.2 Git

Stiahni z [git-scm.com](https://git-scm.com/).

Overenie:
```bash
git --version
```

### 2.3 Java JRE

Potrebná na renderovanie PlantUML diagramov do PNG/SVG.

- Windows: nainštaluj [Eclipse Temurin](https://adoptium.net/) alebo Oracle JRE.
- Linux: `sudo apt install default-jre`
- macOS: `brew install openjdk`

Overenie:
```bash
java -version
```

### 2.4 Ollama (lokálny LLM)

Aplikácia používa lokálne bežiaci LLM model **llama3.2** cez [Ollama](https://ollama.com/).

1. Stiahni Ollama z [ollama.com/download](https://ollama.com/download).
2. Nainštaluj a spusti.
3. Stiahni model:
   ```bash
   ollama pull llama3.2
   ```
4. Over, že beží:
   ```bash
   ollama list
   ```

Ollama štandardne beží na `http://localhost:11434` — aplikácia ho automaticky nájde.

---

## 3. Klonovanie projektu

```bash
git clone https://github.com/T3odor10001/DiplomovaPraca.git
cd DiplomovaPraca
```

---

## 4. Vytvorenie virtuálneho prostredia

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows (cmd)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 5. Inštalácia Python závislostí

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Kompletný zoznam závislostí (`requirements.txt`):
- `streamlit` — webové UI
- `Pillow` — práca s obrázkami
- `gitpython` — klonovanie repozitárov
- `langchain` — orchestrácia LLM
- `langchain-ollama` — Ollama integrácia
- `langchain-core` — základné LangChain typy
- `langgraph` — workflow grafy pre review agenta

---

## 6. Inštalácia PlantUML

PlantUML je `.jar` súbor potrebný na renderovanie diagramov.

1. Vytvor priečinok `tools` v koreňovom adresári projektu:
   ```bash
   mkdir tools
   ```
2. Stiahni `plantuml.jar` z [plantuml.com/download](https://plantuml.com/download).
3. Ulož súbor ako `tools/plantuml.jar`.

Štruktúra by mala vyzerať takto:
```
DiplomovaPraca/
├── tools/
│   └── plantuml.jar
├── app.py
├── main.py
└── ...
```

---

## 7. Spustenie aplikácie

### 7.1 Streamlit web UI (odporúčané)

```bash
streamlit run app.py
```

Otvorí sa prehliadač na adrese `http://localhost:8501`.

V UI:
1. Zadaj URL Git repozitára (napr. `https://github.com/user/repo`).
2. Vyber akciu z menu vľavo:
   - **Generate Documentation** — vygeneruje dokumentáciu súborov
   - **Top 10 Important Functions** — zobrazí 10 najdôležitejších funkcií
   - **Generate Top10 Class Diagram** — UML class diagram top funkcií
   - **Generate Full Class Diagram** — UML diagram všetkých tried
   - **Generate Installation Guide** — vygeneruje inštalačnú príručku
   - **Interactive Code Explanation** — chat s diagramami
   - **Generate Dependency Graph** — graf závislostí top funkcií
   - **Recognize Architecture & Design Patterns** — rozpoznanie architektúry + diagram
3. Klikni **Run**.

### 7.2 CLI verzia

```bash
python main.py
```

Aplikácia sa opýta na URL repozitára a ponúkne menu od 1 po 8.

---

## 8. Riešenie problémov

### Ollama: connection refused / WinError 10061
- Over, že Ollama beží: `ollama list`
- Reštartuj službu Ollama
- Aplikácia obsahuje retry logiku — pri zlyhaní počká 3 s a skúsi znovu (max 2× pre indexovanie)

### PlantUML rendering failed
- Over Java: `java -version`
- Over, že existuje `tools/plantuml.jar`
- Skús ručne: `java -jar tools/plantuml.jar -tpng test.puml`

### `ModuleNotFoundError: No module named 'langchain_ollama'`
- Aktivuj virtuálne prostredie a znovu spusti `pip install -r requirements.txt`

### Pomalé generovanie pri prvom spustení
- Po klonovaní sa raz vytvorí **LLM-enriched code index** (zaberie 30 – 60 s pre stredne veľký repozitár).
- Index je uložený v session state, ďalšie akcie sú už okamžité.

### Streamlit zobrazuje starý cache
- V termináli stlač `Ctrl+C` a spusti znovu, alebo klikni **Reset** v UI.

---

## 9. Štruktúra projektu

```
DiplomovaPraca/
├── app.py                          # Streamlit UI (hlavný entry point)
├── main.py                         # CLI verzia + spoločné triedy (RepositoryReader)
├── code_indexer.py                 # Indexovanie kódu + LLM enrichment
├── context_selector.py             # Výber relevantného kontextu pre otázky
├── code_explainer.py               # LLM vysvetľovač kódu
├── importance_analyzer.py          # Skórovanie dôležitosti funkcií (AST)
├── classdiagram_generator.py       # UML class diagram top 10 funkcií
├── full_classdiagram_generator.py  # UML diagram všetkých tried
├── top_dependency_llm.py           # Graf závislostí (deterministický + LLM)
├── pattern_recognizer.py           # Rozpoznávanie architektúry + arch. diagram
├── diagram_highlighter.py          # Zvýrazňovanie uzlov v diagramoch
├── plantuml_generator.py           # AST analýza modulov
├── plantuml_renderer.py            # Volanie Java + plantuml.jar
├── install_guide_generator.py      # LLM generátor inštalačnej príručky
├── langgraph_workflows.py          # Doc-writer ↔ Reviewer workflow graf
├── code_chunker.py                 # AST-based chunking utilita
├── tools/
│   └── plantuml.jar                # PlantUML renderer (treba stiahnuť)
└── requirements.txt
```

---

## 10. Použité technológie

- **LangChain + LangGraph** — orchestrácia LLM agentov a workflow grafov (DocWriter ↔ Reviewer)
- **Ollama (llama3.2)** — lokálny jazykový model
- **Streamlit** — webové rozhranie
- **PlantUML** — vykresľovanie UML diagramov
- **Python AST** — deterministická analýza kódu
- **GitPython** — klonovanie repozitárov

---

Autor: Teodor Fuček
