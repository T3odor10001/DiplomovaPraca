import os
import ast
from typing import Dict, List

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# využijeme existujúceho analyzátora modulov z plantuml_generator.py
from plantuml_generator import analyze_repo_structure


# ---------- REPREZENTÁCIA TRIED ----------

class ClassInfo:
    """Reprezentácia jednej triedy v projekte."""

    def __init__(self, name: str, module: str):
        self.name = name
        self.module = module
        self.attributes: List[str] = []
        self.methods: List[str] = []

    def add_attribute(self, name: str):
        if name and name not in self.attributes:
            self.attributes.append(name)

    def add_method(self, name: str):
        if name and name not in self.methods:
            self.methods.append(name)


# ---------- PARSOVANIE KÓDU ----------

class ClassCollector(ast.NodeVisitor):
    """
    AST visitor:
    - nájde všetky ClassDef
    - zbiera:
        - class-level atribúty (X = 1)
        - instance atribúty (self.x = ...)
        - metódy (FunctionDef, AsyncFunctionDef)
    """

    def __init__(self, module: str):
        self.module = module
        self.classes: List[ClassInfo] = []
        self._current_class: ClassInfo | None = None

    # ClassDef
    def visit_ClassDef(self, node: ast.ClassDef):
        cls = ClassInfo(name=node.name, module=self.module)
        self.classes.append(cls)
        prev = self._current_class
        self._current_class = cls

        # prejdeme telo triedy
        for stmt in node.body:
            # class-level atribúty: X = ...
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        cls.add_attribute(target.id)

            # metódy
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cls.add_method(stmt.name)
                # vnútri metódy hľadáme self.xxx = ...
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if (
                                isinstance(t, ast.Attribute)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == "self"
                                and isinstance(t.attr, str)
                            ):
                                cls.add_attribute(t.attr)

        # rekurzia (napr. vnorené triedy)
        self.generic_visit(node)
        self._current_class = prev


def extract_classes_from_code(source: str, module: str) -> List[ClassInfo]:
    """Z jedného .py súboru vytiahne všetky triedy + ich atribúty/metódy."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    collector = ClassCollector(module=module)
    collector.visit(tree)
    return collector.classes


def scan_repo_for_classes(files: Dict[str, str], base_dir: str) -> List[ClassInfo]:
    """Prejde všetky .py súbory a nazbiera ClassInfo."""
    all_classes: List[ClassInfo] = []

    for file_path, code in files.items():
        rel = os.path.relpath(file_path, base_dir)
        module_name = rel.replace("\\", ".").replace("/", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        all_classes.extend(extract_classes_from_code(code, module_name))

    return all_classes


# ---------- RENDERING TRIED DO PLANTUML ----------

def render_classes_to_plantuml(classes: List[ClassInfo]) -> str:
    """
    Deterministicky vyrenderuje všetky triedy do PlantUML:
    - class Názov { ... }
    - vnútri len dôležité atribúty a metódy
    """
    lines: List[str] = []
    for cls in sorted(classes, key=lambda c: (c.module, c.name)):
        lines.append(f"class {cls.name} {{")
        # atribúty
        for attr in sorted(cls.attributes):
            lines.append(f"  + {attr}")
        # metódy
        for m in sorted(cls.methods):
            lines.append(f"  + {m}()")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip()  # bez posledného prázdneho riadku


# ---------- PROMPT PRE LLM – VZŤAHY MEDZI TRIEDAMI ----------

RELATIONS_UML_TEMPLATE = """
You are an AI assistant that generates ONLY PlantUML relationship lines between classes.

You will receive an overview of:
- Python modules
- Classes in each module
- Class attributes and methods
- Module imports

Your task:
Generate PlantUML relationship lines that connect the classes.

STRICT RULES:
1. Output ONLY PlantUML relationship lines, nothing else.
2. Do NOT output @startuml or @enduml.
3. Do NOT output class declarations.
4. Do NOT output comments, markdown, prose, or explanations.
5. Valid examples of allowed lines:
   ClassA --> ClassB
   ClassX ..> ClassY
   ClassBase <|-- ClassChild

HEURISTICS FOR RELATIONSHIPS:
- If two classes are in the same module, you MAY connect related ones.
- If a method or attribute name clearly references another class name, you MAY use a dependency arrow (..>).
- If module imports indicate dependency between modules, you MAY connect classes across those modules.
- Keep the number of relationships reasonable and focus on the main architecture.

INPUT STRUCTURE DESCRIPTION:

{class_list}
"""


def create_relations_chain(model_name: str = "llama3.2"):
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(RELATIONS_UML_TEMPLATE)
    return prompt | model


def is_relation_line(line: str) -> bool:
    """
    Heuristicky ponechá len riadky, ktoré vyzerajú ako PlantUML vzťahy medzi triedami.
    """
    s = line.strip()
    if not s:
        return False
    # zakážeme definície tried, komentáre, markdown
    if s.startswith("class ") or s.startswith("@startuml") or s.startswith("@enduml"):
        return False
    if s.startswith("#") or s.startswith("* ") or s.startswith("```"):
        return False

    # jednoduché šípky medzi triedami
    if "--" in s or ".." in s or "<|" in s or "*--" in s or "o--" in s:
        return True

    return False


# ---------- HLAVNÁ FUNKCIA PRE MAIN.PY ----------

def generate_full_classdiagram(
    files: Dict[str, str],
    base_dir: str,
    output_file: str = "full_classes.puml",
    model_name: str = "llama3.2",
):
    """
    - analyzuje moduly (importy) aj triedy (atribúty, metódy)
    - class bloky v PlantUML generuje deterministicky (bez LLM)
    - LLM agent generuje LEN vzťahy medzi triedami
    - výsledok zapíše ako kompletný @startuml ... @enduml diagram
    """
    classes = scan_repo_for_classes(files, base_dir)
    if not classes:
        print("No classes found.")
        return

    modules = analyze_repo_structure(files, base_dir)  # z plantuml_generator.py

    # index: modul -> zoznam tried
    classes_by_module: Dict[str, List[ClassInfo]] = {}
    for cls in classes:
        classes_by_module.setdefault(cls.module, []).append(cls)

    # 1) text pre LLM – modulový prehľad + detaily tried
    overview_lines: List[str] = []

    overview_lines.append("MODULE OVERVIEW:")
    for mname, minfo in sorted(modules.items(), key=lambda kv: kv[0]):
        overview_lines.append(f"MODULE: {mname}")
        if mname in classes_by_module:
            overview_lines.append("  CLASSES:")
            for cls in sorted(classes_by_module[mname], key=lambda c: c.name):
                overview_lines.append(f"    - {cls.name}")
        if minfo.imports:
            overview_lines.append("  IMPORTS:")
            for imp in sorted(minfo.imports):
                overview_lines.append(f"    - {imp}")
        overview_lines.append("")

    overview_lines.append("CLASSES DETAIL:")
    for cls in sorted(classes, key=lambda c: (c.module, c.name)):
        overview_lines.append(f"CLASS: {cls.name}")
        overview_lines.append(f"MODULE: {cls.module}")
        if cls.attributes:
            overview_lines.append("ATTRIBUTES:")
            for a in sorted(cls.attributes):
                overview_lines.append(f"  - {a}")
        if cls.methods:
            overview_lines.append("METHODS:")
            for m in sorted(cls.methods):
                overview_lines.append(f"  - {m}")
        overview_lines.append("")

    class_text = "\n".join(overview_lines)

    # 2) LLM – nech vygeneruje iba šípky
    chain = create_relations_chain(model_name=model_name)

    print("Generating full repository class relationships via LLM agent...")
    raw = chain.invoke({"class_list": class_text})
    text = str(raw) if raw is not None else ""
    text = text.strip()

    relation_lines: List[str] = []
    for ln in text.splitlines():
        if is_relation_line(ln):
            relation_lines.append(ln.strip())

    # 3) deterministic class bloky
    classes_block = render_classes_to_plantuml(classes)

    # 4) zložíme finálny diagram
    out_lines: List[str] = []
    out_lines.append("@startuml")
    out_lines.append(classes_block)
    if relation_lines:
        out_lines.append("")  # medzera
        out_lines.extend(relation_lines)
    out_lines.append("@enduml")

    final_text = "\n".join(out_lines)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"Full class diagram saved to {output_file}")
