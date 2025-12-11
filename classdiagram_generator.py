import os
import ast
from typing import Dict, List

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from importance_analyzer import get_top_important_functions, FunctionRecord


def _get_source_segment(code: str, node: ast.AST) -> str:
    """
    Vráti textový úsek kódu pre daný AST node (podľa lineno / end_lineno).
    """
    lines = code.splitlines()
    start = max(0, getattr(node, "lineno", 1) - 1)
    end = getattr(node, "end_lineno", start + 1)
    end = min(len(lines), end)
    segment = "\n".join(lines[start:end])
    return segment


def _module_name_from_rel_path(rel_path: str) -> str:
    """
    'pkg/subpkg/file.py' -> 'pkg.subpkg.file'
    """
    no_ext = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    parts = no_ext.replace("\\", "/").split("/")
    return ".".join(parts)


class TopFuncVisitor(ast.NodeVisitor):
    """
    Pre jeden .py súbor nájde konkrétne top funkcie (podľa qualname)
    a vytiahne ich kontext (class + source).
    """

    def __init__(self, code: str, module_name: str, targets: List[FunctionRecord]):
        self.code = code
        self.module_name = module_name
        # set kvalifikovaných mien, ktoré hľadáme
        self.target_names = {rec.qualname for rec in targets}
        self.class_stack: List[ast.ClassDef] = []
        self.results: List[dict] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node)
        self.generic_visit(node)
        self.class_stack.pop()

    def _handle_function(self, node: ast.AST, name: str):
        # kvalifikované meno z aktuálneho class stacku
        qualname_parts = [c.name for c in self.class_stack] + [name]
        qualname = ".".join(qualname_parts) if qualname_parts else name

        if qualname not in self.target_names:
            # táto funkcia nie je medzi TOP 10
            return

        class_name = self.class_stack[-1].name if self.class_stack else None
        # container node – buď celá trieda, alebo samotná funkcia (ak je na module)
        container_node = self.class_stack[-1] if self.class_stack else node
        container_source = _get_source_segment(self.code, container_node)

        # zoznam argumentov
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arg_names = [arg.arg for arg in node.args.args]
        else:
            arg_names = []

        self.results.append(
            {
                "module": self.module_name,
                "class_name": class_name,
                "function_name": name,
                "qualname": qualname,
                "args": arg_names,
                "container_source": container_source,
            }
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node, node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, node.name)
        self.generic_visit(node)


CLASS_UML_TEMPLATE = """
You are an expert in UML and PlantUML.

You are given information about up to 10 MOST IMPORTANT functions in a Python project.
For each function you see:
- module name
- optional class name
- function name and arguments
- source code of the enclosing class (or the function itself for module-level functions)

TASK:
Create a PlantUML CLASS DIAGRAM that focuses on the classes and key methods related to these functions.

HARD RULES (VERY STRICT):
- Output ONLY valid PlantUML code for a CLASS diagram.
- Start with: @startuml
- End with: @enduml
- Absolutely DO NOT use Markdown:
  - No headings like '#', '##', '###'
  - No bullet lists like '-', '*'
  - No fenced code blocks like ``` or ```python
  - No explanatory text outside PlantUML
- Each non-empty line must be valid PlantUML syntax.

Use ONLY these constructs:
- class / interface / abstract class declarations
- method signatures inside class bodies
- simple associations between classes (e.g. A --> B)

Example of VALID output format:

@startuml
class DocGenerator {{
  + make_text_documentation(files, output_dir)
  + generate_readme(files, output_dir, repo_root, readme_name)
}}

class RepositoryReader {{
  + clone_repository()
}}

DocGenerator --> RepositoryReader
@enduml

NOTES:
- Group methods into their classes. For module-level functions (no class), you MAY create
  a pseudo-class named after the module, e.g.:

    class "modules.arch_utils (module)" {{}}

- Show methods ONLY for the important functions (you don't have to list all helpers).
- You MAY draw associations between classes if they obviously collaborate, but keep it simple.

Do NOT output anything except PlantUML code between @startuml and @enduml.

Here are the important functions and their context:

{elements}
"""



def _build_elements_summary(
    top_funcs: List[FunctionRecord],
    files: Dict[str, str],
    base_dir: str,
) -> str:
    """
    Z 10 najdôležitejších funkcií vyrobí textový "elements" summary pre LLM.
    """
    # rozdelíme top funkcie podľa súboru
    by_file: Dict[str, List[FunctionRecord]] = {}
    for rec in top_funcs:
        by_file.setdefault(rec.file_path, []).append(rec)

    parts: List[str] = []
    idx = 1

    for file_path, recs in by_file.items():
        code = files[file_path]
        rel_path = os.path.relpath(file_path, base_dir)
        module_name = _module_name_from_rel_path(rel_path)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        visitor = TopFuncVisitor(code=code, module_name=module_name, targets=recs)
        visitor.visit(tree)

        for item in visitor.results:
            parts.append(f"Function #{idx}:")
            parts.append(f"  Module: {item['module']}")
            parts.append(f"  Class: {item['class_name'] or '(module-level)'}")
            parts.append(f"  Name: {item['function_name']}")
            parts.append(f"  Qualified name: {item['qualname']}")
            parts.append(f"  Args: ({', '.join(item['args'])})")
            parts.append("  Container source:")
            parts.append('  """')
            # indentujeme kvôli čitateľnosti
            for line in item["container_source"].splitlines():
                parts.append("  " + line)
            parts.append('  """')
            parts.append("")
            idx += 1

    return "\n".join(parts)


def create_classdiagram_chain(model_name: str = "llama3.2"):
    """
    Vytvorí LLM chain pre generovanie PlantUML class diagramu.
    """
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(CLASS_UML_TEMPLATE)
    return prompt | model

def _is_plantuml_line(line: str) -> bool:
    """
    Heuristicky rozhodne, či riadok vyzerá ako PlantUML pre class diagram.
    Zahodíme markdown / text, necháme len to, čo má šancu byť validné.
    """
    s = line.strip()
    if not s:
        return True  # prázdne riadky necháme kvôli čitateľnosti

    # Povolené kľúčové slová
    if s.startswith("@startuml") or s.startswith("@enduml"):
        return True
    if s.startswith("class ") or s.startswith("interface ") or s.startswith("abstract class "):
        return True
    if s.startswith("package ") or s.startswith("namespace "):
        return True
    if s.startswith("note ") or s.startswith("}"):
        return True

    # Čiary / asociácie medzi triedami
    if "--" in s or "<|" in s or ".." in s or "o--" in s or "*--" in s:
        return True

    # Všetko, čo vyzerá ako markdown, zahodíme
    if s.startswith("#") or s.startswith("**") or s.startswith("* "):
        return False
    if s.startswith("```"):
        return False

    # Keďže chceme radšej prísnu filtráciu, zvyšok vyhodíme
    return False


def generate_classdiagram_for_repo(
    files: Dict[str, str],
    base_dir: str,
    output_file: str = "classes.puml",
    model_name: str = "llama3.2",
    top_n: int = 10,
) -> None:
    """
    - Nájde TOP N (default 10) najdôležitejších funkcií pomocou importance_analyzer
    - Z nich vyrobí summary
    - Pošle ho LLM agentovi, ktorý vygeneruje PlantUML CLASS diagram
    - Výstup prefiltruje tak, aby obsahoval len PlantUML riadky
    """
    top_funcs = get_top_important_functions(files, base_dir, top_n=top_n)
    if not top_funcs:
        print("No functions found to generate class diagram.")
        return

    elements = _build_elements_summary(top_funcs, files, base_dir)
    chain = create_classdiagram_chain(model_name=model_name)

    print(f"\nGenerating PlantUML CLASS diagram from top {len(top_funcs)} functions using LLM agent...")
    result = chain.invoke({"elements": elements})
    text = str(result).strip()

    # vytiahneme len blok medzi @startuml a @enduml (pre istotu)
    lower = text.lower()
    start_idx = lower.find("@startuml")
    end_idx = lower.rfind("@enduml")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + len("@enduml")]
    else:
        if "@startuml" not in text:
            text = "@startuml\n" + text
        if "@enduml" not in text:
            text = text + "\n@enduml"

    # Rozsekáme na riadky a vyhodíme všetko, čo nevyzerá ako PlantUML class diagram
    lines = text.splitlines()
    filtered = [line for line in lines if _is_plantuml_line(line)]

    # Poistka: ak LLM úplne zlyhá a neostane nič, radšej tam dáme aspoň prázdny blok
    if not any(l.strip().startswith("class ") for l in filtered):
        # to isté ako nič: radšej nech je jednoduchý minimálny diagram než README vnútri
        filtered = ["@startuml", "@enduml"]

    final_text = "\n".join(filtered)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"\n PlantUML CLASS diagram written to: {output_file}")
