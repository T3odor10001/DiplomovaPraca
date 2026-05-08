import os
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Set

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class ModuleInfo:
    rel_path: str
    module_name: str
    alias: str
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)

def _module_name_from_rel_path(rel_path: str) -> str:
    """
    'pkg/subpkg/file.py' -> 'pkg.subpkg.file'
    """
    no_ext = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    parts = no_ext.replace("\\", "/").split("/")
    return ".".join(parts)

def _alias_from_module_name(module_name: str) -> str:
    """
    Vytvorí bezpečný alias pre PlantUML – len písmená, čísla a podčiarkovník.

    "modules.CodeAnalyzer" -> "modules_CodeAnalyzer"
    "my-module.utils"      -> "my_module_utils"
    """
    result_chars: List[str] = []
    for ch in module_name:
        if ch.isalnum() or ch == "_":
            result_chars.append(ch)
        else:
            result_chars.append("_")
    alias = "".join(result_chars)
    return alias or "M"

def analyze_repo_structure(files: Dict[str, str], base_dir: str) -> Dict[str, ModuleInfo]:
    """
    Prejde všetky .py súbory a vytvorí:
      - moduly
      - ich classy, funkcie
      - importované moduly
    """
    modules: Dict[str, ModuleInfo] = {}

    for abs_path, code in files.items():
        rel_path = os.path.relpath(abs_path, base_dir)
        module_name = _module_name_from_rel_path(rel_path)
        alias = _alias_from_module_name(module_name)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        info = ModuleInfo(rel_path=rel_path, module_name=module_name, alias=alias)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info.classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                info.functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                info.functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias_node in node.names:
                    info.imports.add(alias_node.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    info.imports.add(node.module)

        modules[module_name] = info

    return modules

def build_repo_summary(modules: Dict[str, ModuleInfo]) -> str:
    """
    Z ModuleInfo objektov spraví textový summary, ktoré pošleme LLM.
    Zároveň mu pošleme aj aliasy, ktoré MUSÍ používať.
    """
    lines: List[str] = []

    lines.append("MODULES AND ALIASES:")
    for mname, info in sorted(modules.items(), key=lambda kv: kv[0]):
        lines.append(f"- MODULE: {mname}  ALIAS: {info.alias}  FILE: {info.rel_path}")
    lines.append("")

    lines.append("DETAILS:")
    for mname, info in sorted(modules.items(), key=lambda kv: kv[0]):
        lines.append(f"Module: {mname}")
        if info.classes:
            lines.append(f"  Classes: {', '.join(sorted(info.classes))}")
        if info.functions:
            lines.append(f"  Functions: {', '.join(sorted(info.functions))}")
        if info.imports:
            lines.append(f"  Imports: {', '.join(sorted(info.imports))}")
        lines.append("")

    return "\n".join(lines)

PLANTUML_TEMPLATE = """
You are a software architect.

You will be given a summary of a Python repository: modules, their classes, functions,
imports AND a mapping of each module to a safe PlantUML alias.

TASK:
Create a PlantUML *component diagram* that shows the main modules and their dependencies.

VERY IMPORTANT RULES (strict):
- Output ONLY valid PlantUML code.
- Start with: @startuml
- End with: @enduml
- For each module from the "MODULES AND ALIASES" list, declare a component in EXACT format:
    component "<MODULE_NAME>" as <ALIAS>
  where:
    - <MODULE_NAME> is the module name exactly as listed (e.g. modules.CodeAnalyzer)
    - <ALIAS> is the alias exactly as listed (e.g. modules_CodeAnalyzer)
- When drawing arrows (dependencies), ALWAYS use ONLY the aliases:
    modules_App --> modules_CodeAnalyzer
  NOT the raw module names.
- Do NOT invent new aliases. Use only aliases from the list.
- Do NOT add any explanation text, comments, or notes outside of PlantUML.
- No text before @startuml or after @enduml.

Dependencies:
- Use imports information to determine which modules depend on which.
- If module A imports module B (directly or via submodule), draw:
    AliasOfA --> AliasOfB

Keep the diagram reasonably small and readable – you don't have to show every tiny helper.

Here is the repository summary and alias mapping:

{summary}
"""

def create_plantuml_chain(model_name: str = "llama3.2"):
    """
    Vytvorí LLM chain pre generovanie PlantUML component diagramu.
    """
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(PLANTUML_TEMPLATE)
    return prompt | model

def generate_plantuml_for_repo(
    files: Dict[str, str],
    base_dir: str,
    output_file: str = "components.puml",
    model_name: str = "llama3.2",
) -> None:
    """
    - Analyzuje štruktúru repozitára (moduly + aliasy + importy)
    - Pošle summary PlantUML agentovi (LLM)
    - Výsledný PlantUML zapíše do súboru (s malou poistkou)
    """
    modules = analyze_repo_structure(files, base_dir)
    if not modules:
        print("No Python modules found to generate PlantUML diagram.")
        return

    summary = build_repo_summary(modules)
    chain = create_plantuml_chain(model_name=model_name)

    print("\nGenerating PlantUML component diagram using LLM agent...")
    plantuml_code = chain.invoke({"summary": summary})
    text = str(plantuml_code).strip()

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

    text = "\n".join(line.rstrip() for line in text.splitlines())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n✅ PlantUML component diagram written to: {output_file}")