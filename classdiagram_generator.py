import os
import ast
from typing import Dict, List, Optional

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from plantuml_renderer import render_plantuml

from importance_analyzer import get_top_important_functions, FunctionRecord
from code_indexer import IndexedFile

def _get_source_segment(code: str, node: ast.AST, max_lines: int = 12) -> str:
    """
    Returns a short source snippet (up to max_lines) for context.
    """
    lines = code.splitlines()
    start = max(0, getattr(node, "lineno", 1) - 1)
    end = min(len(lines), start + max_lines)
    return "\n".join(lines[start:end])

def _module_name_from_rel_path(rel_path: str) -> str:
    no_ext = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    return no_ext.replace("\\", ".").replace("/", ".")

class TopFunctionVisitor(ast.NodeVisitor):
    """
    Finds only TOP functions/methods and extracts short context.
    """

    def __init__(self, code: str, module_name: str, targets: List[FunctionRecord]):
        self.code = code
        self.module_name = module_name
        self.targets = {r.qualname for r in targets}
        self.class_stack: List[str] = []
        self.results: List[dict] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _handle_function(self, node: ast.AST, name: str):
        qualname = ".".join(self.class_stack + [name]) if self.class_stack else name
        if qualname not in self.targets:
            return

        self.results.append(
            {
                "module": self.module_name,
                "class": self.class_stack[-1] if self.class_stack else None,
                "function": name,
                "args": [a.arg for a in node.args.args],
                "source": _get_source_segment(self.code, node),
            }
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node, node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, node.name)
        self.generic_visit(node)

CLASS_DIAGRAM_PROMPT = """
You are a UML expert.

You are given information about the TOP 10 most important functions/methods
in a Python project. For each method you see:
- module
- class (if any)
- method name
- arguments
- short source code context

TASK:
Generate a PlantUML CLASS DIAGRAM that focuses ONLY on these methods
and their surrounding classes.

STRICT OUTPUT RULES (MANDATORY):
- Output ONLY valid PlantUML
- Start with exactly: @startuml
- End with exactly: @enduml
- Use ONLY PlantUML CLASS DIAGRAM syntax
- Declare classes like this:

  class ClassName {{
    + method(arg1, arg2)
  }}

- Include ONLY classes and methods that appear in the input
- Do NOT invent classes
- Do NOT invent methods
- Do NOT use ":" after class names
- Do NOT use comments, markdown, or placeholders
- If a class has no relevant methods, use empty braces {{ }}

RELATIONSHIPS:
- If one class depends on or calls another based on the provided context,
  draw a relationship:
    ClassA --> ClassB

IMPORTANT:
This is UML, NOT Python, NOT pseudocode.

VALID EXAMPLE:

@startuml
class A {{
  + important(x)
}}

class B {{
}}

A --> B
@enduml

INPUT DATA:
{elements}
"""

def _build_elements_summary(
    top_funcs: List[FunctionRecord],
    files: Dict[str, str],
    base_dir: str,
    code_index: Optional[Dict[str, IndexedFile]] = None,
) -> str:
    """
    Builds a rich textual context for LLM from TOP functions.
    When code_index is provided, includes LLM file summaries, docstrings,
    and inheritance info for better diagram accuracy.
    """
    grouped: Dict[str, List[FunctionRecord]] = {}
    for rec in top_funcs:
        grouped.setdefault(rec.file_path, []).append(rec)

    parts: List[str] = []
    idx = 1

    for file_path, recs in grouped.items():
        code = files[file_path]
        rel = os.path.relpath(file_path, base_dir)
        module_name = _module_name_from_rel_path(rel)

        rel_normalized = rel.replace("\\", "/")
        indexed = code_index.get(rel_normalized) if code_index else None

        if indexed and indexed.llm_summary:
            parts.append(f"--- File: {rel} ---")
            parts.append(f"Purpose: {indexed.llm_summary}")
            if indexed.docstring:
                parts.append(f"Module docstring: {indexed.docstring[:200]}")
            if indexed.class_bases:
                for cls_name, bases in indexed.class_bases.items():
                    parts.append(f"Inheritance: {cls_name} extends {', '.join(bases)}")
            parts.append("")

        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        visitor = TopFunctionVisitor(code, module_name, recs)
        visitor.visit(tree)

        for item in visitor.results:
            parts.append(f"Function #{idx}:")
            parts.append(f"Module: {item['module']}")
            parts.append(f"Class: {item['class'] or '(module-level)'}")
            parts.append(f"Method: {item['function']}")
            parts.append(f"Args: {', '.join(item['args'])}")
            parts.append("Source:")
            parts.append(item["source"])
            parts.append("")
            idx += 1

    return "\n".join(parts)

def _create_chain(model_name: str):
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(CLASS_DIAGRAM_PROMPT)
    return prompt | model

def generate_classdiagram_for_repo(
    files: Dict[str, str],
    base_dir: str,
    output_file: str = "classes.puml",
    model_name: str = "llama3.2",
    top_n: int = 10,
    render_png: bool = True,
    render_svg: bool = True,
    code_index: Optional[Dict[str, IndexedFile]] = None,
) -> None:
    """
    Generates a PlantUML CLASS diagram from TOP N important functions using LLM ONLY
    and immediately renders it to image(s).
    When code_index (enriched) is provided, the LLM receives file purposes,
    docstrings, and inheritance info for more accurate diagrams.
    """
    top_funcs = get_top_important_functions(files, base_dir, top_n=top_n)
    if not top_funcs:
        raise RuntimeError("No important functions found.")

    elements = _build_elements_summary(top_funcs, files, base_dir, code_index=code_index)
    chain = _create_chain(model_name)

    print(f"Generating LLM-based CLASS diagram from TOP {len(top_funcs)} functions...")
    result = chain.invoke({"elements": elements})
    text = str(result).strip()

    lower = text.lower()
    start = lower.find("@startuml")
    end = lower.rfind("@enduml")

    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("LLM did not return valid PlantUML output.")

    final_text = text[start : end + len("@enduml")]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✅ LLM CLASS diagram written to: {output_file}")

    try:
        if render_png:
            png = render_plantuml(output_file, "png")
            print(f"🖼 PNG rendered: {png}")

        if render_svg:
            svg = render_plantuml(output_file, "svg")
            print(f"🖼 SVG rendered: {svg}")

    except Exception as e:
        print(f"⚠️ Diagram render failed: {e}")
