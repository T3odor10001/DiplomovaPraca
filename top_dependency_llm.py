import re
import ast
import time
from typing import Dict, List, Optional, Set, Tuple

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from importance_analyzer import get_top_important_functions, FunctionRecord
from plantuml_renderer import render_plantuml
from code_indexer import IndexedFile

def _find_function_node(tree: ast.AST, qualname: str, lineno: int) -> Optional[ast.AST]:
    """Find the AST node for a specific function by qualname and line number."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno == lineno:
                return node
    return None

class _ScopedCallVisitor(ast.NodeVisitor):
    """Visits a specific function's AST subtree to find calls to top-N functions."""

    def __init__(self, bare_to_qualnames: Dict[str, List[str]]):
        self.bare_to_qualnames = bare_to_qualnames
        self.called_qualnames: Set[str] = set()

    def visit_Call(self, node: ast.Call):
        bare = None
        if isinstance(node.func, ast.Name):
            bare = node.func.id
        elif isinstance(node.func, ast.Attribute):
            bare = node.func.attr
        if bare and bare in self.bare_to_qualnames:
            for qn in self.bare_to_qualnames[bare]:
                self.called_qualnames.add(qn)
        self.generic_visit(node)

def collect_dependencies(
    files: Dict[str, str],
    top_funcs: List[FunctionRecord],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Collect dependencies between top-N functions using qualnames.

    Returns:
      nodes: list of qualnames (e.g. "CodeAnalyzer.get_deps")
      edges: list of (qualname_from, qualname_to) — no self-loops
    """
    nodes: List[str] = []
    edges: List[Tuple[str, str]] = []

    bare_to_qualnames: Dict[str, List[str]] = {}
    for f in top_funcs:
        bare_to_qualnames.setdefault(f.name, []).append(f.qualname)

    top_qualnames = {f.qualname for f in top_funcs}

    for f in top_funcs:
        nodes.append(f.qualname)

        try:
            tree = ast.parse(files[f.file_path])
        except SyntaxError:
            continue

        func_node = _find_function_node(tree, f.qualname, f.lineno)
        if func_node is None:
            func_node = tree

        visitor = _ScopedCallVisitor(bare_to_qualnames)
        visitor.visit(func_node)

        for called_qn in visitor.called_qualnames:
            if called_qn in top_qualnames and called_qn != f.qualname:
                edges.append((f.qualname, called_qn))

    edges = sorted(set(edges))
    return nodes, edges

def _make_component_id(index: int) -> str:
    """Generate a stable component ID like C1, C2, ..."""
    return f"C{index}"

def _build_deterministic_plantuml(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    descriptions: Optional[Dict[str, str]] = None,
    groups: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Build valid PlantUML component diagram deterministically.

    Output matches diagram_highlighter.py's _COMPONENT_RE regex:
      component "LABEL" as ID
    """
    qualname_to_id: Dict[str, str] = {}
    for i, qn in enumerate(nodes, start=1):
        qualname_to_id[qn] = _make_component_id(i)

    lines: List[str] = ["@startuml", ""]

    if groups:
        grouped: Set[str] = set()
        for group_name, members in sorted(groups.items()):
            pkg_members = [qn for qn in members if qn in qualname_to_id]
            if not pkg_members:
                continue
            lines.append(f'package "{group_name}" {{')
            for qn in pkg_members:
                cid = qualname_to_id[qn]
                lines.append(f'  component "{qn}" as {cid}')
                grouped.add(qn)
            lines.append("}")
            lines.append("")

        for qn in nodes:
            if qn not in grouped:
                cid = qualname_to_id[qn]
                lines.append(f'component "{qn}" as {cid}')
    else:
        for qn in nodes:
            cid = qualname_to_id[qn]
            lines.append(f'component "{qn}" as {cid}')

    lines.append("")

    for src, dst in edges:
        src_id = qualname_to_id.get(src)
        dst_id = qualname_to_id.get(dst)
        if src_id and dst_id:
            lines.append(f"{src_id} --> {dst_id}")

    if descriptions:
        lines.append("")
        for qn, desc in descriptions.items():
            cid = qualname_to_id.get(qn)
            if cid and desc:
                safe_desc = desc.replace("\n", " ").strip()[:80]
                lines.append(f'note right of {cid} : {safe_desc}')

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)

ENRICHMENT_PROMPT = """
You are analyzing a Python project's most important methods.

For each method below, provide:
1. A SHORT description (max 10 words) of what it does
2. A GROUP name (1-2 words) for logical grouping by responsibility

INPUT METHODS:
{method_details}

OUTPUT FORMAT (strictly follow, one entry per method, no extra text):
METHOD: <qualname>
DESCRIPTION: <10 words max>
GROUP: <group name>
"""

def _build_method_details(
    top_funcs: List[FunctionRecord],
    code_index: Optional[Dict[str, IndexedFile]],
    files: Dict[str, str],
) -> str:
    """Build rich method details for the LLM enrichment prompt."""
    parts: List[str] = []

    for f in top_funcs:
        detail_lines = [f"METHOD: {f.qualname}", f"  File: {f.rel_path}, line {f.lineno}"]

        if code_index:
            rel = f.rel_path.replace("\\", "/")
            indexed = code_index.get(rel)
            if indexed:
                if indexed.llm_summary:
                    detail_lines.append(f"  File purpose: {indexed.llm_summary}")
                if indexed.docstring:
                    detail_lines.append(f"  Module docstring: {indexed.docstring[:150]}")
                for sig in indexed.func_signatures:
                    if sig.startswith(f.name + "("):
                        detail_lines.append(f"  Signature: {sig}")
                        break
                class_name = f.qualname.split(".")[0] if "." in f.qualname else None
                if class_name and class_name in indexed.class_bases:
                    bases = indexed.class_bases[class_name]
                    detail_lines.append(f"  Class {class_name} extends: {', '.join(bases)}")

        code = files.get(f.file_path, "")
        if code:
            source_lines = code.splitlines()
            start = f.lineno - 1
            snippet = source_lines[start:start + 8]
            if snippet:
                detail_lines.append("  Source preview:")
                for sl in snippet:
                    detail_lines.append(f"    {sl}")

        parts.append("\n".join(detail_lines))

    return "\n---\n".join(parts)

def _parse_enrichment_response(
    text: str,
    valid_qualnames: Set[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Parse LLM enrichment response into descriptions and groups.

    Returns:
      descriptions: {qualname: description}
      groups: {group_name: [qualnames]}
    """
    descriptions: Dict[str, str] = {}
    groups_raw: Dict[str, str] = {}

    current_method = None
    current_desc = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.upper().startswith("METHOD:"):
            candidate = stripped[7:].strip()
            if candidate in valid_qualnames:
                current_method = candidate
            else:
                current_method = None
                for qn in valid_qualnames:
                    if candidate in qn or qn in candidate:
                        current_method = qn
                        break

        elif stripped.upper().startswith("DESCRIPTION:") and current_method:
            current_desc = stripped[12:].strip()
            descriptions[current_method] = current_desc

        elif stripped.upper().startswith("GROUP:") and current_method:
            group_name = stripped[6:].strip()
            if group_name:
                groups_raw[current_method] = group_name
            current_method = None

    groups: Dict[str, List[str]] = {}
    for qn, gname in groups_raw.items():
        groups.setdefault(gname, []).append(qn)

    return descriptions, groups

def _enrich_via_llm(
    top_funcs: List[FunctionRecord],
    files: Dict[str, str],
    code_index: Optional[Dict[str, IndexedFile]],
    model_name: str = "llama3.2",
    max_retries: int = 2,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Ask LLM for method descriptions and grouping.
    Returns (descriptions, groups). Returns empty dicts on failure.
    """
    method_details = _build_method_details(top_funcs, code_index, files)
    valid_qualnames = {f.qualname for f in top_funcs}

    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(ENRICHMENT_PROMPT)
    chain = prompt | model

    for attempt in range(max_retries + 1):
        try:
            raw = str(chain.invoke({"method_details": method_details})).strip()
            descriptions, groups = _parse_enrichment_response(raw, valid_qualnames)
            enriched = len(descriptions)
            print(f"  LLM enrichment: {enriched}/{len(top_funcs)} methods described, "
                  f"{len(groups)} groups identified.")
            return descriptions, groups
        except Exception as e:
            if attempt < max_retries:
                print(f"  LLM enrichment failed, retrying in 3s... ({e})")
                time.sleep(3)
            else:
                print(f"  Warning: LLM enrichment failed after {max_retries + 1} attempts: {e}")

    return {}, {}

_COMPONENT_DECL_RE = re.compile(
    r'component\s+"[^"]+"\s+as\s+([A-Za-z_][A-Za-z0-9_]*)'
)
_EDGE_RE = re.compile(
    r'([A-Za-z_][A-Za-z0-9_]*)\s*-->\s*([A-Za-z_][A-Za-z0-9_]*)'
)
_STRING_EDGE_RE = re.compile(
    r'[A-Za-z_][A-Za-z0-9_]*\s*-->\s*"'
)

def _validate_plantuml_errors(text: str) -> List[str]:
    """Return list of specific validation errors. Empty list = valid."""
    errors: List[str] = []

    if not text or not text.strip():
        return ["Empty input"]

    if "```" in text:
        errors.append("Contains markdown code fences (```)")

    lines = text.strip().splitlines()
    non_blank = [l for l in lines if l.strip()]

    if not non_blank:
        return ["No non-blank lines"]

    if not non_blank[0].strip().lower().startswith("@startuml"):
        errors.append(f"First line is not @startuml: '{non_blank[0].strip()[:50]}'")
    if not non_blank[-1].strip().lower().startswith("@enduml"):
        errors.append(f"Last line is not @enduml: '{non_blank[-1].strip()[:50]}'")

    declared_ids = set(_COMPONENT_DECL_RE.findall(text))
    if not declared_ids:
        errors.append("No component declarations found")

    for match in _EDGE_RE.finditer(text):
        src_id, dst_id = match.group(1), match.group(2)
        if src_id not in declared_ids:
            errors.append(f"Edge source '{src_id}' is not a declared component ID")
        if dst_id not in declared_ids:
            errors.append(f"Edge target '{dst_id}' is not a declared component ID")

    if _STRING_EDGE_RE.search(text):
        errors.append("Edge uses string label instead of component ID")

    lower = text.lower()
    forbidden = ["participant", "activate", "deactivate", "->>", "actor", "usecase"]
    for tok in forbidden:
        if tok in lower:
            errors.append(f"Contains forbidden token: '{tok}'")

    return errors

def is_valid_dependency_uml(text: str) -> bool:
    """Check if text is valid PlantUML dependency diagram."""
    return len(_validate_plantuml_errors(text)) == 0

def _strip_llm_wrapper(text: str) -> str:
    """Remove markdown code fences and commentary from LLM output."""
    text = re.sub(r'```(?:plantuml|puml|uml|text)?\s*\n?', '', text)

    match = re.search(r'(@startuml.*?@enduml)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text.strip()

_LEGACY_GENERATOR_PROMPT = """
You are a diagram renderer.

TASK:
Generate a PlantUML DEPENDENCY DIAGRAM (static dependency graph).

MANDATORY SYNTAX:
- Each method MUST be declared as a component:
    component "<METHOD_NAME>" as <ID>
- Dependencies MUST be drawn ONLY using IDs:
    <ID1> --> <ID2>

STRICT PROHIBITIONS:
- Do NOT use: participant, actor, usecase, class
- Do NOT create lifelines
- Do NOT wrap output in markdown code fences
- Do NOT add any explanatory text before or after the diagram

OUTPUT RULES:
- Output ONLY valid PlantUML
- MUST start with @startuml
- MUST end with @enduml

INPUT:

METHODS:
{methods}

DEPENDENCIES:
{dependencies}
"""

_LEGACY_REVIEWER_PROMPT = """
You are a PlantUML reviewer and repair agent.

Fix the given PlantUML so that it becomes a VALID PlantUML DEPENDENCY DIAGRAM.

SPECIFIC ERRORS FOUND:
{errors}

MANDATORY REQUIREMENTS:
- Output ONLY PlantUML
- Start with @startuml, end with @enduml
- Use ONLY 'component' declarations: component "<NAME>" as <ID>
- Draw dependencies ONLY using IDs: <ID1> --> <ID2>
- Do NOT wrap in markdown code fences
- Do NOT add any explanatory text

INVALID INPUT:
{plantuml}

OUTPUT (VALID DEPENDENCY DIAGRAM ONLY):
"""

def _ensure_valid_plantuml_legacy(
    text: str,
    model_name: str,
    rounds: int = 3,
) -> str:
    """Legacy self-healing loop. Kept as fallback."""
    current = _strip_llm_wrapper(text)

    for _ in range(rounds):
        errors = _validate_plantuml_errors(current)
        if not errors:
            return current

        reviewer = (
            ChatPromptTemplate.from_template(_LEGACY_REVIEWER_PROMPT)
            | OllamaLLM(model=model_name)
        )

        raw = str(reviewer.invoke({
            "plantuml": current,
            "errors": "\n".join(f"- {e}" for e in errors),
        })).strip()
        current = _strip_llm_wrapper(raw)

    raise RuntimeError("Reviewer agent could not fix PlantUML output")

def generate_llm_dependency_graph(
    files: Dict[str, str],
    base_dir: str,
    output_file: str = "top_dependencies.puml",
    model_name: str = "llama3.2",
    top_n: int = 10,
    render_png: bool = True,
    render_svg: bool = True,
    code_index: Optional[Dict[str, IndexedFile]] = None,
):
    """
    Generates a METHOD-LEVEL DEPENDENCY DIAGRAM for TOP N important methods.

    Pipeline (hybrid deterministic + LLM enrichment):
      1. TOP-N selection (importance_analyzer)
      2. AST-based dependency collection (qualname-aware)
      3. LLM enrichment: descriptions + grouping (optional, graceful fallback)
      4. Deterministic PlantUML generation (always valid)
      5. Render to PNG/SVG
    """

    top_funcs = get_top_important_functions(files, base_dir, top_n=top_n)
    if not top_funcs:
        raise RuntimeError("No top functions found")

    print(f"Top {len(top_funcs)} functions identified:")
    for i, f in enumerate(top_funcs, 1):
        print(f"  {i}. {f.qualname} (importance={f.importance})")

    nodes, edges = collect_dependencies(files, top_funcs)
    print(f"Collected {len(edges)} dependency edges between top functions.")

    descriptions: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {}

    try:
        print("Enriching diagram with LLM descriptions and grouping...")
        descriptions, groups = _enrich_via_llm(
            top_funcs, files, code_index, model_name=model_name,
        )
    except Exception as e:
        print(f"  LLM enrichment skipped: {e}")

    uml = _build_deterministic_plantuml(
        nodes, edges,
        descriptions=descriptions if descriptions else None,
        groups=groups if groups else None,
    )

    errors = _validate_plantuml_errors(uml)
    if errors:
        print(f"Warning: deterministic PlantUML has errors (unexpected): {errors}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(uml)

    print(f"Dependency diagram written to: {output_file}")

    if render_png:
        try:
            png_path = render_plantuml(output_file, "png")
            print(f"PNG rendered: {png_path}")
        except Exception as e:
            print(f"PNG rendering failed: {e}")
    if render_svg:
        try:
            svg_path = render_plantuml(output_file, "svg")
            print(f"SVG rendered: {svg_path}")
        except Exception as e:
            print(f"SVG rendering failed: {e}")
