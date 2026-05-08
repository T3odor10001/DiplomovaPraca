import os
import re
import ast
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from plantuml_generator import analyze_repo_structure
from full_classdiagram_generator import scan_repo_for_classes, ClassInfo
from code_indexer import IndexedFile
from plantuml_renderer import render_plantuml

@dataclass
class PatternSignals:
    """Aggregated structural signals extracted from the repository via AST."""
    module_count: int = 0
    class_count: int = 0
    function_count: int = 0
    classes: List[dict] = field(default_factory=list)
    import_graph: Dict[str, Set[str]] = field(default_factory=dict)
    decorators: Dict[str, List[str]] = field(default_factory=dict)
    abstract_classes: List[str] = field(default_factory=list)
    callback_methods: List[str] = field(default_factory=list)
    singleton_hints: List[str] = field(default_factory=list)
    layer_modules: Dict[str, List[str]] = field(default_factory=dict)

LAYER_KEYWORDS = {
    "controller": ["controller", "handler", "view", "endpoint", "route", "api"],
    "service": ["service", "usecase", "interactor", "workflow", "logic"],
    "repository": ["repository", "repo", "dao", "store", "persistence", "database", "db"],
    "model": ["model", "entity", "schema", "domain"],
    "util": ["util", "helper", "common", "shared", "lib", "tool"],
    "config": ["config", "settings", "env"],
    "test": ["test", "spec", "fixture"],
}

BEHAVIORAL_METHOD_PREFIXES = ("on_", "handle_", "notify_", "emit_", "dispatch_", "do_",
                              "execute_", "accept_", "visit_", "update_")

def _extract_base_names(node: ast.ClassDef) -> List[str]:
    """Extract base class names from a ClassDef node."""
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(base.attr)
    return bases

def _extract_decorator_names(node) -> List[str]:
    """Extract decorator names from a function or class node."""
    names = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(dec.attr)
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                names.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                names.append(dec.func.attr)
    return names

def collect_pattern_signals(files: Dict[str, str], base_dir: str) -> PatternSignals:
    """
    Walk all .py files with AST and collect structural signals
    useful for architecture / design-pattern recognition.
    """
    signals = PatternSignals()
    modules = analyze_repo_structure(files, base_dir)
    signals.module_count = len(modules)

    for mname, minfo in modules.items():
        signals.import_graph[mname] = set(minfo.imports)

    for mname in modules:
        lower = mname.lower()
        for layer, keywords in LAYER_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                signals.layer_modules.setdefault(layer, []).append(mname)

    for abs_path, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        rel = os.path.relpath(abs_path, base_dir)
        module_name = rel.replace("\\", ".").replace("/", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                signals.class_count += 1
                bases = _extract_base_names(node)
                methods = [n.name for n in node.body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                attrs = []
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for t in stmt.targets:
                            if isinstance(t, ast.Name):
                                attrs.append(t.id)

                signals.classes.append({
                    "name": node.name,
                    "module": module_name,
                    "methods": methods,
                    "attributes": attrs,
                    "bases": bases,
                })

                dec_names = _extract_decorator_names(node)
                if dec_names:
                    signals.decorators[node.name] = dec_names

                if any(b in ("ABC", "ABCMeta") for b in bases):
                    signals.abstract_classes.append(node.name)
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        mdecs = _extract_decorator_names(m)
                        if "abstractmethod" in mdecs:
                            if node.name not in signals.abstract_classes:
                                signals.abstract_classes.append(node.name)

                if "_instance" in attrs or "__new__" in methods:
                    signals.singleton_hints.append(node.name)

                for m in methods:
                    if any(m.startswith(prefix) for prefix in BEHAVIORAL_METHOD_PREFIXES):
                        signals.callback_methods.append(f"{node.name}.{m}")

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                signals.function_count += 1

    return signals

def format_signals_for_llm(
    signals: PatternSignals,
    files: Dict[str, str],
    base_dir: str,
    code_index: Optional[Dict[str, IndexedFile]] = None,
) -> str:
    """Build a concise textual summary of structural signals for the LLM prompt.
    When code_index is provided, includes LLM-generated file purpose descriptions."""
    lines: List[str] = []

    lines.append(f"REPOSITORY STATISTICS:")
    lines.append(f"  Modules: {signals.module_count}")
    lines.append(f"  Classes: {signals.class_count}")
    lines.append(f"  Functions/methods: {signals.function_count}")
    lines.append("")

    if code_index:
        entries_with_summary = [(p, e) for p, e in code_index.items() if e.llm_summary]
        if entries_with_summary:
            lines.append("MODULE PURPOSES (LLM-generated):")
            for rel_path, entry in sorted(entries_with_summary):
                lines.append(f"  {rel_path}: {entry.llm_summary}")
                if entry.docstring:
                    lines.append(f"    docstring: {entry.docstring[:150]}")
            lines.append("")

    if signals.layer_modules:
        lines.append("DETECTED MODULE LAYERS (by naming convention):")
        for layer, mods in sorted(signals.layer_modules.items()):
            lines.append(f"  {layer}: {', '.join(mods)}")
        lines.append("")

    lines.append("CLASS DETAILS:")
    for cls in sorted(signals.classes, key=lambda c: (c["module"], c["name"])):
        base_str = f" extends {', '.join(cls['bases'])}" if cls["bases"] else ""
        lines.append(f"  class {cls['name']}{base_str}  (module: {cls['module']})")
        if cls["methods"]:
            lines.append(f"    methods: {', '.join(cls['methods'])}")
        if cls["attributes"]:
            lines.append(f"    attributes: {', '.join(cls['attributes'])}")
    lines.append("")

    if signals.abstract_classes:
        lines.append(f"ABSTRACT CLASSES / ABCs: {', '.join(signals.abstract_classes)}")
        lines.append("")

    if signals.singleton_hints:
        lines.append(f"SINGLETON HINTS (classes with _instance or __new__): {', '.join(signals.singleton_hints)}")
        lines.append("")

    if signals.callback_methods:
        lines.append(f"BEHAVIORAL METHOD PATTERNS (on_*/handle_*/notify_*/visit_*/execute_*/...):")
        for m in signals.callback_methods:
            lines.append(f"  - {m}")
        lines.append("")

    if signals.decorators:
        lines.append("DECORATOR USAGE:")
        for name, decs in sorted(signals.decorators.items()):
            lines.append(f"  {name}: {', '.join(decs)}")
        lines.append("")

    lines.append("MODULE IMPORT GRAPH:")
    for mod, imports in sorted(signals.import_graph.items()):
        if imports:
            lines.append(f"  {mod} -> {', '.join(sorted(imports))}")
    lines.append("")

    return "\n".join(lines)

PATTERN_TEMPLATE = """
You are an expert software architect specializing in design pattern recognition.

You will receive a detailed structural analysis of a Python repository (classes, methods, attributes, inheritance, imports, module layers, decorators, etc.).

YOUR TASK:
Analyze the repository and identify:

1. **SYSTEM ARCHITECTURE PATTERN(S)** — the high-level architectural style of the project.
   Examples: Layered Architecture, MVC, MVP, MVVM, Pipeline/Pipe-and-Filter, Microservices,
   Monolithic, Event-Driven, Client-Server, Plugin/Extension, Repository Pattern, CQRS, etc.

2. **BEHAVIORAL DESIGN PATTERN(S)** — GoF behavioral patterns or similar recognized patterns
   found in the code.
   Examples: Observer, Strategy, Command, State, Template Method, Visitor, Iterator,
   Chain of Responsibility, Mediator, Memento, Interpreter, etc.

3. **OTHER DESIGN PATTERNS** — any creational or structural patterns you also detect
   (e.g. Singleton, Factory, Builder, Adapter, Decorator, Facade, Proxy, Composite, etc.).

OUTPUT FORMAT (strict — follow exactly):

## System Architecture

**Pattern:** <pattern name>
**Confidence:** <High / Medium / Low>
**Evidence:**
- <bullet point explaining WHY you think this pattern is used, referencing specific modules/classes>
- <more evidence>

(Repeat if multiple architectural patterns are detected.)

## Behavioral Design Patterns

**Pattern:** <pattern name>
**Confidence:** <High / Medium / Low>
**Evidence:**
- <bullet point with specific class/method names as evidence>
- <more evidence>

(Repeat for each behavioral pattern found.)

## Other Design Patterns

**Pattern:** <pattern name>
**Confidence:** <High / Medium / Low>
**Evidence:**
- <evidence>

(Repeat for each. If none found, write "None detected.")

## Summary

A short paragraph (3-5 sentences) summarizing the overall architectural style and the most
prominent design patterns, and how they work together in this project.

IMPORTANT RULES:
- Reference SPECIFIC class names, method names, and module names from the input as evidence.
- Do NOT invent patterns that are not supported by the structural evidence.
- If you are unsure about a pattern, use Low confidence and explain your reasoning.
- Output ONLY the analysis in the format above, no extra text.

REPOSITORY STRUCTURAL ANALYSIS:
{analysis}
"""

def create_pattern_chain(model_name: str = "llama3.2"):
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(PATTERN_TEMPLATE)
    return prompt | model

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}

def _parse_patterns(text: str) -> List[dict]:
    """
    Parse the LLM markdown output to extract patterns with confidence.
    Returns list of {name, confidence, confidence_rank, section} sorted by confidence descending.
    """
    patterns = []
    current_section = ""

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            current_section = stripped[3:].strip()

        elif stripped.startswith("**Pattern:**"):
            name = stripped.replace("**Pattern:**", "").strip()
            patterns.append({
                "name": name,
                "confidence": "",
                "confidence_rank": 0,
                "section": current_section,
            })

        elif stripped.startswith("**Confidence:**") and patterns:
            conf = stripped.replace("**Confidence:**", "").strip().lower()
            patterns[-1]["confidence"] = conf
            patterns[-1]["confidence_rank"] = _CONFIDENCE_RANK.get(conf, 0)

    patterns.sort(key=lambda p: p["confidence_rank"], reverse=True)
    return patterns

def get_top_architecture_pattern(text: str) -> Optional[str]:
    """Extract the highest-confidence system architecture pattern name."""
    patterns = _parse_patterns(text)
    for p in patterns:
        if "architecture" in p["section"].lower():
            return p["name"]
    return patterns[0]["name"] if patterns else None

ARCH_DIAGRAM_PROMPT = """
You are assigning modules and classes to architectural layers for a "{pattern}" architecture diagram.

Given the modules/classes below, assign EACH ONE to a layer/component of the "{pattern}" architecture.

MODULES AND CLASSES:
{modules_info}

OUTPUT FORMAT (strictly follow, one entry per module, no extra text):
MODULE: <module_name>
LAYER: <layer name that fits the "{pattern}" pattern>

Rules:
- Every module MUST be assigned to exactly one layer
- Layer names should reflect the "{pattern}" architecture (e.g., for MVC: "Model", "View", "Controller"; for Layered: "Presentation", "Business Logic", "Data Access", etc.)
- Use at most 6 different layer names
- Do NOT add explanations, just MODULE/LAYER pairs
"""

def _build_modules_info(
    signals: PatternSignals,
    code_index: Optional[Dict[str, IndexedFile]] = None,
) -> str:
    """Build a concise module list for the architecture diagram LLM prompt."""
    lines = []
    module_classes: Dict[str, List[str]] = {}
    for cls in signals.classes:
        module_classes.setdefault(cls["module"], []).append(cls["name"])

    for mod in signals.import_graph:
        if mod not in module_classes:
            module_classes[mod] = []

    for mod in sorted(module_classes):
        parts = [f"- {mod}"]
        cls_list = module_classes[mod]
        if cls_list:
            parts.append(f"  classes: {', '.join(cls_list)}")
        if code_index:
            for rel_path, entry in code_index.items():
                mod_from_path = rel_path.replace("/", ".").replace("\\", ".")
                if mod_from_path.endswith(".py"):
                    mod_from_path = mod_from_path[:-3]
                if mod_from_path == mod and entry.llm_summary:
                    parts.append(f"  purpose: {entry.llm_summary}")
                    break
        lines.append("\n".join(parts))

    return "\n".join(lines)

def _parse_layer_assignments(
    text: str,
    valid_modules: Set[str],
) -> Dict[str, List[str]]:
    """
    Parse LLM layer assignment response.
    Returns {layer_name: [module_names]}.
    """
    layers: Dict[str, List[str]] = {}
    current_module = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.upper().startswith("MODULE:"):
            candidate = stripped[7:].strip()
            if candidate in valid_modules:
                current_module = candidate
            else:
                current_module = None
                for mod in valid_modules:
                    if candidate in mod or mod in candidate:
                        current_module = mod
                        break

        elif stripped.upper().startswith("LAYER:") and current_module:
            layer = stripped[6:].strip()
            if layer:
                layers.setdefault(layer, []).append(current_module)
            current_module = None

    return layers

def _sanitize_id(name: str) -> str:
    """Create a PlantUML-safe ID from a module/class name."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)

def _build_architecture_plantuml(
    pattern_name: str,
    layers: Dict[str, List[str]],
    signals: PatternSignals,
) -> str:
    """Build a PlantUML architecture diagram deterministically."""
    lines = ["@startuml", ""]
    lines.append(f'title "{pattern_name}" Architecture')
    lines.append("")

    all_modules = set()
    for mods in layers.values():
        all_modules.update(mods)

    module_classes: Dict[str, List[str]] = {}
    for cls in signals.classes:
        if cls["module"] in all_modules:
            module_classes.setdefault(cls["module"], []).append(cls["name"])

    for layer_name, modules in layers.items():
        layer_id = _sanitize_id(layer_name)
        lines.append(f'package "{layer_name}" as {layer_id} {{')
        for mod in modules:
            mod_id = _sanitize_id(mod)
            classes = module_classes.get(mod, [])
            if classes:
                label = f"{mod}\\n({', '.join(classes[:3])})"
                if len(classes) > 3:
                    label += f"\\n+{len(classes) - 3} more"
            else:
                label = mod
            lines.append(f'  component "{label}" as {mod_id}')
        lines.append("}")
        lines.append("")

    for mod, imports in signals.import_graph.items():
        if mod not in all_modules:
            continue
        src_id = _sanitize_id(mod)
        for imp in imports:
            if imp in all_modules and imp != mod:
                dst_id = _sanitize_id(imp)
                lines.append(f"{src_id} --> {dst_id}")

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)

def generate_architecture_diagram(
    signals: PatternSignals,
    pattern_name: str,
    output_file: str = "architecture.puml",
    model_name: str = "llama3.2",
    code_index: Optional[Dict[str, IndexedFile]] = None,
    render_png: bool = True,
    render_svg: bool = True,
) -> Optional[str]:
    """
    Generate a PlantUML architecture diagram for the given pattern.

    1. Ask LLM to assign modules to architectural layers
    2. Build diagram deterministically
    3. Render to PNG/SVG

    Returns the path to the PNG file, or None on failure.
    """
    modules_info = _build_modules_info(signals, code_index)
    valid_modules = set(signals.import_graph.keys())

    for cls in signals.classes:
        valid_modules.add(cls["module"])

    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(ARCH_DIAGRAM_PROMPT)
    chain = prompt | model

    print(f"Generating architecture diagram for: {pattern_name}")

    layers = {}
    for attempt in range(3):
        try:
            raw = str(chain.invoke({
                "pattern": pattern_name,
                "modules_info": modules_info,
            })).strip()
            layers = _parse_layer_assignments(raw, valid_modules)
            if layers:
                break
        except Exception as e:
            if attempt < 2:
                print(f"  LLM layer assignment failed, retrying... ({e})")
                time.sleep(3)

    if not layers:
        print("  Warning: LLM could not assign module layers. Using heuristic layers.")
        if signals.layer_modules:
            layers = dict(signals.layer_modules)
        else:
            layers = {"Application": list(valid_modules)}

    assigned = sum(len(v) for v in layers.values())
    print(f"  Assigned {assigned} modules to {len(layers)} layers: {', '.join(layers.keys())}")

    uml = _build_architecture_plantuml(pattern_name, layers, signals)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(uml)

    print(f"Architecture diagram written to: {output_file}")

    png_path = None
    if render_png:
        try:
            png_path = str(render_plantuml(output_file, "png"))
            print(f"PNG rendered: {png_path}")
        except Exception as e:
            print(f"PNG rendering failed: {e}")
    if render_svg:
        try:
            render_plantuml(output_file, "svg")
        except Exception as e:
            print(f"SVG rendering failed: {e}")

    return png_path

def recognize_patterns(
    files: Dict[str, str],
    base_dir: str,
    output_file: str = "patterns.md",
    model_name: str = "llama3.2",
    code_index: Optional[Dict[str, IndexedFile]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Analyze the repository for system architecture and design patterns.

    1. Collects structural signals via AST
    2. Sends to LLM for pattern recognition
    3. Generates architecture diagram for the highest-confidence pattern
    4. Saves result to output_file and returns (text, architecture_png_path)
    """
    print("Collecting structural signals from repository...")
    signals = collect_pattern_signals(files, base_dir)

    if signals.class_count == 0 and signals.function_count == 0:
        msg = "No classes or functions found in the repository."
        print(msg)
        return msg, None

    analysis = format_signals_for_llm(signals, files, base_dir, code_index=code_index)

    print("Analyzing architecture and design patterns via LLM agent...")
    chain = create_pattern_chain(model_name=model_name)
    result = chain.invoke({"analysis": analysis})
    text = str(result).strip()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Pattern analysis written to: {output_file}")

    arch_png = None
    top_pattern = get_top_architecture_pattern(text)
    if top_pattern:
        print(f"Top architecture pattern: {top_pattern}")
        try:
            arch_png = generate_architecture_diagram(
                signals=signals,
                pattern_name=top_pattern,
                output_file="architecture.puml",
                model_name=model_name,
                code_index=code_index,
            )
        except Exception as e:
            print(f"Architecture diagram generation failed: {e}")
    else:
        print("No architecture pattern detected, skipping diagram.")

    return text, arch_png
