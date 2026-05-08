from __future__ import annotations

import os
import re
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class IndexedFile:
    rel_path: str
    abs_path: str
    code: str

    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)

    docstring: str = ""
    func_signatures: List[str] = field(default_factory=list)
    class_bases: Dict[str, List[str]] = field(default_factory=dict)

    llm_summary: str = ""

    summary: str = ""

def _safe_relpath(abs_path: str, base_dir: str) -> str:
    try:
        rel = os.path.relpath(abs_path, base_dir)
    except Exception:
        rel = os.path.basename(abs_path)
    return rel.replace("\\", "/")

def _extract_signals(code: str):
    """
    AST-based extraction of:
      - classes (with base classes)
      - module-level functions (with parameter names)
      - imports
      - module docstring
    """
    classes: List[str] = []
    functions: List[str] = []
    imports: Set[str] = set()
    docstring: str = ""
    func_signatures: List[str] = []
    class_bases: Dict[str, List[str]] = {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return classes, functions, imports, docstring, func_signatures, class_bases

    doc = ast.get_docstring(tree)
    if doc:
        docstring = doc[:300]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            if bases:
                class_bases[node.name] = bases

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            params = []
            for arg in node.args.args:
                if arg.arg != "self" and arg.arg != "cls":
                    params.append(arg.arg)
            sig = f"{node.name}({', '.join(params)})"
            func_signatures.append(sig)

        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name:
                    imports.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    classes = sorted(set(classes))
    functions = sorted(set(functions))
    return classes, functions, set(sorted(imports)), docstring, func_signatures, class_bases

def _build_signal_text(entry: IndexedFile) -> str:
    """Build a human-readable signal string from an IndexedFile."""
    sig = []
    if entry.classes:
        class_parts = []
        for c in entry.classes[:12]:
            bases = entry.class_bases.get(c)
            if bases:
                class_parts.append(f"{c}({', '.join(bases)})")
            else:
                class_parts.append(c)
        sig.append(f"classes: {', '.join(class_parts)}")
    if entry.func_signatures:
        sig.append(f"functions: {', '.join(entry.func_signatures[:12])}")
    elif entry.functions:
        sig.append(f"functions: {', '.join(entry.functions[:12])}")
    if entry.imports:
        sig.append(f"imports: {', '.join(list(sorted(entry.imports))[:12])}")
    if entry.docstring:
        sig.append(f"docstring: {entry.docstring[:150]}")
    return " | ".join(sig)

def build_code_index(files: Dict[str, str], base_dir: str) -> Dict[str, IndexedFile]:
    """
    Builds a code index of the whole repository (no LLM, fast).

    INPUT:
      files: {abs_path: code}

    OUTPUT:
      index: {rel_path: IndexedFile(...)}
    """
    index: Dict[str, IndexedFile] = {}

    for abs_path, code in files.items():
        abs_path = os.path.abspath(abs_path)
        rel_path = _safe_relpath(abs_path, base_dir)

        classes, functions, imports, docstring, func_signatures, class_bases = _extract_signals(code)

        entry = IndexedFile(
            rel_path=rel_path,
            abs_path=abs_path,
            code=code,
            classes=classes,
            functions=functions,
            imports=imports,
            docstring=docstring,
            func_signatures=func_signatures,
            class_bases=class_bases,
        )

        entry.summary = f"{rel_path} | {_build_signal_text(entry)}" if classes or functions else rel_path
        index[rel_path] = entry

    return index

_SUMMARY_BATCH_TEMPLATE = """
For each Python file below, write a 1-2 sentence PURPOSE description.
Focus on WHAT the file does and WHY it exists in the project (its role/responsibility).
Do NOT list class or function names — describe the purpose at a higher level.

Output format (strictly follow this, one entry per file):
FILE: <path>
PURPOSE: <1-2 sentence description>

{file_entries}
"""

def _generate_file_summaries_llm(
    entries: List[IndexedFile],
    model_name: str = "llama3.2",
    batch_size: int = 4,
    max_retries: int = 2,
) -> Dict[str, str]:
    """
    Generate LLM purpose summaries for a batch of IndexedFiles.
    Returns: {rel_path: llm_summary}
    Retries on connection errors (Ollama cold-start).
    """
    import time

    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(_SUMMARY_BATCH_TEMPLATE)
    chain = prompt | model

    results: Dict[str, str] = {}

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(entries) + batch_size - 1) // batch_size
        print(f"  Generating LLM summaries (batch {batch_num}/{total_batches})...")

        file_entries_parts = []
        for entry in batch:
            signals = _build_signal_text(entry)
            code_preview = "\n".join(entry.code.splitlines()[:80])
            file_entries_parts.append(
                f"FILE: {entry.rel_path}\n"
                f"Signals: {signals}\n"
                f"Code preview:\n{code_preview}\n"
            )

        file_entries_text = "\n---\n".join(file_entries_parts)

        for attempt in range(max_retries + 1):
            try:
                raw = str(chain.invoke({"file_entries": file_entries_text})).strip()
                current_path = None
                for line in raw.splitlines():
                    line_stripped = line.strip()
                    if line_stripped.upper().startswith("FILE:"):
                        current_path = line_stripped[5:].strip()
                    elif line_stripped.upper().startswith("PURPOSE:") and current_path:
                        purpose = line_stripped[8:].strip()
                        for entry in batch:
                            if entry.rel_path == current_path or entry.rel_path in current_path:
                                results[entry.rel_path] = purpose
                                break
                        current_path = None
                break
            except Exception as e:
                if attempt < max_retries:
                    print(f"  LLM connection failed, retrying in 3s... ({e})")
                    time.sleep(3)
                else:
                    print(f"  Warning: LLM summary batch failed after {max_retries + 1} attempts: {e}")

    return results

def build_enriched_code_index(
    files: Dict[str, str],
    base_dir: str,
    model_name: str = "llama3.2",
) -> Dict[str, IndexedFile]:
    """
    Builds a code index enriched with LLM-generated purpose summaries.

    1. Builds the standard index (AST-based, fast)
    2. Generates LLM summaries in batches
    3. Updates each entry's llm_summary and summary fields
    """
    index = build_code_index(files, base_dir)

    if not index:
        return index

    entries = list(index.values())
    print(f"Generating LLM file summaries for {len(entries)} files...")
    summaries = _generate_file_summaries_llm(entries, model_name=model_name)

    for rel_path, entry in index.items():
        llm_desc = summaries.get(rel_path, "")
        entry.llm_summary = llm_desc

        signal_text = _build_signal_text(entry)
        if llm_desc:
            entry.summary = f"{rel_path}: {llm_desc} | {signal_text}"
        else:
            entry.summary = f"{rel_path} | {signal_text}" if signal_text else rel_path

    enriched_count = sum(1 for e in index.values() if e.llm_summary)
    print(f"Enriched {enriched_count}/{len(index)} files with LLM summaries.")

    return index
