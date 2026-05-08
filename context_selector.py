from __future__ import annotations

import re
import ast
import json
from typing import Dict, List, Tuple, Optional

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from code_indexer import IndexedFile

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

def _keywords(question: str) -> List[str]:
    kws = [
        w.lower()
        for w in _WORD_RE.findall(question or "")
        if len(w) >= 3
    ]
    seen = set()
    out = []
    for k in kws:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

_KEYWORD_EXPANSION_TEMPLATE = """
Given this question about a Python codebase, list related technical keywords, synonyms, and common code identifiers that might appear in source files relevant to answering it.

Think about:
- Synonyms (authenticate -> login, auth, credentials)
- Related concepts (database -> query, connection, cursor, session)
- Common naming patterns (e.g. snake_case and CamelCase variants)

Question: {question}

Return ONLY a comma-separated list of lowercase keywords (10-20 keywords). No explanations, no numbering.
"""

def _expand_keywords_llm(question: str, model_name: str = "llama3.2") -> List[str]:
    """
    Use LLM to expand the user's question into related technical keywords/synonyms.
    Returns a list of expanded keywords (lowercase, identifier-like).
    """
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(_KEYWORD_EXPANSION_TEMPLATE)
    chain = prompt | model

    try:
        raw = str(chain.invoke({"question": question})).strip()
        expanded = []
        for token in raw.split(","):
            token = token.strip().lower()
            token = re.sub(r"[^a-z0-9_]", "", token)
            if len(token) >= 3:
                expanded.append(token)
        return expanded[:25]
    except Exception:
        return []

def _get_expanded_keywords(
    question: str,
    conversation_history: Optional[List[dict]],
    use_llm: bool,
    model_name: str,
) -> List[str]:
    """
    Build an expanded keyword list from:
    1) Current question (regex)
    2) Last 2 questions from conversation history (regex)
    3) LLM keyword expansion (if use_llm=True)
    """
    kws = _keywords(question)
    seen = set(kws)

    if conversation_history:
        recent_questions = [
            m["content"] for m in conversation_history[-4:]
            if m.get("role") == "user"
        ][-2:]
        for q in recent_questions:
            for k in _keywords(q):
                if k not in seen:
                    kws.append(k)
                    seen.add(k)

    if use_llm:
        expanded = _expand_keywords_llm(question, model_name=model_name)
        for k in expanded:
            if k not in seen:
                kws.append(k)
                seen.add(k)

    return kws

def _score_file(entry: IndexedFile, kws: List[str], question: str) -> float:
    """
    Explainable scoring:
      - keyword hits in code
      - keyword hits in file path
      - keyword hits in class/function/import signals
      - keyword hits in LLM summary (if available)
    """
    if not kws:
        return 0.0

    code_lower = (entry.code or "").lower()
    path_lower = (entry.rel_path or "").lower()

    sig_text = " ".join(entry.classes + entry.functions + list(entry.imports)).lower()
    llm_text = (entry.llm_summary or "").lower()

    score = 0.0
    for k in kws:
        if k in path_lower:
            score += 3.0
        if k in sig_text:
            score += 2.0
        if k in llm_text:
            score += 2.5
        if k in code_lower:
            score += 1.0

    if "." in (question or ""):
        score += 0.2

    return score

def _extract_ast_relevant_blocks(code: str, kws: List[str], max_blocks: int = 6) -> List[str]:
    """
    Skusi AST:
    - vybrat class/function definicie, ktore obsahuju keyword v nazve alebo v tele
    - vrati segmenty (source) ak sa daju ziskat
    """
    blocks: List[str] = []
    if not code or not kws:
        return blocks

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return blocks

    code_lower = code.lower()

    def node_name(n: ast.AST) -> Optional[str]:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return n.name
        return None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        name = node_name(node) or ""
        name_l = name.lower()

        hit = any(k in name_l for k in kws)
        if not hit:
            hit = any(k in code_lower for k in kws)

        if not hit:
            continue

        seg = ast.get_source_segment(code, node)
        if seg:
            blocks.append(seg.strip())
            if len(blocks) >= max_blocks:
                break

    return blocks

def _extract_grep_windows(code: str, kws: List[str], window: int = 18, max_windows: int = 6) -> List[str]:
    """
    Ak AST nevie vratit segmenty (alebo je to malo),
    vyrobime okna okolo riadkov, kde sa keyword nachadza.
    """
    if not code or not kws:
        return []

    lines = code.splitlines()
    hits: List[Tuple[int, str]] = []

    for i, line in enumerate(lines):
        ll = line.lower()
        if any(k in ll for k in kws):
            hits.append((i, line))
            if len(hits) >= max_windows * 3:
                break

    windows: List[str] = []
    used_ranges: List[Tuple[int, int]] = []

    for i, _ in hits:
        start = max(0, i - window)
        end = min(len(lines), i + window)

        skip = False
        for a, b in used_ranges:
            if start >= a and end <= b:
                skip = True
                break
        if skip:
            continue

        used_ranges.append((start, end))
        snippet = "\n".join(lines[start:end]).strip()
        if snippet:
            windows.append(snippet)
        if len(windows) >= max_windows:
            break

    return windows

def _pack_context(selected: List[IndexedFile], question: str, kws: List[str], max_chars: int) -> str:
    """
    Posklada kontext tak, aby:
      - bol v limite
      - obsahoval len relevantne casti (AST segmenty + grep okna)
    """
    parts: List[str] = []
    total = 0

    for entry in selected:
        header = f"# FILE: {entry.rel_path}\n"
        blocks = []

        ast_blocks = _extract_ast_relevant_blocks(entry.code, kws, max_blocks=5)
        grep_blocks = _extract_grep_windows(entry.code, kws, window=16, max_windows=4)

        for b in ast_blocks:
            blocks.append(b)
        for b in grep_blocks:
            if b not in blocks:
                blocks.append(b)

        if not blocks:
            lines = entry.code.splitlines()
            blocks = ["\n".join(lines[:120]).strip()]

        body = "\n\n---\n\n".join(blocks).strip()
        block = header + body + "\n"

        if total + len(block) > max_chars and parts:
            break

        parts.append(block)
        total += len(block)

        if total >= max_chars:
            break

    return "\n\n".join(parts).strip()

_PICKER_TEMPLATE = """
You are selecting which repository files should be sent as context to answer a user question.

You will receive:
1) USER QUESTION
2) CONVERSATION CONTEXT (previous Q&A, if any)
3) CANDIDATE FILES with purpose descriptions and structural signals

Return STRICT JSON ONLY in this format:
{{
  "selected": ["path1", "path2", ...],
  "reasoning": "very short explanation of why these files are most relevant"
}}

Rules:
- Select 2 to {max_files} files.
- Prefer files that contain the IMPLEMENTATION needed to answer the question.
- Consider the purpose descriptions — a file about "authentication middleware" is relevant to a question about "how login works" even if the word "login" does not appear in its path.
- For follow-up questions, consider what was discussed in previous conversation turns.
- Use ONLY paths that appear in the candidate list.
- Output JSON only. No markdown.

USER QUESTION:
{question}

CONVERSATION CONTEXT:
{history}

CANDIDATE FILES:
{candidates}
"""

def _format_conversation_history(history: Optional[List[dict]], max_entries: int = 4) -> str:
    """Format conversation history for inclusion in prompts."""
    if not history:
        return "(no previous conversation)"

    recent = history[-max_entries:]
    parts = []
    for msg in recent:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        if role == "ASSISTANT" and len(content) > 400:
            content = content[:400] + "..."
        parts.append(f"{role}: {content}")

    return "\n".join(parts) if parts else "(no previous conversation)"

def _llm_pick_files(
    question: str,
    candidates: List[IndexedFile],
    max_files: int,
    model_name: str,
    conversation_history: Optional[List[dict]] = None,
) -> Optional[List[str]]:
    if not candidates:
        return None

    candidates_text = "\n".join(f"- {c.summary}" for c in candidates)
    history_text = _format_conversation_history(conversation_history)

    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(_PICKER_TEMPLATE)
    chain = prompt | model

    raw = str(
        chain.invoke(
            {
                "question": question,
                "candidates": candidates_text,
                "max_files": max_files,
                "history": history_text,
            }
        )
    ).strip()

    try:
        data = json.loads(raw)
        selected = data.get("selected")
        if isinstance(selected, list) and all(isinstance(x, str) for x in selected):
            return selected[:max_files]
    except Exception:
        match = re.search(r'\{[^{}]*"selected"\s*:\s*\[.*?\][^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                selected = data.get("selected")
                if isinstance(selected, list) and all(isinstance(x, str) for x in selected):
                    return selected[:max_files]
            except Exception:
                pass

    return None

def select_context(
    question: str,
    code_index: Dict[str, IndexedFile],
    max_chars: int = 6000,
    *,
    model_name: str = "llama3.2",
    shortlist_size: int = 12,
    max_files: int = 6,
    use_llm: bool = True,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """
    1) expands keywords via LLM (synonyms, related terms)
    2) deterministicky zoradi subory podla skore (klucove slova + signaly + LLM summaries)
    3) spravi shortlist
    4) LLM vyberie finalne subory (with conversation context)
    5) do kontextu vlozi len relevantne bloky (nie cele subory)
    """
    kws = _get_expanded_keywords(question, conversation_history, use_llm, model_name)

    entries = list(code_index.values())

    scored: List[Tuple[float, IndexedFile]] = []
    for e in entries:
        s = _score_file(e, kws, question)
        scored.append((s, e))

    scored.sort(key=lambda x: x[0], reverse=True)

    shortlist = [e for s, e in scored if s > 0][:shortlist_size]
    if not shortlist:
        shortlist = [e for _, e in scored[:min(shortlist_size, len(scored))]]

    selected_entries: List[IndexedFile] = []
    if use_llm:
        picked = _llm_pick_files(
            question, shortlist,
            max_files=max_files,
            model_name=model_name,
            conversation_history=conversation_history,
        )
        if picked:
            picked_set = set(picked)
            selected_entries = [e for e in shortlist if e.rel_path in picked_set]

    if not selected_entries:
        selected_entries = shortlist[:max_files]

    return _pack_context(selected_entries, question=question, kws=kws, max_chars=max_chars)
