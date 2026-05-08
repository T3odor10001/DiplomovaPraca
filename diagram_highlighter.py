import re
import difflib
from typing import Dict, List, Set, Tuple, Optional

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

COLOR_EXACT = "#FFEE99"
COLOR_FUZZY = "#FFB347"
COLOR_LLM   = "#9EC9FF"

_COMPONENT_RE = re.compile(
    r'^(?P<indent>\s*)component\s+"(?P<label>[^"]+)"\s+as\s+(?P<id>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<rest>.*)$'
)

def extract_dependency_components(puml: str) -> Dict[str, str]:
    """Returns mapping: label -> id"""
    out: Dict[str, str] = {}
    for line in puml.splitlines():
        m = _COMPONENT_RE.match(line)
        if m:
            out[m.group("label")] = m.group("id")
    return out

def highlight_dependency_diagram(puml: str, label_to_color: Dict[str, str]) -> str:
    """Adds inline color to matching component declarations."""
    new_lines: List[str] = []
    for line in puml.splitlines():
        m = _COMPONENT_RE.match(line)
        if not m:
            new_lines.append(line)
            continue

        label = m.group("label")
        indent = m.group("indent")
        cid = m.group("id")
        rest = (m.group("rest") or "").rstrip()

        if label in label_to_color:
            color = label_to_color[label]
            rest_no_color = re.sub(r"\s+#\S+\s*", " ", rest).rstrip()
            colored = f'{indent}component "{label}" as {cid} {rest_no_color} {color}'.rstrip()
            new_lines.append(colored)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)

_CLASS_DECL_RE = re.compile(
    r'^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<rest>.*)$'
)
_METHOD_LINE_RE = re.compile(r'^\s*\+\s*(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*\(')

def extract_classes_and_methods(puml: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Returns:
      classes: set of class names
      methods_by_class: class -> set(method names)

    Lightweight parsing based on the generator output.
    """
    classes: Set[str] = set()
    methods_by_class: Dict[str, Set[str]] = {}
    current_class: Optional[str] = None

    for line in puml.splitlines():
        m = _CLASS_DECL_RE.match(line)
        if m:
            current_class = m.group("name")
            classes.add(current_class)
            methods_by_class.setdefault(current_class, set())
            continue

        if current_class:
            mm = _METHOD_LINE_RE.match(line)
            if mm:
                methods_by_class[current_class].add(mm.group("method"))

            if line.strip() == "}":
                current_class = None

    return classes, methods_by_class

def highlight_class_diagram(puml: str, class_to_color: Dict[str, str]) -> str:
    """Adds inline color to matching class declarations."""
    new_lines: List[str] = []
    for line in puml.splitlines():
        m = _CLASS_DECL_RE.match(line)
        if not m:
            new_lines.append(line)
            continue

        name = m.group("name")
        indent = m.group("indent")
        rest = (m.group("rest") or "").rstrip()

        if name in class_to_color:
            color = class_to_color[name]
            rest_no_color = re.sub(r"\s+#\S+\s*", " ", rest).rstrip()
            colored = f"{indent}class {name} {rest_no_color} {color}".rstrip()
            new_lines.append(colored)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def _alnum_words(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", _normalize(text)).strip()

def match_exact_dependency_labels(question: str, labels: List[str]) -> Set[str]:
    q = _normalize(question)
    hits: Set[str] = set()
    for lab in labels:
        if lab and _normalize(lab) in q:
            hits.add(lab)
    return hits

def match_exact_classes(question: str, classes: Set[str], methods_by_class: Dict[str, Set[str]]) -> Set[str]:
    q = _normalize(question)
    hits: Set[str] = set()

    for cls in classes:
        if _normalize(cls) in q:
            hits.add(cls)
            continue

        for m in methods_by_class.get(cls, set()):
            m_norm = _normalize(m)
            m_space = _normalize(m.replace("_", " "))
            if m_norm in q or m_space in q:
                hits.add(cls)
                break

    return hits

def _best_fuzzy_match(query: str, candidates: List[str]) -> Tuple[Optional[str], float]:
    """Returns (best_candidate, score in [0..1])."""
    if not candidates:
        return None, 0.0

    q = _alnum_words(query)
    best = None
    best_score = 0.0

    for c in candidates:
        c_norm = _alnum_words(c)
        if not c_norm:
            continue

        score = difflib.SequenceMatcher(None, q, c_norm).ratio()
        if score > best_score:
            best_score = score
            best = c

    return best, best_score

def match_fuzzy_dependency(question: str, labels: List[str], threshold: float = 0.70) -> Set[str]:
    """
    Fuzzy match for dependency labels.
    Strategy:
      - compare the whole question vs each label
      - if a label is close enough => highlight it
    """
    hits: Set[str] = set()
    q = _alnum_words(question)

    for lab in labels:
        lab_norm = _alnum_words(lab)
        if not lab_norm:
            continue
        score = difflib.SequenceMatcher(None, q, lab_norm).ratio()
        if score >= threshold:
            hits.add(lab)

    return hits

def match_fuzzy_classes(question: str, classes: Set[str], methods_by_class: Dict[str, Set[str]], threshold: float = 0.72) -> Set[str]:
    """
    Fuzzy match for class diagram.
    A class is selected if either:
      - class name is close to the question, OR
      - any method name is close to the question

    Note: This is intentionally simple + explainable.
    """
    hits: Set[str] = set()
    q = _alnum_words(question)

    for cls in classes:
        cls_norm = _alnum_words(cls)
        if cls_norm:
            if difflib.SequenceMatcher(None, q, cls_norm).ratio() >= threshold:
                hits.add(cls)
                continue

        for m in methods_by_class.get(cls, set()):
            m1 = _alnum_words(m)
            m2 = _alnum_words(m.replace("_", " "))
            if m1 and difflib.SequenceMatcher(None, q, m1).ratio() >= threshold:
                hits.add(cls)
                break
            if m2 and difflib.SequenceMatcher(None, q, m2).ratio() >= threshold:
                hits.add(cls)
                break

    return hits

_LLM_PICK_PROMPT = """
You are helping highlight the most relevant UML node.

You are given:
1) USER QUESTION
2) RELEVANT CODE FRAGMENTS
3) CANDIDATE UML NODE NAMES

Your task:
Select the single most relevant UML node based on the actual code logic.

Rules:
- Use the code fragments to understand which class implements the logic.
- Return EXACTLY ONE of the candidate names.
- Output ONLY the class name, no explanation.

USER QUESTION:
{question}

RELEVANT CODE:
{context}

CANDIDATE NODES:
{candidates}
"""

def llm_pick_best_node(
    question: str,
    candidates: List[str],
    context,
    model_name: str = "llama3.2"
) -> Optional[str]:

    if not candidates:
        return None

    context_text = context_to_text(context)

    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template("""
You are selecting the most relevant UML node.

USER QUESTION:
{question}

RELEVANT CODE FRAGMENTS:
{context}

CANDIDATE UML NODES:
{candidates}

Choose the SINGLE most relevant node.
Return ONLY its exact name.
""")

    chain = prompt | model

    candidates_text = "\n".join(f"- {c}" for c in candidates)

    result = chain.invoke({
        "question": question,
        "context": context_text,
        "candidates": candidates_text
    })

    picked = _normalize(str(result))

    cand_map = {_normalize(c): c for c in candidates}

    if picked in cand_map:
        return cand_map[picked]

    picked_clean = re.sub(r"[^a-z0-9_\- ]+", "", picked).strip()
    if picked_clean in cand_map:
        return cand_map[picked_clean]

    best, score = _best_fuzzy_match(picked, candidates)
    return best

def context_to_text(context, per_item_limit: int = 1500, total_limit: int = 6000) -> str:
    """
    Robust conversion of context into text for LLM prompt.
    Supports str, list[dict], list[str], dict, and fallback.
    """
    if context is None:
        return ""

    if isinstance(context, str):
        return context[:total_limit]

    parts: List[str] = []

    if isinstance(context, (list, tuple)):
        for item in context:
            if isinstance(item, dict):
                txt = item.get("content") or item.get("text") or item.get("snippet") or ""
            else:
                txt = str(item)
            if txt:
                parts.append(txt[:per_item_limit])
            if sum(len(p) for p in parts) >= total_limit:
                break

        joined = "\n\n".join(parts)
        return joined[:total_limit]

    if isinstance(context, dict):
        txt = context.get("content") or context.get("text") or str(context)
        return str(txt)[:total_limit]

    return str(context)[:total_limit]