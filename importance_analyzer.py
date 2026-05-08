import ast
import logging
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple

@dataclass
class FunctionRecord:
    file_path: str
    rel_path: str
    qualname: str
    name: str
    lineno: int
    importance: float = 0.0

class RepoAnalyzer(ast.NodeVisitor):
    """
    Pre jeden .py súbor:
    - nazbiera všetky funkcie/metódy
    - nazbiera všetky volania (Call) kvôli 'dependents'
    """

    def __init__(self, file_path: str, base_dir: str):
        self.file_path = file_path
        self.rel_path = file_path.replace(base_dir + "\\", "").replace(base_dir + "/", "")
        self._class_stack: List[str] = []
        self.functions: List[Tuple[FunctionRecord, ast.AST]] = []
        self.called_names: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        qualname = ".".join(self._class_stack + [node.name]) if self._class_stack else node.name
        rec = FunctionRecord(
            file_path=self.file_path,
            rel_path=self.rel_path,
            qualname=qualname,
            name=node.name,
            lineno=node.lineno,
        )
        self.functions.append((rec, node))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        qualname = ".".join(self._class_stack + [node.name]) if self._class_stack else node.name
        rec = FunctionRecord(
            file_path=self.file_path,
            rel_path=self.rel_path,
            qualname=qualname,
            name=node.name,
            lineno=node.lineno,
        )
        self.functions.append((rec, node))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name:
            self.called_names.append(name)
        self.generic_visit(node)

def _compute_function_metrics(node: ast.AST) -> Tuple[int, int, int, int, int]:
    """
    Vypočíta základné metriky pre jednu funkciu/metódu.

    Vracia:
      method_count (1),
      call_count,
      loc (počet riadkov),
      attr_count,
      complexity (hrubý odhad cyklomatickej zložitosti)
    """
    method_count = 1
    call_count = sum(isinstance(n, ast.Call) for n in ast.walk(node))

    if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
        loc = max(1, int(node.end_lineno) - int(node.lineno) + 1)
    else:
        loc = 1

    attr_count = sum(isinstance(n, ast.Attribute) for n in ast.walk(node))

    complexity_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.BoolOp)
    complexity = sum(isinstance(n, complexity_nodes) for n in ast.walk(node))

    return method_count, call_count, loc, attr_count, complexity

def _calculate_importance_index(
    method_count: int,
    call_count: int,
    loc: int,
    attr_count: int,
    complexity: int,
    dependents: int,
    total_functions: int,
) -> float:
    
    norm_dependents = dependents / max(1, total_functions)
    norm_dependents = norm_dependents ** 1.5
    normalized_loc = loc / 10.0

    index = (
        0.25 * method_count +
        0.15 * call_count +
        0.10 * normalized_loc +
        0.10 * attr_count +
        0.10 * complexity +
        0.30 * norm_dependents
    )
    return round(index, 2)

def print_top_important_functions(
    files: Dict[str, str],
    base_dir: str,
    top_n: int = 10,
) -> None:
    """
    Prejde všetky .py súbory v `files` (dict: abs_path -> obsah),
    spočíta index dôležitosti pre každú funkciu/metódu
    a v CLI vypíše TOP N najdôležitejších.

    base_dir = root repozitára 
    """
    all_func_records: List[FunctionRecord] = []
    func_nodes: List[ast.AST] = []
    all_called_names: List[str] = []

    for file_path, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logging.warning(f"SyntaxError v súbore, preskakujem: {file_path}")
            continue

        analyzer = RepoAnalyzer(file_path=file_path, base_dir=base_dir)
        analyzer.visit(tree)

        for rec, node in analyzer.functions:
            all_func_records.append(rec)
            func_nodes.append(node)

        all_called_names.extend(analyzer.called_names)

    if not all_func_records:
        print("V repozitári neboli nájdené žiadne funkcie/metódy.")
        return

    total_functions = len(all_func_records)
    call_counter = Counter(all_called_names)

    for rec, node in zip(all_func_records, func_nodes):
        method_count, call_count, loc, attr_count, complexity = _compute_function_metrics(node)
        dependents = call_counter.get(rec.name, 0)

        rec.importance = _calculate_importance_index(
            method_count=method_count,
            call_count=call_count,
            loc=loc,
            attr_count=attr_count,
            complexity=complexity,
            dependents=dependents,
            total_functions=total_functions,
        )

    top_funcs = sorted(all_func_records, key=lambda r: r.importance, reverse=True)[:top_n]

    print(f"\nTop {len(top_funcs)} most important functions/methods in repository:\n")
    for idx, rec in enumerate(top_funcs, start=1):
        print(
            f"{idx}. [index={rec.importance}] {rec.qualname}  "
            f"(file: {rec.rel_path}, line: {rec.lineno})"
        )
def get_top_important_functions(
    files: Dict[str, str],
    base_dir: str,
    top_n: int = 10,
) -> List[FunctionRecord]:
    """
    Vráti TOP N najdôležitejších funkcií/metód v repozitári ako zoznam FunctionRecord.
    """
    all_func_records: List[FunctionRecord] = []
    func_nodes: List[ast.AST] = []
    all_called_names: List[str] = []

    for file_path, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logging.warning(f"SyntaxError v súbore, preskakujem: {file_path}")
            continue

        analyzer = RepoAnalyzer(file_path=file_path, base_dir=base_dir)
        analyzer.visit(tree)

        for rec, node in analyzer.functions:
            all_func_records.append(rec)
            func_nodes.append(node)

        all_called_names.extend(analyzer.called_names)

    if not all_func_records:
        return []

    total_functions = len(all_func_records)
    call_counter = Counter(all_called_names)

    for rec, node in zip(all_func_records, func_nodes):
        method_count, call_count, loc, attr_count, complexity = _compute_function_metrics(node)
        dependents = call_counter.get(rec.name, 0)

        rec.importance = _calculate_importance_index(
            method_count=method_count,
            call_count=call_count,
            loc=loc,
            attr_count=attr_count,
            complexity=complexity,
            dependents=dependents,
            total_functions=total_functions,
        )

    top_funcs = sorted(all_func_records, key=lambda r: r.importance, reverse=True)[:top_n]
    return top_funcs

def print_top_important_functions(
    files: Dict[str, str],
    base_dir: str,
    top_n: int = 10,
) -> None:
    """
    Pôvodná funkcia na výpis TOP N, teraz len využíva get_top_important_functions.
    """
    top_funcs = get_top_important_functions(files, base_dir, top_n=top_n)

    if not top_funcs:
        print("V repozitári neboli nájdené žiadne funkcie/metódy.")
        return

    print(f"\nTop {len(top_funcs)} most important functions/methods in repository:\n")
    for idx, rec in enumerate(top_funcs, start=1):
        print(
            f"{idx}. [index={rec.importance}] {rec.qualname}  "
            f"(file: {rec.rel_path}, line: {rec.lineno})"
        )