import os
import shutil
import logging
import tempfile
from urllib.parse import urlparse

from git import Repo
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from importance_analyzer import print_top_important_functions
from classdiagram_generator import generate_classdiagram_for_repo
from full_classdiagram_generator import generate_full_classdiagram
from install_guide_generator import generate_installation_guide
from langgraph_workflows import build_doc_review_graph
from plantuml_renderer import render_plantuml

from code_indexer import build_code_index, build_enriched_code_index
from context_selector import select_context
from code_explainer import explain_code

from pattern_recognizer import recognize_patterns

from diagram_highlighter import (
    extract_dependency_components,
    highlight_dependency_diagram,
    extract_classes_and_methods,
    highlight_class_diagram,
    match_exact_dependency_labels,
    match_exact_classes,
    match_fuzzy_dependency,
    match_fuzzy_classes,
    llm_pick_best_node,
    COLOR_EXACT,
    COLOR_FUZZY,
    COLOR_LLM,
)

logging.basicConfig(level=logging.INFO)

def normalize_repo_url(url: str) -> str:
    """
    Upraví GitHub URL do tvaru vhodného pre `git clone`.
    """
    url = url.strip()
    if url.endswith(".git"):
        return url

    parsed = urlparse(url)
    if "github.com" in parsed.netloc:
        path = parsed.path.rstrip("/")
        for marker in ("/tree/", "/blob/"):
            idx = path.find(marker)
            if idx != -1:
                path = path[:idx]
                break
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            user, repo = parts[0], parts[1]
            return f"https://github.com/{user}/{repo}"
    return url

class RepositoryReader:
    """
    Klonuje Git repozitár a číta všetky .py súbory.
    """

    def __init__(self, repo_url: str, clone_dir: str = None):
        self.original_url = repo_url
        self.repo_url = normalize_repo_url(repo_url)
        self.clone_dir = clone_dir or tempfile.mkdtemp(prefix="cloned_repo_")
        self.local_path = os.path.abspath(self.clone_dir)

    def clone_repository(self):
        """
        Naklonuje repozitár do self.clone_dir. Ak existuje a nie je prázdny, vymaže ho.
        """
        if os.path.exists(self.clone_dir) and os.listdir(self.clone_dir):
            logging.info(f"Removing existing directory contents: {self.clone_dir}")
            shutil.rmtree(self.clone_dir)
        os.makedirs(self.clone_dir, exist_ok=True)

        print(f"Cloning repository from {self.repo_url} into {self.clone_dir}...")
        try:
            Repo.clone_from(self.repo_url, self.clone_dir)
        except Exception as e:
            logging.error("Klonovanie zlyhalo.", exc_info=True)
            raise RuntimeError(
                f"Nepodarilo sa naklonovať repozitár z URL: {self.original_url} "
                f"(normalized: {self.repo_url}). Skontrolujte prosím URL alebo pripojenie."
            ) from e

    def read_files(self) -> dict:
        """
        Načíta všetky .py súbory z repozitára.
        Vráti slovník: {abs_path: source_code}
        """
        files_dict = {}

        for root, _, files in os.walk(self.clone_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        files_dict[os.path.abspath(file_path)] = f.read()

                except UnicodeDecodeError:
                    logging.warning(f"Nepodarilo sa prečítať súbor: {file_path}")

        return files_dict
    def delete_repository(self):
        """
        Vymaže lokálny temp klon. Ak zlyhá, len warning.
        """
        if os.path.exists(self.clone_dir):
            try:
                shutil.rmtree(self.clone_dir, ignore_errors=False)
            except Exception as e:
                logging.warning(
                    f"Nepodarilo sa vymazať priečinok '{self.clone_dir}', preskakujem. Dôvod: {e}"
                )

DOCWRITER_TEMPLATE = """
You are the Documentation Maker.

TASK:
Write clear, correct documentation for each function, method, and class in the given Python file.

OUTPUT STYLE (English):
- Start with: "## File: {path}"
- For each function/method:
  - **Name**: <name>
  - **Parameters**: <param list with short explanations>
  - **Description**: 1–3 sentences
- For each class:
  - **Class**: <name> — short description
  - **Key Methods**: document the key ones (same style as functions)

CONSTRAINTS:
- Do NOT paste code, only documentation.
- Ignore empty/import-only files.

If feedback from a reviewer is provided, incorporate it.

Reviewer feedback (may be empty):
{feedback}

Source file path: {path}

Source code:
{code}
"""

REVIEWER_TEMPLATE = """
You are the Documentation Reviewer.

Given the original source code and the proposed documentation, check:
- Coverage: all non-trivial functions, methods, and classes are documented.
- Correctness: descriptions reflect what the code actually does.
- Clarity: concise, readable, structured as requested.
- Parameters: key parameters are captured with short explanations.

Return your decision in EXACT format:

DECISION: APPROVE
FEEDBACK: <optional short praise or tiny nits>

-- or, if issues exist --

DECISION: REVISE
FEEDBACK: <bullet points of what to fix/add/remove; be specific and concise>

Now review.

File: {path}

SOURCE CODE:
{code}

PROPOSED DOCUMENTATION:
{doc}
"""

def make_docwriter(model_name: str = "llama3.2"):
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(DOCWRITER_TEMPLATE)
    return prompt | model

def make_reviewer(model_name: str = "llama3.2"):
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(REVIEWER_TEMPLATE)
    return prompt | model

def parse_decision(text: str):
    upper = text.upper()
    if "DECISION: REVISE" in upper:
        decision = "REVISE"
    elif "DECISION: APPROVE" in upper:
        decision = "APPROVE"
    else:
        decision = None

    feedback = ""
    if "FEEDBACK:" in upper:
        idx = upper.find("FEEDBACK:")
        feedback = text[idx + len("FEEDBACK:") :].strip()

    return decision, feedback

def choose_single_file(files: dict, base_dir: str) -> str:
    if not files:
        raise RuntimeError("V repozitári neboli nájdené žiadne .py súbory.")

    rel_paths = sorted(os.path.relpath(p, base_dir) for p in files.keys())

    print("\nDostupné .py súbory:")
    for idx, rel in enumerate(rel_paths, start=1):
        print(f"[{idx}] {rel}")

    choice = input("\nZadaj číslo súboru pre ktorý chceš vygenerovať dokumentáciu: ").strip()
    if not choice.isdigit():
        raise RuntimeError("Neplatná voľba (nie je číslo).")

    choice_idx = int(choice)
    if not (1 <= choice_idx <= len(rel_paths)):
        raise RuntimeError("Neplatná voľba (mimo rozsah).")

    selected_rel = rel_paths[choice_idx - 1]
    selected_abs = os.path.join(base_dir, selected_rel)
    return selected_abs

def document_with_review(path: str, code: str, docwriter, reviewer, max_rounds: int = 2):
    documentation = docwriter.invoke({"path": path, "code": code, "feedback": ""})
    last_review = ""

    for _ in range(max_rounds):
        review_text = reviewer.invoke({"path": path, "code": code, "doc": str(documentation)})
        last_review = review_text
        decision, feedback = parse_decision(str(review_text))

        if decision == "APPROVE":
            return str(documentation), str(last_review)

        fix_feedback = feedback or "Please fix coverage, correctness, clarity, and parameter notes as needed."
        documentation = docwriter.invoke({"path": path, "code": code, "feedback": fix_feedback})

    return str(documentation), str(last_review)

def generate_docs_for_repo(
    reader: RepositoryReader,
    files: dict,
    output_file: str,
    model_name: str = "llama3.2",
    mode: str = "1",
    enable_review: bool = True,
    max_review_rounds: int = 2,
):
    if mode not in ("1", "2"):
        raise ValueError("mode must be '1' or '2'")

    if mode == "2":
        selected_file = choose_single_file(files, reader.clone_dir)
        files = {selected_file: files[selected_file]}
        print(f"\nVybraný súbor: {os.path.relpath(selected_file, reader.clone_dir)}")

    docwriter = make_docwriter(model_name)
    reviewer = make_reviewer(model_name) if enable_review else None

    doc_graph = build_doc_review_graph(docwriter, reviewer) if enable_review and reviewer is not None else None

    print(f"\nGenerating documentation for {len(files)} file(s)... (review={'on' if enable_review else 'off'})")

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"Documentation for repository: {reader.original_url}\n")
        out.write(f"(Mode: {'single file' if mode=='2' else 'all .py files'})\n")
        out.write(f"(Review: {'enabled' if enable_review else 'disabled'})\n\n")

        for file_path, code in sorted(files.items()):
            rel_path = os.path.relpath(file_path, reader.clone_dir)
            print(f"- Processing {rel_path}")

            if not code.strip():
                continue

            if enable_review and doc_graph is not None:
                out_state = doc_graph.invoke(
                    {
                        "path": rel_path,
                        "code": code,
                        "round": 0,
                        "max_rounds": max_review_rounds,
                        "feedback_for_writer": "",
                    }
                )
                final_doc = out_state.get("documentation", "")
            else:
                final_doc = docwriter.invoke({"path": rel_path, "code": code, "feedback": ""})

            out.write(f"## File: {rel_path}\n\n")
            out.write(str(final_doc).strip())
            out.write("\n\n")

    print(f"\n✅ Documentation written to: {output_file}")

def main():
    repo_url_input = input("Enter GitHub repository URL: ").strip()
    if not repo_url_input:
        raise SystemExit("No URL provided, exiting.")

    print("\nWhat do you want to do?")
    print("[1] Generate documentation")
    print("[2] Show top 10 most important methods/functions")
    print("[3] Generate PlantUML CLASS diagram from top 10 functions")
    print("[4] Generate FULL SYSTEM class diagram")
    print("[5] Generate INSTALLATION GUIDE for this repository")
    print("[6] Interactive code explanation")
    print("[7] Generate dependency graph of top 10 functions")
    print("[8] Recognize architecture & design patterns")

    main_mode = input("Choose option [1/2/3/4/5/6/7/8]: ").strip() or "1"

    if main_mode not in ("1", "2", "3", "4", "5", "6", "7", "8"):
        raise SystemExit("Invalid option. Use 1, 2, 3, 4, 5, 6, 7 or 8.")

    reader = RepositoryReader(repo_url=repo_url_input)
    reader.clone_repository()

    try:
        files = reader.read_files()
        if not files:
            print("No Python files found in the repository.")
            return

        code_index = None
        if main_mode in ("3", "4", "6", "7", "8"):
            code_index = build_enriched_code_index(files, reader.clone_dir, model_name="llama3.2")

        if main_mode == "2":
            print_top_important_functions(
                files=files,
                base_dir=reader.clone_dir,
                top_n=10,
            )
        
        elif main_mode == "3":
            output_puml = input(
                "Output PlantUML CLASS diagram file (default: classes.puml): "
            ).strip() or "classes.puml"
            model_name = "llama3.2"
            generate_classdiagram_for_repo(
                files=files,
                base_dir=reader.clone_dir,
                output_file=output_puml,
                model_name=model_name,
                code_index=code_index,
            )
            rendered = render_plantuml(output_puml, "png")
            print(f"🖼 Diagram rendered to: {rendered}")
        elif main_mode == "4":
            output_puml = input("Output file (default full_system.puml): ").strip() or "full_system.puml"
            
            generate_full_classdiagram(files, reader.clone_dir, output_file=output_puml, code_index=code_index)
            rendered = render_plantuml(output_puml, "png")
            print(f"🖼 Diagram rendered to: {rendered}")

        elif main_mode == "5":
            output_install = input(
                "Output installation guide file (default: INSTALLATION.md): "
            ).strip() or "INSTALLATION.md"
            generate_installation_guide(
                base_dir=reader.clone_dir,
                output_file=output_install,
                model_name="llama3.2",
            )

        elif main_mode == "6":
            print("\n====== FULL CODE INDEX (LLM-enriched) ======\n")

            for path, entry in code_index.items():
                llm_desc = f" — {entry.llm_summary}" if entry.llm_summary else ""
                print(f"  {path}{llm_desc}")

            print("\n=============================\n")

            print("Interactive code explanation (empty line to exit)")
            print("Conversation history enabled — follow-up questions work.")
            print("Diagram highlighting:")
            print("  🟡 exact match   🟠 fuzzy match   🔵 LLM guess (fallback)\n")

            conversation_history = []

            while True:
                q = input("> ").strip()
                if not q:
                    break

                context = select_context(
                    q,
                    code_index,
                    max_chars=6000,
                    model_name="llama3.2",
                    shortlist_size=12,
                    max_files=6,
                    use_llm=True,
                    conversation_history=conversation_history[-4:],
                )
                print("\n====== CONTEXT SENT TO LLM ======\n")
                print(context)
                print("\n====== END CONTEXT ======\n")
                answer = explain_code(
                    q, context,
                    conversation_history=conversation_history[-4:],
                )

                conversation_history.append({"role": "user", "content": q})
                conversation_history.append({"role": "assistant", "content": answer})

                print("\nAnswer:\n")
                print(answer)

                try:
                    dep_puml = "top_dependencies.puml"
                    if os.path.isfile(dep_puml):
                        dep_text = open(dep_puml, "r", encoding="utf-8").read()
                        label_to_id = extract_dependency_components(dep_text)
                        labels = list(label_to_id.keys())

                        exact = match_exact_dependency_labels(q, labels)
                        fuzzy = set()
                        if not exact:
                            fuzzy = match_fuzzy_dependency(q, labels)

                        label_to_color = {lab: COLOR_EXACT for lab in exact}
                        for lab in fuzzy:
                            if lab not in label_to_color:
                                label_to_color[lab] = COLOR_FUZZY

                        if not label_to_color and labels:
                            picked = llm_pick_best_node(q, labels, context)
                            if picked:
                                label_to_color[picked] = COLOR_LLM

                        if label_to_color:
                            highlighted = highlight_dependency_diagram(dep_text, label_to_color)
                            out_puml = "top_dependencies.highlight.puml"
                            with open(out_puml, "w", encoding="utf-8") as f:
                                f.write(highlighted)
                            png = render_plantuml(out_puml, "png")
                            svg = render_plantuml(out_puml, "svg")
                            print(f"\n🖼 Highlighted dependency diagram: {png} / {svg}")

                    class_puml = "classes.puml"
                    if os.path.isfile(class_puml):
                        class_text = open(class_puml, "r", encoding="utf-8").read()
                        classes, methods_by_class = extract_classes_and_methods(class_text)

                        exact = match_exact_classes(q, classes, methods_by_class)
                        fuzzy = set()
                        if not exact:
                            fuzzy = match_fuzzy_classes(q, classes, methods_by_class)

                        class_to_color = {c: COLOR_EXACT for c in exact}
                        for c in fuzzy:
                            if c not in class_to_color:
                                class_to_color[c] = COLOR_FUZZY

                        if not class_to_color and classes:
                            picked = llm_pick_best_node(q, sorted(list(classes)), context)
                            if picked:
                                class_to_color[picked] = COLOR_LLM

                        if class_to_color:
                            highlighted = highlight_class_diagram(class_text, class_to_color)
                            out_puml = "classes.highlight.puml"
                            with open(out_puml, "w", encoding="utf-8") as f:
                                f.write(highlighted)
                            png = render_plantuml(out_puml, "png")
                            svg = render_plantuml(out_puml, "svg")
                            print(f"\n🖼 Highlighted class diagram: {png} / {svg}")

                except Exception as e:
                    print(f"\n⚠️ Diagram highlight skipped due to error: {e}")

                print("\n" + "-" * 60)

        elif main_mode == "7":
            from top_dependency_llm import generate_llm_dependency_graph

            generate_llm_dependency_graph(
                files=files,
                base_dir=reader.clone_dir,
                output_file="top_dependencies.puml",
                code_index=code_index,
            )

        elif main_mode == "8":
            output_patterns = input(
                "Output pattern analysis file (default: patterns.md): "
            ).strip() or "patterns.md"
            result_text, arch_png = recognize_patterns(
                files=files,
                base_dir=reader.clone_dir,
                output_file=output_patterns,
                model_name="llama3.2",
                code_index=code_index,
            )
            print("\n" + result_text)
            if arch_png:
                print(f"\n🖼 Architecture diagram: {arch_png}")

        else:
            mode_input = input("Documentation mode: [1] All .py files, [2] Single .py file: ").strip() or "1"
            review_input = (input("Enable review agent? [Y/n]: ").strip() or "y").lower()
            enable_review = review_input.startswith("y")

            rounds_raw = input("Max review rounds (default 2): ").strip()
            max_rounds = int(rounds_raw) if rounds_raw.isdigit() else 2

            output_file_input = input(
                "Output file name (default: documentation.txt): "
            ).strip() or "documentation.txt"

            generate_docs_for_repo(
                reader=reader,
                files=files,
                output_file=output_file_input,
                model_name="llama3.2",
                mode=mode_input,
                enable_review=enable_review,
                max_review_rounds=max_rounds,
            )

    finally:
        reader.delete_repository()

if __name__ == "__main__":
    main()
