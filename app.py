import os
import streamlit as st
from PIL import Image

from main import RepositoryReader, generate_docs_for_repo

from importance_analyzer import get_top_important_functions
from classdiagram_generator import generate_classdiagram_for_repo
from full_classdiagram_generator import generate_full_classdiagram
from install_guide_generator import generate_installation_guide
from plantuml_renderer import render_plantuml

from diagram_highlighter import (
    extract_dependency_components, highlight_dependency_diagram,
    extract_classes_and_methods, highlight_class_diagram,
    match_exact_dependency_labels, match_fuzzy_dependency,
    match_exact_classes, match_fuzzy_classes,
    llm_pick_best_node,
    COLOR_EXACT, COLOR_FUZZY, COLOR_LLM,
)
from code_indexer import build_code_index, build_enriched_code_index
from context_selector import select_context
from code_explainer import explain_code

from top_dependency_llm import generate_llm_dependency_graph
from pattern_recognizer import recognize_patterns

def _highlight_diagrams(question, context):
    """
    Check for existing .puml diagrams, highlight matching nodes,
    render to PNG. Returns list of (label, png_path) tuples.
    """
    results = []

    try:
        dep_puml = "top_dependencies.puml"
        if os.path.isfile(dep_puml):
            dep_text = open(dep_puml, "r", encoding="utf-8").read()
            label_to_id = extract_dependency_components(dep_text)
            labels = list(label_to_id.keys())

            if labels:
                exact = match_exact_dependency_labels(question, labels)
                fuzzy = set()
                if not exact:
                    fuzzy = match_fuzzy_dependency(question, labels)

                label_to_color = {lab: COLOR_EXACT for lab in exact}
                for lab in fuzzy:
                    if lab not in label_to_color:
                        label_to_color[lab] = COLOR_FUZZY

                if not label_to_color:
                    picked = llm_pick_best_node(question, labels, context)
                    if picked:
                        label_to_color[picked] = COLOR_LLM

                if label_to_color:
                    highlighted = highlight_dependency_diagram(dep_text, label_to_color)
                    out_puml = "top_dependencies.highlight.puml"
                    with open(out_puml, "w", encoding="utf-8") as f:
                        f.write(highlighted)
                    png = render_plantuml(out_puml, "png")
                    results.append(("Dependency Graph", str(png)))

        class_puml = "classes.puml"
        if os.path.isfile(class_puml):
            class_text = open(class_puml, "r", encoding="utf-8").read()
            classes, methods_by_class = extract_classes_and_methods(class_text)

            if classes:
                exact = match_exact_classes(question, classes, methods_by_class)
                fuzzy = set()
                if not exact:
                    fuzzy = match_fuzzy_classes(question, classes, methods_by_class)

                class_to_color = {c: COLOR_EXACT for c in exact}
                for c in fuzzy:
                    if c not in class_to_color:
                        class_to_color[c] = COLOR_FUZZY

                if not class_to_color:
                    picked = llm_pick_best_node(question, sorted(list(classes)), context)
                    if picked:
                        class_to_color[picked] = COLOR_LLM

                if class_to_color:
                    highlighted = highlight_class_diagram(class_text, class_to_color)
                    out_puml = "classes.highlight.puml"
                    with open(out_puml, "w", encoding="utf-8") as f:
                        f.write(highlighted)
                    png = render_plantuml(out_puml, "png")
                    results.append(("Class Diagram", str(png)))

    except Exception as e:
        results.append(("__error__", str(e)))

    return results

def _render_if_needed(puml_path, fmt="png"):
    """Render a .puml file to the given format if the output doesn't exist yet."""
    from pathlib import Path
    out = Path(puml_path).with_suffix(f".{fmt}")
    if not out.is_file():
        render_plantuml(puml_path, fmt)
    return str(out)

st.set_page_config(page_title="AI Code Doc Assistant", layout="wide")
st.title("AI Documentation & Code Explorer")

repo_url = st.sidebar.text_input("GitHub Repository URL")

action = st.sidebar.selectbox(
    "Choose action",
    [
        "Generate Documentation",
        "Top 10 Important Functions",
        "Generate Top10 Class Diagram",
        "Generate Full Class Diagram",
        "Generate Installation Guide",
        "Interactive Code Explanation",
        "Generate Dependency Graph",
        "Recognize Architecture & Design Patterns",
    ],
)

st.session_state.setdefault("loaded", False)
st.session_state.setdefault("repo_url", "")
st.session_state.setdefault("action", "")
st.session_state.setdefault("reader", None)
st.session_state.setdefault("files", None)
st.session_state.setdefault("code_index", None)
st.session_state.setdefault("chat", [])

st.session_state.setdefault("diagrams", {})

colA, colB = st.sidebar.columns(2)
run_button = colA.button("Run")
reset_button = colB.button("Reset")

if reset_button:
    try:
        if st.session_state.reader is not None:
            st.session_state.reader.delete_repository()
    except Exception:
        pass

    st.session_state.loaded = False
    st.session_state.repo_url = ""
    st.session_state.action = ""
    st.session_state.reader = None
    st.session_state.files = None
    st.session_state.code_index = None
    st.session_state.chat = []
    st.session_state.diagrams = {}
    st.rerun()

if run_button:
    if not repo_url:
        st.sidebar.error("Please enter a GitHub repository URL.")
    else:
        try:
            if st.session_state.reader is not None:
                st.session_state.reader.delete_repository()
        except Exception:
            pass

        reader = RepositoryReader(repo_url=repo_url)
        with st.spinner("Cloning repository..."):
            reader.clone_repository()
            files = reader.read_files()

        if not files:
            st.error("No Python files found in the repository.")
        else:
            st.session_state.loaded = True
            st.session_state.repo_url = repo_url
            st.session_state.action = action
            st.session_state.reader = reader
            st.session_state.files = files
            st.session_state.chat = []
            st.session_state.diagrams = {}

            with st.spinner("Building LLM-enriched code index (one-time)..."):
                st.session_state.code_index = build_enriched_code_index(files, reader.clone_dir)

            st.success("Repository loaded and indexed")

if not st.session_state.loaded:
    st.info("Enter repo URL, choose action, click **Run**.")
    st.stop()

reader = st.session_state.reader
files = st.session_state.files

if action == "Generate Documentation":
    st.subheader("Generate Documentation")

    doc_scope = st.radio("Documentation scope", ["All .py files", "Single file"])

    selected_rel = None
    if doc_scope == "Single file":
        rel_paths = sorted(os.path.relpath(p, reader.clone_dir) for p in files.keys())
        selected_rel = st.selectbox("Select file", rel_paths)

    enable_review = st.checkbox("Enable review agent", value=True)
    max_rounds = 2
    if enable_review:
        max_rounds = st.number_input("Max review rounds", min_value=1, max_value=10, value=2)

    output_file = st.text_input("Output filename", value="documentation.txt")

    if st.button("Generate documentation now"):
        if doc_scope == "Single file" and selected_rel:
            selected_abs = os.path.abspath(os.path.join(reader.clone_dir, selected_rel))
            gen_files = {selected_abs: files[selected_abs]}
        else:
            gen_files = files

        with st.spinner(f"Generating documentation for {len(gen_files)} file(s)..."):
            generate_docs_for_repo(
                reader=reader,
                files=gen_files,
                output_file=output_file,
                model_name="llama3.2",
                mode="1",
                enable_review=enable_review,
                max_review_rounds=max_rounds,
            )
        content = open(output_file, "r", encoding="utf-8").read()
        st.download_button("Download documentation", content, file_name=output_file)
        st.text_area("Preview", content, height=500)

elif action == "Top 10 Important Functions":
    st.subheader("Top 10 Most Important Functions/Methods")

    if st.button("Analyze Top 10 Functions"):
        with st.spinner("Analyzing function importance..."):
            top_funcs = get_top_important_functions(
                files=files, base_dir=reader.clone_dir, top_n=10
            )
        if not top_funcs:
            st.warning("No functions/methods found in the repository.")
        else:
            table_data = []
            for idx, rec in enumerate(top_funcs, start=1):
                table_data.append({
                    "Rank": idx,
                    "Function": rec.qualname,
                    "File": rec.rel_path,
                    "Line": rec.lineno,
                    "Importance": rec.importance,
                })
            st.table(table_data)

elif action == "Generate Top10 Class Diagram":

    st.subheader("Top 10 Class Diagram")

    output_puml = "classes.puml"

    if "class_diagram" not in st.session_state.diagrams:
        if not os.path.exists(output_puml):
            with st.spinner("Generating class diagram (LLM-enriched)..."):
                generate_classdiagram_for_repo(
                    files=files,
                    base_dir=reader.clone_dir,
                    output_file=output_puml,
                    model_name="llama3.2",
                    code_index=st.session_state.code_index,
                )
        png = render_plantuml(output_puml, "png")
        st.session_state.diagrams["class_diagram"] = str(png)

    st.image(st.session_state.diagrams["class_diagram"], use_container_width=True)

    st.markdown("---")
    st.header("Explore classes")

    class_text = open(output_puml, "r", encoding="utf-8").read()
    classes, methods_by_class = extract_classes_and_methods(class_text)

    if not classes:
        st.warning("No classes detected in diagram.")
        st.stop()

    selected_class = st.selectbox("Select class from diagram", sorted(list(classes)))

    if selected_class:
        st.subheader(f"Class: {selected_class}")
        methods = methods_by_class.get(selected_class, [])

        if methods:
            st.markdown("### Methods")
            for m in methods:
                st.write(f"- {m}")
        else:
            st.info("No methods detected.")

        if st.button("Explain this class"):
            with st.spinner("Analyzing class..."):
                question = f"Explain the class {selected_class} and its purpose in the project."
                context = select_context(question, st.session_state.code_index)
                answer = explain_code(question, context)
            st.markdown("### Explanation")
            st.write(answer)

elif action == "Generate Full Class Diagram":

    output_puml = "full_system.puml"

    if "full_class_diagram" not in st.session_state.diagrams:
        if st.button("Generate full class diagram"):
            with st.spinner("Generating full diagram (LLM-enriched)..."):
                generate_full_classdiagram(
                    files, reader.clone_dir,
                    output_file=output_puml,
                    code_index=st.session_state.code_index,
                )
                png = render_plantuml(output_puml, "png")
            st.session_state.diagrams["full_class_diagram"] = str(png)
            st.rerun()
    else:
        st.image(st.session_state.diagrams["full_class_diagram"], use_container_width=True)

elif action == "Generate Installation Guide":
    if st.button("Generate installation guide"):
        output_install = "INSTALLATION.md"
        with st.spinner("Generating guide..."):
            generate_installation_guide(reader.clone_dir, output_install, "llama3.2")
        content = open(output_install, "r", encoding="utf-8").read()
        st.download_button("Download INSTALLATION.md", content, file_name="INSTALLATION.md")
        st.text_area("Preview", content, height=500)

elif action == "Generate Dependency Graph":

    output_file = "top_dependencies.puml"

    if "dependency_graph" not in st.session_state.diagrams:
        if st.button("Generate dependency graph"):
            with st.spinner("Generating dependency graph (LLM-enriched)..."):
                generate_llm_dependency_graph(
                    files, reader.clone_dir, output_file,
                    code_index=st.session_state.code_index,
                    render_png=False, render_svg=False,
                )
            try:
                png = render_plantuml(output_file, "png")
                render_plantuml(output_file, "svg")
                st.session_state.diagrams["dependency_graph"] = str(png)
                st.rerun()
            except Exception as e:
                st.error(f"Diagram rendering failed: {e}")
                if os.path.exists(output_file):
                    st.code(open(output_file, "r", encoding="utf-8").read(), language="text")
    else:
        st.image(st.session_state.diagrams["dependency_graph"], use_container_width=True)

elif action == "Recognize Architecture & Design Patterns":
    st.subheader("Architecture & Design Pattern Recognition")

    if "patterns_result" not in st.session_state.diagrams:
        if st.button("Analyze patterns"):
            output_file = "patterns.md"
            with st.spinner("Analyzing architecture and design patterns (LLM-enriched)..."):
                result_text, arch_png = recognize_patterns(
                    files=files,
                    base_dir=reader.clone_dir,
                    output_file=output_file,
                    model_name="llama3.2",
                    code_index=st.session_state.code_index,
                )
            st.session_state.diagrams["patterns_result"] = result_text
            if arch_png and os.path.isfile(arch_png):
                st.session_state.diagrams["architecture_diagram"] = arch_png
            st.rerun()
    else:
        result_text = st.session_state.diagrams["patterns_result"]
        st.markdown(result_text)
        st.download_button("Download patterns.md", result_text, file_name="patterns.md")

        if "architecture_diagram" in st.session_state.diagrams:
            arch_png = st.session_state.diagrams["architecture_diagram"]
            if os.path.isfile(arch_png):
                st.markdown("---")
                st.subheader("Architecture Diagram (highest confidence pattern)")
                st.image(arch_png, use_container_width=True)

elif action == "Interactive Code Explanation":

    st.subheader("Chat with the code")

    has_diagrams = False
    if st.session_state.diagrams:
        with st.expander("Generated diagrams (click to expand)", expanded=False):
            for key, png_path in st.session_state.diagrams.items():
                if os.path.isfile(png_path):
                    label = key.replace("_", " ").title()
                    st.markdown(f"**{label}**")
                    st.image(png_path, use_container_width=True)
                    has_diagrams = True
        if has_diagrams:
            st.caption("Ask a question and relevant diagram nodes will be highlighted below the answer.")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("images"):
                for label, img_path in msg["images"]:
                    if os.path.isfile(img_path):
                        st.image(img_path, caption=label, use_container_width=True)
                st.caption("Highlight colors:  yellow = exact match  |  orange = fuzzy match  |  blue = LLM guess")

    prompt = st.chat_input("Ask a question about the code...")

    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        recent_history = st.session_state.chat[-4:]

        context = select_context(
            prompt, st.session_state.code_index,
            max_chars=6000,
            model_name="llama3.2",
            shortlist_size=12,
            max_files=6,
            use_llm=True,
            conversation_history=recent_history,
        )

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                answer = explain_code(
                    prompt, context,
                    conversation_history=recent_history,
                )

            st.markdown(answer)

            if context:
                with st.expander("Source files used as context"):
                    blocks = context.split("# FILE:")
                    for block in blocks:
                        block = block.strip()
                        if not block:
                            continue
                        lines = block.split("\n", 1)
                        file_path = lines[0].strip()
                        code = lines[1] if len(lines) > 1 else ""
                        snippet = "\n".join(code.splitlines()[:40])
                        st.markdown(f"**{file_path}**")
                        st.code(snippet, language="python")

            highlighted_images = []
            if os.path.isfile("top_dependencies.puml") or os.path.isfile("classes.puml"):
                with st.spinner("Highlighting diagrams..."):
                    highlight_results = _highlight_diagrams(prompt, context)

                for label, path in highlight_results:
                    if label == "__error__":
                        st.warning(f"Diagram highlighting error: {path}")
                    elif os.path.isfile(path):
                        st.image(path, caption=label, use_container_width=True)
                        highlighted_images.append((label, path))

                if highlighted_images:
                    st.caption("Highlight colors:  yellow = exact match  |  orange = fuzzy match  |  blue = LLM guess")

        st.session_state.chat.append({
            "role": "assistant",
            "content": answer,
            "images": highlighted_images,
        })
