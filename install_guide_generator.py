import os
from typing import Dict, List

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

INSTALL_TEMPLATE = """
You are an expert technical writer responsible for producing a highly detailed, step-by-step INSTALLATION GUIDE for a Python project. The output MUST be in English and formatted strictly in Markdown.

Your installation guide must be extremely detailed, covering:
- installation,
- configuration,
- running the application,
- command examples,
- troubleshooting,
- optional setup,
- OS-specific notes,
- best practices.

============================================================
# REQUIRED OUTPUT STRUCTURE
============================================================

Your output MUST contain the following sections in this exact order:

# 1. Introduction
- Explain what the project does (infer from file structure and modules)
- Explain the purpose of the project
- Who the project is intended for
- Why a user might want to install it
- Short explanation of the architecture based on module names

# 2. System Requirements
MUST INCLUDE:
- Minimum Python version (if unknown, recommend Python 3.10+)
- Required tools (GIT, pip, virtualenv, Ollama, Docker — if relevant)
- Supported operating systems (Windows, Linux, macOS)
- CPU/RAM recommendations if the project interacts with LLMs
- Download links (Python, Git, Ollama)

# 3. Cloning the Repository
Must contain instructions for:
- Windows (PowerShell)
- Linux (bash)
- macOS (zsh)

And MUST include the commands:
git clone <repo_url>
cd <repo_folder>

# 4. Creating a Virtual Environment
Detailed OS-specific instructions:

## Windows (PowerShell)
py -3 -m venv venv
venv\\\\Scripts\\\\activate

## Linux / macOS
python3 -m venv venv
source venv/bin/activate

Also include a brief explanation of what a virtual environment is and why it is recommended.

# 5. Installing Dependencies
Explain both cases:
- If requirements.txt exists → use: pip install -r requirements.txt
- If pyproject.toml exists → use: pip install .

Explain what dependencies are and why they are needed.

# 6. Setting Up Ollama / LLM Models (if used)
If input files hint that the project depends on LLMs or Ollama:

Must include:
- How to install Ollama
- How to verify daemon status:
  ollama list
  ollama run llama3.2
- How to download LLM model:
  ollama pull llama3.2

# 7. Running the Project
MUST include:
- How to run main.py
- How to run CLI entrypoints
- Examples of basic commands and usage
- Expected output

# 8. Project Configuration
If any configuration files exist (e.g., config.py, .env), explain:
- What each configuration value means
- How to modify configuration
- Where output/log files are stored

# 9. Project Structure
Based on file_tree, generate a section explaining:
- the purpose of each major directory,
- the purpose of key modules,
- how parts of the project interact.

# 10. Post-Installation Testing
Must include:
python -c "import <module>"
Testing functions
Testing LLM integration (if relevant)

# 11. Common Issues (Troubleshooting)
Must include AT LEAST 10 real issues and fixes:
- pip cannot find module
- virtualenv cannot activate
- Permission denied / Access denied
- Windows execution policy blocking scripts
- Git clone errors
- Missing dependencies
- UnicodeDecodeError
- SSL/TLS errors
- Ollama model not downloaded
- LangChain invalid prompt input
- Missing library errors

# 12. Additional Recommendations
Provide recommended workflows:
- Using VS Code / PyCharm
- How to maintain virtual environments
- Performance tips
- Suggested IDE extensions
- Recommended practices for LLM-heavy projects

Also mention that we need to install libraries from requirements.txt

# 13. Conclusion
Short summary + recommendations for future steps.

============================================================
# INPUT DATA
============================================================

## File Structure:
{file_tree}

## pyproject.toml:
{pyproject}

## requirements.txt:
{requirements}

## Entry Point Scripts:
{entrypoints}
"""

def _build_file_tree(base_dir: str, max_files: int = 200) -> str:
    """
    Vytvorí jednoduchý zoznam relatívnych ciest všetkých súborov,
    obmedzený počtom prehľadu.
    """
    paths: List[str] = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".venv", "venv")]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), base_dir)
            paths.append(rel)
            if len(paths) >= max_files:
                break
        if len(paths) >= max_files:
            break

    if not paths:
        return "Žiadne súbory neboli nájdené."

    return "\n".join(f"- {p}" for p in sorted(paths))

def _read_if_exists(path: str, max_chars: int = 5000) -> str:
    if not os.path.isfile(path):
        return "Súbor neexistuje."
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return "Súbor sa nepodarilo prečítať."

    if len(content) > max_chars:
        return content[:max_chars] + "\n...\n(orezané kvôli dĺžke)"
    return content

def _detect_entrypoints(base_dir: str) -> str:
    candidates = ["__main__.py", "cli.py", "main.py", "manage.py"]
    found: List[str] = []

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f in candidates:
                rel = os.path.relpath(os.path.join(root, f), base_dir)
                found.append(rel)

    if not found:
        return "Nenašli sa žiadne typické entrypoint skripty."
    return "\n".join(f"- {p}" for p in sorted(found))

def create_install_chain(model_name: str = "llama3.2"):
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(INSTALL_TEMPLATE)
    return prompt | model

def generate_installation_guide(
    base_dir: str,
    output_file: str = "INSTALLATION.md",
    model_name: str = "llama3.2",
) -> None:
    """
    Vygeneruje inštalačnú príručku pre daný repozitár (base_dir) a uloží ju do output_file.
    """
    file_tree = _build_file_tree(base_dir)
    pyproject = _read_if_exists(os.path.join(base_dir, "pyproject.toml"))
    requirements = _read_if_exists(os.path.join(base_dir, "requirements.txt"))
    entrypoints = _detect_entrypoints(base_dir)

    chain = create_install_chain(model_name=model_name)

    print("Generating installation guide via LLM agent...")
    guide = chain.invoke(
        {
            "file_tree": file_tree,
            "pyproject": pyproject,
            "requirements": requirements,
            "entrypoints": entrypoints,
        }
    )

    text = str(guide).strip()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Installation guide written to: {output_file}")
