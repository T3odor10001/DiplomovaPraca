# Installation Guide

## 1. Introduction

This project is a Python-based weather data management system designed to process and analyze weather log data from various sources. The primary purpose of this project is to provide a centralized platform for managing, processing, and visualizing weather-related data.

The architecture of the project consists of several modules, including `background_manager.py`, which handles data ingestion and processing, and `main.py`, which serves as the entry point for running the application. This guide will walk you through installing and configuring the project on your local machine.

## 2. System Requirements

- Minimum Python version: Python 3.10+
- Required tools:
    - Git (version 2.x or higher)
    - pip
    - virtualenv (optional, but recommended for development and production environments)
    - Ollama (for LLM model management; installation instructions below)
    - Docker (if relevant to your use case)
- Supported operating systems: Windows, Linux, macOS
- CPU/RAM recommendations:
    - For general usage: 2-4 GB RAM
    - For heavy usage or LLM integration: 8-16 GB RAM
- Download links:
    - Python: <https://www.python.org/downloads/>
    - Git: <https://git-scm.com/downloads>
    - virtualenv: <https://virtualenv.pypa.io/en/stable/>
    - Ollama: <https://ollama.dev/>

## 3. Cloning the Repository

To clone the repository, follow these steps:

### Windows (PowerShell)

```bash
git clone https://github.com/your-repo-url.git
cd your-repo-folder
```

### Linux / macOS

```bash
git clone https://github.com/your-repo-url.git
cd your-repo-folder
```

## 4. Creating a Virtual Environment

A virtual environment is a self-contained Python environment that allows you to manage dependencies and isolation between projects.

Create a new virtual environment using the following commands:

### Windows (PowerShell)

```bash
py -3 -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 5. Installing Dependencies

To install the project's dependencies, follow these steps:

- If `requirements.txt` exists:
    ```bash
pip install -r requirements.txt
```
- If `pyproject.toml` exists:
    ```bash
pip install .
```

The dependencies listed in `requirements.txt` are necessary for running the project.

## 6. Setting Up Ollama / LLM Models (if used)

If your project depends on Ollama or LLM models, follow these steps to set them up:

- Install Ollama:
    ```bash
pip install ollama
```
- Verify daemon status:
    ```
ollama list
ollama run llama3.2
```
- Download an LLM model (e.g., `llama3.2`):
    ```
ollama pull llama3.2
```

## 7. Running the Project

To run the project, follow these steps:

### Running main.py

```bash
python main.py
```

This will start the application and begin processing weather log data.

### Running CLI entrypoints

- List available entrypoints:
    ```bash
python -c "import ollama; print(ollama.__version__)"
```
- Run a specific entrypoint (e.g., `weather_analysis`):
    ```
python main.py weather_analysis
```

## 8. Project Configuration

If you need to modify configuration settings, follow these steps:

- Locate the configuration file (`config.py` or `.env`).
- Understand what each configuration value means:
    - `API_KEY`: Your API key for accessing weather data.
    - `LOG_FILE`: The path where output and log files will be stored.
- Modify configuration values as needed:
    ```python
# config.py

class Config:
    API_KEY = "YOUR_API_KEY"
    LOG_FILE = "/path/to/log/file.log"

config = Config()
```
    ```
# .env

API_KEY=YOUR_API_KEY
LOG_FILE=/path/to/log/file.log
```

## 9. Project Structure

The project consists of the following major directories and modules:

*   `background_manager.py`: Handles data ingestion and processing.
*   `main.py`: Serves as the entry point for running the application.
*   `sc\`: Contains static images (e.g., `sc_1.png`, `sc_2.png`).
*   `data\weather_log.xlsx`: A sample weather log file.

## 10. Post-Installation Testing

To verify that everything is working correctly, follow these steps:

- Run a basic testing function:
    ```bash
python -c "import background_manager; background_manager.test()"
```
    If this passes, your installation should be successful.

## 11. Common Issues (Troubleshooting)

Here are some common issues you might encounter and their fixes:

1.  `pip cannot find module`:

    *   Make sure the Python environment is correctly activated.
    *   Check that the package is not installed in the current virtual environment.
2.  `virtualenv cannot activate`:

    *   Ensure that the virtual environment is created correctly.
    *   Verify that the activate command works without arguments.
3.  `Permission denied / Access denied`:

    *   Ensure you have the necessary permissions to access the project directory.
4.  Windows execution policy blocking scripts:

    *   Open Command Prompt or PowerShell as administrator.
5.  Git clone errors:

    *   Check that the repository URL is correct.
6.  Missing dependencies:

    *   Run `pip install -r requirements.txt` to reinstall dependencies.
7.  UnicodeDecodeError:

    *   Ensure your system locale and encoding are correctly set up.
8.  SSL/TLS errors:

    *   Verify that your HTTPS certificates are correctly installed on the server (if applicable).
9.  Ollama model not downloaded:

    *   Check if the `pull` command was executed successfully.
10. LangChain invalid prompt input:

    *   Review and adjust the prompt format according to LangChain documentation.
11. Missing library errors:

    *   Check that you have installed all required packages.

## 12. Additional Recommendations

-   **Using VS Code / PyCharm**: Take advantage of these IDEs for improved coding experience, code completion, debugging, and performance optimization tools.
-   **Maintaining virtual environments**: Use tools like `venv` or `conda` to create isolated Python environments for each project and team member.
-   **Performance tips**:
    -   Optimize database queries for faster execution times.
    -   Leverage caching mechanisms where possible (e.g., Redis).
    -   Implement efficient data structures and algorithms.
-   **Suggested IDE extensions**: Explore various plugins for enhancing your coding workflow, such as syntax highlighting, code completion, and debugging tools.

## 13. Conclusion

Congratulations on successfully installing the project! You can now proceed to explore the features of this weather management system further. Consider testing more advanced functionalities or modifying existing modules to suit your needs.