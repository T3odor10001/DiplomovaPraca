import subprocess
import os
from pathlib import Path

def render_plantuml(
    puml_file: str,
    output_format: str = "png",
    plantuml_jar: str = "tools/plantuml.jar",
):
    """
    Render PlantUML (.puml) file into image (PNG/SVG).

    Creates output next to the .puml file.
    """
    puml_path = Path(puml_file).resolve()
    jar_path = Path(plantuml_jar).resolve()

    if not puml_path.is_file():
        raise FileNotFoundError(f"PlantUML file not found: {puml_path}")

    if not jar_path.is_file():
        raise FileNotFoundError(f"plantuml.jar not found: {jar_path}")

    cmd = [
        "java",
        "-jar",
        str(jar_path),
        f"-t{output_format}",
        str(puml_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        err = result.stderr or result.stdout or "unknown error"
        print(f"PlantUML rendering failed (exit {result.returncode}):\n{err}")
        raise RuntimeError(f"PlantUML rendering failed: {err}")

    output_file = puml_path.with_suffix(f".{output_format}")

    if not output_file.is_file():
        print(f"Warning: PlantUML did not produce output file: {output_file}")
        raise FileNotFoundError(f"PlantUML output not created: {output_file}")

    return output_file
