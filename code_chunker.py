import ast

def chunk_python_file(file_path, code):

    chunks = []

    try:
        tree = ast.parse(code)
    except Exception:
        return chunks

    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):

            chunk = {
                "path": file_path,
                "type": "function",
                "name": node.name,
                "code": ast.get_source_segment(code, node)
            }

            chunks.append(chunk)

        if isinstance(node, ast.ClassDef):

            chunk = {
                "path": file_path,
                "type": "class",
                "name": node.name,
                "code": ast.get_source_segment(code, node)
            }

            chunks.append(chunk)

    return chunks