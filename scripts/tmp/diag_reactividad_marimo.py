"""Diagnóstico de reactividad v2 para notebooks marimo convertidos.

Solo reporta conflictos REALES:
- Variables globales (sin prefijo _) asignadas en mas de una celda -> error MB002.
- Variables globales usadas que ninguna celda define ni recibe como parametro.
"""
import ast
import sys
from pathlib import Path


def extract_cells(source: str):
    tree = ast.parse(source)
    cells = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_cell = any(
            (isinstance(d, ast.Attribute) and d.attr == "cell")
            or (isinstance(d, ast.Name) and d.id == "cell")
            for d in node.decorator_list
        )
        if is_cell:
            params = {a.arg for a in node.args.args}
            stmts = [ast.unparse(s) for s in node.body if not isinstance(s, ast.Return)]
            code = "\n".join(stmts)
            cells.append((node.lineno, params, code))
    return cells


def main(path: Path):
    source = path.read_text(encoding="utf-8")
    cells = extract_cells(source)
    print(f"Celdas encontradas: {len(cells)}\n")

    global_defs: dict[str, list[int]] = {}
    global_uses: dict[str, list[int]] = {}
    all_params: set[str] = set()

    for i, (_, params, code) in enumerate(cells, 1):
        all_params |= params
        tree = ast.parse(code)
        assigned, loaded = set(), set()

        def walk_scope(node):
            """Recorre solo el scope de la celda: no entra en
            funciones, lambdas ni comprensiones (scopes propios)."""
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child,
                    ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
                ):
                    continue
                if isinstance(
                    child,
                    ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
                ):
                    continue
                if isinstance(child, ast.Import):
                    for a in child.names:
                        assigned.add(a.asname or a.name.split(".")[0])
                elif isinstance(child, ast.ImportFrom):
                    for a in child.names:
                        assigned.add(a.asname or a.name)
                elif isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Store):
                        assigned.add(child.id)
                    else:
                        loaded.add(child.id)
                elif isinstance(child, ast.FunctionDef | ast.ClassDef):
                    assigned.add(child.name)
                walk_scope(child)

        walk_scope(tree)
        for n in assigned:
            if not n.startswith("_"):
                global_defs.setdefault(n, []).append(i)
        for n in loaded:
            if not n.startswith("_"):
                global_uses.setdefault(n, []).append(i)

    print("=" * 60)
    print("1) GLOBALES REDEFINIDAS ENTRE CELDAS (error MB002 real)")
    print("=" * 60)
    redefined = {k: v for k, v in global_defs.items() if len(v) > 1}
    if not redefined:
        print("   Ninguna ✓")
    line_of = {i: ln for i, (ln, _, _) in enumerate(cells, 1)}
    for name, nums in sorted(redefined.items()):
        locs = ", ".join(f"celda {n} (linea {line_of[n]})" for n in sorted(set(nums)))
        print(f"   {name}: {locs}")

    print()
    print("=" * 60)
    print("2) GLOBALES USADAS SIN DEFINICION NI PARAMETRO")
    print("=" * 60)
    import builtins as b
    bi = set(dir(b))
    missing = {
        n for n in global_uses
        if n not in global_defs and n not in all_params and n not in bi
    }
    if not missing:
        print("   Ninguna ✓")
    for name in sorted(missing):
        print(f"   {name}: usado en celdas {sorted(set(global_uses[name]))[:8]}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "TP1_corregido/tp1_exploracion_marimo.py"
    )
    main(target)
