#!/usr/bin/env python3
"""
Convertidor de Jupyter Notebooks (.ipynb) a PDF con gráficos y estilo profesional.
"""

import sys
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Colorantes ANSI para consola
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
MAGENTA = "\033[1;35m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header():
    print(f"\n{BLUE}=============================================================={RESET}")
    print(f"{CYAN}{BOLD}  📄 CONVERTIDOR DE JUPYTER NOTEBOOKS (.ipynb) A PDF{RESET}")
    print(f"{BLUE}=============================================================={RESET}\n")

def check_dependencies():
    """Verifica e instala dependencias requeridas si faltan."""
    missing = []
    try:
        import nbconvert
    except ImportError:
        missing.append("nbconvert")
    try:
        import bs4
    except ImportError:
        missing.append("beautifulsoup4")

    if missing:
        print(f"{YELLOW}Installing missing dependencies: {', '.join(missing)}...{RESET}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print(f"{GREEN}✔ Dependencies installed successfully.{RESET}\n")

def find_chrome_binary():
    """Busca el ejecutable de Chrome o Chromium en el sistema."""
    candidates = [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        path = shutil.which(candidate) or (candidate if os.path.exists(candidate) else None)
        if path:
            return path
    return None

def find_notebooks(root_dir="."):
    """Busca todos los notebooks en la estructura del proyecto."""
    root_path = Path(root_dir).resolve()
    notebooks = []
    ignored_dirs = {".ipynb_checkpoints", ".git", ".venv", "venv", "env", "node_modules"}
    
    for path in root_path.rglob("*.ipynb"):
        if not any(part in ignored_dirs for part in path.parts):
            notebooks.append(path)
            
    return sorted(notebooks)

def generate_custom_css(theme="blue"):
    """Genera estilos CSS de alta calidad para la impresión a PDF."""
    
    accent_color = "#2563eb" if theme == "blue" else "#059669" if theme == "green" else "#4f46e5"
    header_color = "#1d4ed8" if theme == "blue" else "#047857" if theme == "green" else "#4338ca"
    
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@page {{
    size: A4;
    margin: 1.6cm 1.3cm 1.6cm 1.3cm;
}}

body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #1e293b !important;
    background-color: #ffffff !important;
    line-height: 1.6 !important;
    font-size: 10pt !important;
}}

/* Encabezados */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Inter', sans-serif !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    page-break-after: avoid !important;
    break-after: avoid !important;
}}

h1 {{
    font-size: 21pt !important;
    border-bottom: 2.5px solid {accent_color} !important;
    padding-bottom: 8px !important;
    margin-top: 24px !important;
    margin-bottom: 16px !important;
    color: {header_color} !important;
}}

h2 {{
    font-size: 15pt !important;
    border-bottom: 1px solid #e2e8f0 !important;
    padding-bottom: 6px !important;
    margin-top: 20px !important;
    color: #0f172a !important;
}}

h3 {{
    font-size: 12.5pt !important;
    margin-top: 16px !important;
    color: #334155 !important;
}}

/* Celdas de código */
.jp-CodeCell, .cell.code, div.input {{
    page-break-inside: avoid !important;
    break-inside: avoid !important;
    margin-bottom: 12px !important;
}}

.jp-InputArea, div.input_area, div.code_cell pre {{
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 8.8pt !important;
    padding: 10px !important;
}}

/* Gráficos e Imágenes */
img, .output_png img, .jp-OutputArea-output img {{
    max-width: 100% !important;
    height: auto !important;
    display: block !important;
    margin: 14px auto !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid #e2e8f0 !important;
    page-break-inside: avoid !important;
    break-inside: avoid !important;
}}

/* Tablas */
table, .dataframe {{
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 16px 0 !important;
    font-size: 9.2pt !important;
    page-break-inside: avoid !important;
    break-inside: avoid !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #cbd5e1 !important;
}}

table th, .dataframe th {{
    background-color: #1e293b !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    text-align: left !important;
}}

table td, .dataframe td {{
    padding: 7px 12px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    color: #334155 !important;
}}

table tr:nth-child(even), .dataframe tr:nth-child(even) {{
    background-color: #f8fafc !important;
}}

/* Citas y Callouts */
blockquote {{
    border-left: 4px solid {accent_color} !important;
    background-color: #eff6ff !important;
    padding: 10px 16px !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 14px 0 !important;
    color: #1e40af !important;
}}

.jp-Notebook {{
    padding: 0 !important;
}}
</style>
"""

def convert_notebook(notebook_path, output_pdf_path, theme="blue"):
    """Convierte el notebook especificado a PDF utilizando HTML + Chrome Headless."""
    check_dependencies()
    import nbconvert
    from bs4 import BeautifulSoup

    chrome_bin = find_chrome_binary()
    if not chrome_bin:
        raise RuntimeError("No se encontró Google Chrome o Chromium en el sistema para renderizar el PDF.")

    notebook_path = Path(notebook_path).resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {notebook_path}")

    output_pdf_path = Path(output_pdf_path).resolve()
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{CYAN}[1/3] Cargando y convirtiendo cuaderno a HTML...{RESET}")
    exporter = nbconvert.HTMLExporter()
    exporter.theme = "light"
    html_content, _ = exporter.from_filename(str(notebook_path))

    print(f"{CYAN}[2/3] Aplicando tipografía, gráficos y estilos profesionales...{RESET}")
    soup = BeautifulSoup(html_content, "html.parser")
    custom_css = generate_custom_css(theme)
    head = soup.find("head")
    if head:
        head.append(BeautifulSoup(custom_css, "html.parser"))

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp_html:
        tmp_html_path = tmp_html.name
        tmp_html.write(str(soup))

    try:
        print(f"{CYAN}[3/3] Generando documento PDF mediante Chrome Headless...{RESET}")
        cmd = [
            chrome_bin,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=6000",
            f"--print-to-pdf={output_pdf_path}",
            tmp_html_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    finally:
        if os.path.exists(tmp_html_path):
            os.remove(tmp_html_path)

    if not output_pdf_path.exists():
        raise RuntimeError("La conversión falló y no se generó el archivo PDF.")

    file_size_kb = output_pdf_path.stat().st_size / 1024
    
    pages_str = ""
    try:
        import fitz
        doc = fitz.open(str(output_pdf_path))
        pages_str = f" ({len(doc)} páginas)"
    except Exception:
        pass

    print(f"\n{GREEN}{BOLD}✔ CONVERSIÓN EXITOSA!{RESET}")
    print(f"{BOLD}  📄 Archivo PDF:{RESET} {output_pdf_path}")
    print(f"{BOLD}  📊 Tamaño:{RESET} {file_size_kb:.1f} KB{pages_str}\n")

def main():
    print_header()

    # Si se pasa un argumento por línea de comandos
    if len(sys.argv) > 1 and sys.argv[1].strip():
        input_path = sys.argv[1].strip()
    else:
        # Modo interactivo
        repo_notebooks = find_notebooks(".")
        
        if repo_notebooks:
            print(f"{BOLD}Notebooks encontrados en este proyecto:{RESET}\n")
            for i, nb_path in enumerate(repo_notebooks, 1):
                rel_path = nb_path.relative_to(Path(".").resolve()) if nb_path.is_relative_to(Path(".").resolve()) else nb_path
                print(f"  {CYAN}[{i}]{RESET} {rel_path}")
            print()
            
            choice = input(f"{BOLD}Seleccioná un número [1-{len(repo_notebooks)}] o ingresá una ruta (.ipynb): {RESET}").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(repo_notebooks):
                input_path = str(repo_notebooks[int(choice) - 1])
            elif choice:
                input_path = choice
            else:
                input_path = str(repo_notebooks[0])
                print(f"{YELLOW}Selección por defecto: {input_path}{RESET}")
        else:
            input_path = input(f"{BOLD}Ingresá la ruta del archivo notebook (.ipynb): {RESET}").strip()

    if not input_path:
        print(f"{YELLOW}No se especificó ninguna ruta. Operación cancelada.{RESET}")
        sys.exit(1)

    notebook_path = Path(input_path).resolve()
    if not notebook_path.exists():
        print(f"\n❌ Error: El archivo '{input_path}' no existe.")
        sys.exit(1)

    # Sugerir ruta de salida en la misma carpeta del notebook
    default_output = notebook_path.with_suffix(".pdf")
    
    output_prompt = input(f"\n{BOLD}Ruta de salida del PDF [{default_output.name}]: {RESET}").strip()
    if output_prompt:
        output_pdf_path = Path(output_prompt).resolve()
    else:
        output_pdf_path = default_output

    # Opciones de tema estilizado
    print(f"\n{BOLD}Temas visuales disponibles:{RESET}")
    print(f"  {CYAN}[1]{RESET} Azul Ejecutivo (Recomendado)")
    print(f"  {CYAN}[2]{RESET} Verde Menta / Naturaleza")
    print(f"  {CYAN}[3]{RESET} Índigo Moderno")
    
    theme_choice = input(f"\n{BOLD}Elegí un tema [1-3] (Default: 1): {RESET}").strip()
    theme_map = {"1": "blue", "2": "green", "3": "indigo"}
    selected_theme = theme_map.get(theme_choice, "blue")

    print()
    try:
        convert_notebook(notebook_path, output_pdf_path, theme=selected_theme)
    except Exception as e:
        print(f"\n❌ Error durante la conversión: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
