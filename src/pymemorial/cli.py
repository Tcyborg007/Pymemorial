# src/pymemorial/cli.py
import json
from pathlib import Path
import click
from rich.console import Console

console = Console()

@click.group()
def main():
    """CLI do PyMemorial 2.0 — gere memoriais de cálculo com alto nível profissional."""
    pass

@main.command("init")
@click.option("--title", default="Memorial de Cálculo", help="Título do memorial.")
@click.option("--author", default="Autor", help="Autor do memorial.")
@click.option("--output", default="memorial.json", help="Arquivo JSON de saída.")
def init(title: str, author: str, output: str):
    """Inicializa um memorial mínimo em JSON."""
    data = {
        "metadata": {"title": title, "author": author},
        "variables": {},
        "sections": [{"title": "Introdução", "level": 1, "content": ["Documento inicializado."]}],
    }
    Path(output).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green]OK[/bold green] Memorial criado: {output}")

@main.command("version")
def version():
    """Exibe a versão instalada."""
    from pymemorial import __version__
    # Use print() para stdout capturável em testes
    print(f"PyMemorial {__version__}")

# Permite execução via python -m pymemorial.cli
if __name__ == "__main__":
    main()
