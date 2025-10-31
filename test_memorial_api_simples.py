# Salve este arquivo como "run_memorial_test.py" na raiz do seu projeto
# (no mesmo nível da pasta "src" ou "pymemorial")

import logging
import os
import base64
from pathlib import Path

# --- 1. IMPORTAÇÕES DAS CLASSES DO PYMEMORIAL ---
# Estas importações funcionam por causa do seu __init__.py
try:
    from pymemorial.core.calculator import Calculator
    from pymemorial.document import (
        Memorial,
        DocumentMetadata,
        NormCode,
        Section,
        TextBlock,  # Importado de content_block via __init__.py
        Table,      # Importado de content_block via __init__.py
        Figure,     # Importado de content_block via __init__.py
        TableStyle
    )
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Certifique-se de que seu ambiente Python pode encontrar o pacote 'pymemorial'.")
    print("Tente executar 'pip install -e .' na raiz do projeto (C:\\...\\Pymemorial).")
    exit()

# Configuração de logging para ver o que está acontecendo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PyMemorialTest")

def create_placeholder_image(filepath: Path):
    """
    Função auxiliar RÁPIDA apenas para o teste.
    Cria um arquivo de imagem placeholder para o doc.add_figure() testar.
    Isto NÃO faz parte do pymemorial.
    """
    if not filepath.exists():
        logger.info(f"Criando imagem placeholder: {filepath.name}")
        # Um PNG 1x1 pixel transparente
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        try:
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(png_b64))
        except Exception as e:
            logger.error(f"Não foi possível criar a imagem placeholder: {e}")

# --- 2. CONFIGURAÇÃO DO CALCULATOR ---
#
logger.info("Configurando o Calculator...")
calc = Calculator()
calc.set_variable("f_ck", 25, unit="MPa", description="Resistência do concreto")
calc.set_variable("gamma_c", 1.4, unit="", description="Coef. Minoração Concreto")
calc.set_variable("gamma_f", 1.4, unit="", description="Coef. Majoração Cargas")
calc.set_variable("M_k", 120, unit="kN*m", description="Momento Característico")


# --- 3. CONFIGURAÇÃO DO MEMORIAL ---
logger.info("Iniciando o Memorial...")
metadata = DocumentMetadata(
    title="Teste Funcional do PyMemorial",
    author="Eng. Testador",
    company="PyMemorial Dev"
)

# Injeta o 'calc' no 'Memorial' para manter o contexto
#
doc = Memorial(
    metadata=metadata,
    calculator_instance=calc,
    norm=NormCode.NBR6118_2023
)

# --- 4. ADIÇÃO DE CONTEÚDO (Fluxo Inteligente) ---

# Adiciona Seção
#
doc.add_section("1.0 Dados de Entrada")

# Adiciona Texto Inteligente
# 'f_ck' e 'gamma_c' serão convertidos em $f_{ck}$ e $\gamma_{c}$
#
doc.add_paragraph(
    "Este memorial verifica uma viga de concreto. "
    "A resistência do concreto adotada é f_ck. "
    "O coeficiente de minoração da resistência é gamma_c."
)

# Adiciona Tabela
#
table_data = [
    ["Variável", "Valor", "Unidade", "Descrição"],
    ["f_ck", 25, "MPa", "Resistência do concreto"],
    ["gamma_c", 1.4, "-", "Coef. minoração (concreto)"],
    ["gamma_f", 1.4, "-", "Coef. majoração (cargas)"],
    ["M_k", 120, "kN*m", "Momento Característico"],
]
doc.add_table(
    data=table_data,
    caption="Valores de entrada para o cálculo.",
    style=TableStyle.STRIPED
)

# Adiciona Seção de Cálculo
doc.add_section("2.0 Cálculos Principais")

# Adiciona Cálculo Inteligente (1)
#
doc.add_calculation(
    "f_cd = f_ck / gamma_c",
    description="Cálculo da resistência de cálculo do concreto:"
)

# Adiciona Cálculo Inteligente (2) - Teste de Contexto
# O 'f_cd' calculado acima é usado aqui.
#
doc.add_calculation(
    "sigma_cd_max = 0.85 * f_cd",
    description="Tensão máxima de cálculo no concreto:"
)

# Adiciona Cálculo Inteligente (3) - Teste de outra variável
doc.add_calculation(
    "M_d = gamma_f * M_k",
    description="Cálculo do momento fletor de cálculo:"
)

# Adiciona Figura
img_path = Path("figura_teste.png")
create_placeholder_image(img_path)
doc.add_figure(
    path=str(img_path),
    caption="Diagrama de esforços (imagem placeholder)."
)

# --- 5. RENDERIZAÇÃO FINAL ---
output_file = "memorial_final.html"
logger.info(f"Renderizando o documento HTML: {output_file}...")

# Chama o renderizador HTML robusto que implementamos
#
doc.render_html(output_file, open_on_save=True)

logger.info(f"Memorial gerado com sucesso! Arquivo salvo em: {os.path.abspath(output_file)}")