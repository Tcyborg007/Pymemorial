"""Testes de ContentBlock."""
import pytest
from pymemorial.builder.content import (
    ContentBlock,
    ContentType,
    create_text_block,
    create_equation_block,
    create_figure_block,
    create_table_block,
)


def test_content_block_creation():
    """Testa criação de bloco de conteúdo."""
    block = ContentBlock(type=ContentType.TEXT, content="Texto de teste")
    assert block.type == ContentType.TEXT
    assert block.content == "Texto de teste"


def test_text_block_helper():
    """Testa helper de texto."""
    block = create_text_block("Hello")
    assert block.type == ContentType.TEXT
    assert block.content == "Hello"


def test_equation_block_helper():
    """Testa helper de equação."""
    block = create_equation_block("x = 5", label="eq:1")
    assert block.type == ContentType.EQUATION
    assert block.label == "eq:1"


def test_figure_block_helper():
    """Testa helper de figura."""
    block = create_figure_block("image.png", caption="Figura 1", label="fig:1")
    assert block.type == ContentType.FIGURE
    assert block.caption == "Figura 1"
    assert block.label == "fig:1"


def test_table_block_helper():
    """Testa helper de tabela."""
    data = [["A", "B"], [1, 2]]
    block = create_table_block(data, caption="Tabela 1", label="tab:1")
    assert block.type == ContentType.TABLE
    assert block.content == data


def test_to_dict_text():
    """Testa serialização de texto."""
    block = create_text_block("Test")
    data = block.to_dict()
    assert data["type"] == "text"
    assert data["content"] == "Test"


def test_to_dict_with_caption():
    """Testa serialização com legenda."""
    block = create_figure_block("img.png", caption="Imagem", label="fig:1")
    data = block.to_dict()
    assert "caption" in data
    assert data["caption"] == "Imagem"
    assert data["label"] == "fig:1"


def test_serialize_equation_with_latex():
    """Testa serialização de equação com método latex()."""
    class MockEquation:
        def latex(self):
            return r"x^2 + y^2 = r^2"
    
    block = ContentBlock(type=ContentType.EQUATION, content=MockEquation())
    data = block.to_dict()
    assert data["content"] == r"x^2 + y^2 = r^2"
def test_serialize_content_code():
    """Testa serialização de código."""
    block = ContentBlock(type=ContentType.CODE, content="x = 5")
    data = block.to_dict()
    assert data["content"] == "x = 5"


def test_serialize_content_table():
    """Testa serialização de tabela."""
    table_data = [["A", "B"], [1, 2]]
    block = ContentBlock(type=ContentType.TABLE, content=table_data)
    data = block.to_dict()
    assert data["content"] == table_data


def test_serialize_content_figure_dict():
    """Testa serialização de figura como dict."""
    fig_data = {"path": "img.png", "dpi": 300}
    block = ContentBlock(type=ContentType.FIGURE, content=fig_data)
    data = block.to_dict()
    assert data["content"] == fig_data
