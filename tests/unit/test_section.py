"""Testes de Section."""
import pytest
from pymemorial.builder.section import Section
from pymemorial.builder.content import ContentBlock, ContentType


def test_section_creation():
    """Testa criação de seção."""
    section = Section(title="Introdução", level=1)
    assert section.title == "Introdução"
    assert section.level == 1


def test_add_content():
    """Testa adição de conteúdo."""
    section = Section(title="Teste", level=1)
    block = ContentBlock(type=ContentType.TEXT, content="Texto")
    section.add_content(block)
    
    assert len(section.content) == 1


def test_add_subsection():
    """Testa adição de subseção."""
    section = Section(title="Pai", level=1)
    subsection = Section(title="Filho", level=2)
    section.add_subsection(subsection)
    
    assert len(section.subsections) == 1
    assert section.subsections[0].title == "Filho"


def test_to_dict():
    """Testa serialização."""
    section = Section(title="Teste", level=1)
    section.add_content(ContentBlock(type=ContentType.TEXT, content="abc"))
    
    data = section.to_dict()
    assert data["title"] == "Teste"
    assert len(data["content"]) == 1
