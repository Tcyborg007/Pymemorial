# Changelog

Todas as mudanças notáveis deste projeto serão documentadas neste arquivo.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

---

## [Unreleased]

### Adicionado
- Nada no momento

### Modificado
- Nada no momento

### Corrigido
- Nada no momento

---

## [0.5.2] - 2025-10-18

### Adicionado
- **SteelSection completo** com sectionproperties 3.x
  - Suporte para perfis I, tubular circular, retangular, channel
  - Propriedades geométricas: área, inércia, torção (j)
  - Propriedades plásticas: zxx, zyy
  - Raios de giração: rx, ry
  - Eixos principais: i11, i22, phi
  - Cálculo de momentos de escoamento
  - Plotagem de geometria (matplotlib)
  - Integração com SectionFactory
- 18 testes para SteelSection (100% passando)
- Conversão automática mm → m (SI)
- Safe conversion para valores None
- Material properties (E, nu, fy)

### Modificado
- Refatoração de nomes: `base.py` → `backend_base.py` e `section_base.py`
- Melhorias em `equation.py`:
  - Método `steps()` com granularidade controlada
  - `simplify_one_step()` incremental
  - `build_substitutions()` robusto
  - `format_result()` com unidades
  - 93 testes (84% cobertura)

### Corrigido
- Conflito de nomes entre `backends/base.py` e `sections/base.py`
- Imports circulares em módulos
- API do sectionproperties 3.x (mesh, warping, section_props)
- Tratamento de valores None em propriedades opcionais

---

## [0.5.1] - 2025-10-18

### Adicionado
- **SectionFactory** com detecção automática de bibliotecas
  - `available_analyzers()` para listar backends disponíveis
  - `create()` com lazy imports
  - Suporte para steel, concrete, composite
- **Section Base abstratas**
  - `SectionAnalyzer` (ABC)
  - `SectionProperties` (dataclass)
  - Conversão para dict
  - Sistema de cache
- 16 testes (9 base + 7 factory)

---

## [0.4.0] - 2025-10-17

### Adicionado
- **Backends estruturais** (Fase 4)
  - PyniteBackend com 7 testes
  - OpenSeesBackend com 2 testes
  - BackendFactory com detecção automática
  - StructuralAdapter para integração
- 21 testes de backends (86%+ cobertura)

---

## [0.3.0] - 2025-10-16

### Adicionado
- **Builder Pattern** (Fase 3)
  - MemorialBuilder
  - ContentBuilder
  - SectionBuilder
  - Validators
- 37 testes de builders

---

## [0.2.0] - 2025-10-15

### Adicionado
- **Recognition** (Fase 2)
  - Greek letter parsing
  - Pattern recognition
  - Text processor
- 37 testes de recognition

---

## [0.1.0] - 2025-10-14

### Adicionado
- **Core modules** (Fase 1)
  - Variable
  - Equation
  - Calculator
  - Units
  - Cache
- 31 testes de core

---

## [0.0.1] - 2025-10-13

### Adicionado
- Estrutura base do projeto
- Configuração Poetry
- Pytest setup
- 3 testes base

---

## Tipos de Mudanças
- **Adicionado**: Novas funcionalidades
- **Modificado**: Mudanças em funcionalidades existentes
- **Depreciado**: Funcionalidades que serão removidas
- **Removido**: Funcionalidades removidas
- **Corrigido**: Correções de bugs
- **Segurança**: Vulnerabilidades corrigidas
