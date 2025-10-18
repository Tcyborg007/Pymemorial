# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

## [0.1.0] - 2025-10-17

### Added
- **Núcleo de cálculo (core)**: sistema completo de cálculo simbólico e numérico
  - `units.py`: integração Pint + forallpeople
  - `variable.py`: variáveis com tipos e unidades
  - `equation.py`: equações simbólicas com SymPy
  - `calculator.py`: motor de cálculo com cache
  - `cache.py`: sistema de cache
- **Testes**: 31 testes unitários com 100% cobertura no core
- **CLI**: comandos `init` e `version`
- **Infraestrutura**: Poetry, pytest, coverage

### Planejado
- Fase 2: Recognition (parsing automático)
- Fase 3: Builder (API fluente)
- Fase 4: Backends (PyNite, OpenSees)

## [Unreleased]

### Added
- **Módulo Recognition**: sistema de reconhecimento automático de variáveis
  - `greek.py`: conversão bidirecional de símbolos gregos (Unicode ↔ ASCII)
  - `patterns.py`: padrões regex compilados para variáveis, números, unidades
  - `parser.py`: parser inteligente de variáveis com valores, unidades e descrições
  - `text_processor.py`: processador de templates com placeholders `{{var}}`
- **37 testes unitários** para recognition com 100% de cobertura
- Suporte a variáveis customizadas (ex: `KL_pt_nb`) reconhecidas automaticamente
- Validação de templates e detecção de placeholders malformados

## [Unreleased]

### Added
- **Módulo Builder**: API fluente para construção de memoriais (95% cobertura)
  - `memorial.py`: MemorialBuilder com encadeamento de métodos
  - `section.py`: Section com suporte a hierarquia e subseções
  - `content.py`: ContentBlock com tipos (text, equation, figure, table)
  - `validators.py`: validação de nomes, níveis e templates
- **37 testes unitários** para builder
- Integração completa: Builder → Recognition → Core → Calculator
- Exemplo completo de dimensionamento de viga biapoiada
- Suporte a templates `{{var}}` com substituição automática
- Exportação para dict/JSON do memorial completo

## [Unreleased]

### Added
- **Módulo Backends**: Sistema de backends estruturais (86%+ cobertura)
  - `base.py`: Classes abstratas StructuralBackend com interface unificada
  - `factory.py`: BackendFactory com detecção automática de backends instalados
  - `pynite.py`: Backend Pynite para análise linear de estruturas 3D
  - `opensees.py`: Backend OpenSeesPy para análise não-linear
  - `adapter.py`: Sistema de adapters para conversão de modelos (dict → backend)
- **21 testes unitários** para backends (base: 5, factory: 4, pynite: 4, opensees: 3, adapters: 5)
- Factory pattern que detecta Pynite e OpenSees automaticamente
- SimpleFrameAdapter: conversão JSON/dict → modelo estrutural
- Método `analyze_and_get_results()` retorna deslocamentos e esforços
- Suporte a materiais (aço 200 GPa) e seções (IPE200)
- Apoios parametrizáveis (dx, dy, dz, rx, ry, rz)
- Cargas nodais em qualquer direção (X, Y, Z)
- Integração numérica Gauss-Lobatto (5 pontos) no OpenSees
- Elementos dispBeamColumn com 6 graus de liberdade por nó
- Testes estruturais: viga biapoiada, cantilever, cargas distribuídas
- Compatibilidade com PyNiteFEA e openseespy

### Fixed
- Correção import `from .pynite import PyniteBackend` (minúsculo)
- Correção beamIntegration tag no OpenSees (evita erro "BeamIntegrationRule not found")
- Correção apoios estruturais (3 nós + restrições de rotação para estabilidade)
