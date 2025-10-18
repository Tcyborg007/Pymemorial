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
