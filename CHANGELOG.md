# 📋 CHANGELOG - PyMemorial v2.0

Todas as mudanças notáveis do projeto PyMemorial serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0-alpha.7.3] - 2025-10-20

### ✨ Novidades

#### **PHASE 7.3 - WeasyPrint PDF Generator (COMPLETO)**

- **BaseGenerator (ABC)**: Interface abstrata para geradores de documentos
  - `GeneratorConfig`: Configuração de página, margens, metadata
  - `PageConfig`: Tamanho (A4, Letter), orientação, margens customizáveis
  - `PDFMetadata`: Metadata Dublin Core (title, author, subject, keywords)
  - `GenerationError`, `ValidationError`: Tratamento de erros específico

- **WeasyPrintGenerator**: Gerador de PDF profissional com WeasyPrint
  - ✅ Numeração automática de páginas (CSS @page counter)
  - ✅ Headers/Footers customizáveis (CSS @top-center, @bottom-center)
  - ✅ Margens personalizadas (mm)
  - ✅ Suporte a first page (sem header/footer)
  - ✅ Suporte a left/right pages (páginas pares/ímpares)
  - ✅ Embedding de imagens (file:/// URI)
  - ✅ Metadata PDF (Dublin Core)
  - ✅ Geração in-memory (`to_bytes()`)
  - ✅ Debug mode (salva HTML intermediário)

- **memorial.to_pdf()**: Método de conveniência para geração de PDF
  ```
  memorial.to_pdf("output.pdf")
  memorial.to_pdf("output.pdf", generator='weasyprint', config=custom_config)
  ```

- **memorial.render_to_string()**: Renderização HTML em memória
  - Usado internamente pelos geradores de PDF
  - Suporte a UTF-8 com fallback para Windows
  - Tratamento de arquivos temporários

### 🐛 Correções

#### **Windows Compatibility**

- **Caminhos de arquivo → URIs**: Conversão automática de paths Windows para `file:///`
  - Antes: `C:\Users\...\image.png` → ❌ Erro WeasyPrint
  - Depois: `file:///C:/Users/.../image.png` → ✅ Funciona
  - Implementação: `Path.as_uri()` em `memorial.py::_generate_html_content()`

- **Encoding UTF-8 no Windows**: Correção de UnicodeDecodeError
  - Antes: `tempfile.NamedTemporaryFile()` → encoding cp1252 → ❌ Erro
  - Depois: `tempfile.mkstemp()` + explicit UTF-8 → ✅ Funciona
  - Implementação: `memorial.py::render_to_string()` com fallback

- **Image paths relativos**: Conversão para paths absolutos
  - Implementação: `Path.resolve()` garante paths absolutos
  - Previne erros de "file not found" no WeasyPrint

### 🔧 Melhorias

- **base_document.py**: Adicionado método abstrato `render_to_string()`
  - Define contrato para todas as subclasses (Memorial, Report, Article)
  - Documentação NumPy completa

- **Logging aprimorado**: Todos os generators usam logging estruturado
  - DEBUG: Detalhes de rendering
  - INFO: Operações bem-sucedidas
  - ERROR: Falhas com traceback

- **Validação de documentos**: Validação antes de gerar PDF
  - Verifica metadata obrigatória
  - Verifica conteúdo mínimo (seções, etc)

### 🧪 Testes

- **test_pdf_generation_debug.py**: Suite completa de testes (5/5 passando)
  - ✅ TEST 1: Basic PDF Generation (18.4 KB)
  - ✅ TEST 2: PDF with Custom Configuration (18.7 KB)
  - ✅ TEST 3: memorial.to_pdf() Convenience Method (18.4 KB)
  - ✅ TEST 4: PDF to Bytes (In-Memory) (18.4 KB)
  - ✅ TEST 5: Error Handling (ValueError)

- **Performance**: ~317ms por PDF em média ⚡

### 📦 Dependências

```
weasyprint = ">=60.0"
pyphen = ">=0.14.0"
```

### 📁 Arquivos Criados

```
src/pymemorial/document/generators/
├── __init__.py (exports: BaseGenerator, WeasyPrintGenerator, etc)
├── base_generator.py (300 linhas - ABC + Config classes)
└── weasyprint_generator.py (420 linhas - Implementação completa)
```

### 📊 Estatísticas

- **Linhas adicionadas**: ~800 linhas
- **Arquivos modificados**: 5 (base_document.py, memorial.py, generators/*)
- **Testes criados**: 1 (5 test cases)
- **Cobertura**: 100% dos geradores testados

---

## [2.0.0-alpha.7.2] - 2025-10-18

### ✨ Novidades

#### **PHASE 7.2 - Auto-Numbering System (COMPLETO)**

- **Auto-numeração de figuras**: `add_figure()` incrementa automaticamente
  - Exemplo: "Figura 1", "Figura 2", "Figura 3", ...
  - Método: `base_document.py::add_figure()`

- **Auto-numeração de tabelas**: `add_table()` incrementa automaticamente
  - Exemplo: "Tabela 1", "Tabela 2", "Tabela 3", ...
  - Método: `base_document.py::add_table()`

- **Auto-numeração de equações**: `add_equation()` incrementa automaticamente
  - Exemplo: "Eq. 1", "Eq. 2", "Eq. 3", ...
  - Método: `base_document.py::add_equation()`

- **Listas automáticas**:
  - `get_list_of_figures()`: Retorna lista ordenada de figuras
  - `get_list_of_tables()`: Retorna lista ordenada de tabelas
  - `get_list_of_equations()`: Retorna lista ordenada de equações

### 🧪 Testes

- `test_auto_numbering_debug.py`: Testes de numeração automática
- `test_auto_lists_debug.py`: Testes de listas automáticas

---

## [2.0.0-alpha.7.1] - 2025-10-17

### ✨ Novidades

#### **PHASE 7.1 - Base Document Structure (COMPLETO)**

- **BaseDocument (ABC)**: Classe abstrata base para todos os documentos
  - 2400 linhas de código
  - Metadata system (author, company, revisions)
  - Validation system
  - Section management
  - Cross-reference system

- **Memorial**: Implementação concreta para memoriais de cálculo
  - 1050 linhas de código
  - Templates NBR 8800, AISC 360, Eurocode
  - Automatic TOC generation
  - Verification system

### 📁 Estrutura Criada

```
src/pymemorial/document/
├── __init__.py
├── base_document.py (2400 linhas)
└── memorial.py (1050 linhas)
```

---

## [2.0.0-alpha.6.0] - 2025-10-15

### ✨ Novidades

- **Visualization Module**: Exportadores 3D (PyVista, Plotly)
- **Factory Pattern**: `VisualizationFactory` para criar exportadores

---

## [2.0.0-alpha.5.0] - 2025-10-10

### ✨ Novidades

- **FEM Backends**: Integração com AnalysisP, OpenSees, SAP2000
- **Cross-platform**: Suporte Windows, Linux, macOS

---

## [Unreleased]

### 🔜 Próximas Features (PHASE 7.4)

- **Templates & Styles**: CSS profissional (NBR, AISC, Modern)
- **Jinja2 Templates**: HTML flexível e reutilizável
- **ABNT Elements**: Capa, folha de rosto, listas automáticas

### 🔜 Próximas Features (PHASE 7.5)

- **Bibliografia NBR 6023**: Sistema completo de referências
- **Citações inline**: Parser automático de citações
- **Formatação ABNT**: Formatador automático segundo normas

### 🔜 Próximas Features (PHASE 7.6+)

- **Quarto Generator**: Documentos acadêmicos ABNT
- **Playwright Generator**: 3D models + PDF
- **LaTeX Generator**: ABNTeX2 profissional

---

## 🔗 Links

- **Repositório**: [GitHub](https://github.com/yourusername/pymemorial)
- **Documentação**: [Read the Docs](https://pymemorial.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/pymemorial/issues)

---

## 📝 Notas de Versão

### Formato de Versionamento

```
MAJOR.MINOR.PATCH-alpha.PHASE.SUBPHASE

Exemplo: 2.0.0-alpha.7.3
         │ │ │       │ │
         │ │ │       │ └─ Subphase (0-9)
         │ │ │       └─── Phase (1-8)
         │ │ └─────────── Patch version
         │ └───────────── Minor version
         └─────────────── Major version
```

### Tipos de Mudanças

- **✨ Novidades**: Novas features
- **🐛 Correções**: Bug fixes
- **🔧 Melhorias**: Melhorias em features existentes
- **⚡ Performance**: Otimizações de performance
- **📚 Documentação**: Mudanças na documentação
- **🧪 Testes**: Adição ou correção de testes
- **🔥 Breaking Changes**: Mudanças incompatíveis com versões anteriores

---

Desenvolvido com 🇧🇷 por PyMemorial Team