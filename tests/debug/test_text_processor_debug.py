# tests/debug/test_text_processor_debug.py
"""
Script de Debug Completo para text_processor.py v2.0

Valida TODOS os ajustes implementados:
1. ✅ validate_template() completo (detecta malformados)
2. ✅ Auto-detecção de variáveis engenharia (M_k, gamma_s)
3. ✅ Conversão LaTeX automática (M_k → $M_{k}$)
4. ✅ safe_eval_expression() seguro (AST, não SymPy)
5. ✅ Formatação inteligente (Pint, float, int)
6. ✅ Backward compatibility (TextProcessor wrapper)
7. ✅ Performance (tempo de execução)
8. ✅ Segurança (injection attacks bloqueados)

Usage:
    python tests/debug/test_text_processor_debug.py
    
    # Ou com pytest:
    pytest tests/debug/test_text_processor_debug.py -v -s

Author: PyMemorial Team
Date: 2025-10-21
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Adiciona src ao path (para import direto)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Colors para output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_test_header(title: str):
    """Imprime header de teste."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(msg: str):
    """Imprime mensagem de sucesso."""
    print(f"{Colors.OKGREEN}✅ {msg}{Colors.ENDC}")

def print_fail(msg: str):
    """Imprime mensagem de falha."""
    print(f"{Colors.FAIL}❌ {msg}{Colors.ENDC}")

def print_info(msg: str):
    """Imprime mensagem informativa."""
    print(f"{Colors.OKCYAN}ℹ️  {msg}{Colors.ENDC}")

def print_warning(msg: str):
    """Imprime aviso."""
    print(f"{Colors.WARNING}⚠️  {msg}{Colors.ENDC}")


# ============================================================================
# TEST 1: IMPORTS E DISPONIBILIDADE
# ============================================================================

def test_imports():
    """Testa se todos os imports estão funcionando."""
    print_test_header("TEST 1: IMPORTS E DISPONIBILIDADE")
    
    try:
        from pymemorial.recognition.text_processor import (
            SmartTextEngine,
            TextProcessor,
            DetectedVariable,  # FIX: Nome correto
            get_engine,
            PLACEHOLDER
        )
        print_success("Imports básicos OK")
        
        # Testa instanciação
        engine = SmartTextEngine()
        print_success(f"SmartTextEngine instanciado: {type(engine).__name__}")
        
        processor = TextProcessor()
        print_success(f"TextProcessor instanciado: {type(processor).__name__}")
        
        # Testa singleton
        engine1 = get_engine()
        engine2 = get_engine()
        assert engine1 is engine2, "Singleton falhou!"
        print_success("get_engine() singleton OK")
        
        return True
    
    except Exception as e:
        print_fail(f"Import falhou: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 2: VALIDATE_TEMPLATE() COMPLETO
# ============================================================================

def test_validate_template():
    """Testa validação completa de templates."""
    print_test_header("TEST 2: VALIDATE_TEMPLATE() COMPLETO")
    
    from pymemorial.recognition.text_processor import SmartTextEngine
    engine = SmartTextEngine()
    
    # Test case 1: Template válido
    print_info("Test 2.1: Template válido")
    template_valid = "Momento {{M_k}} = 150 kN, fator {{gamma_s}} = 1.4"
    is_valid, vars_required = engine.validate_template(template_valid)
    
    assert is_valid == True, "Template válido detectado como inválido!"
    assert set(vars_required) == {'M_k', 'gamma_s'}, f"Variáveis incorretas: {vars_required}"
    print_success(f"Template válido: is_valid={is_valid}, vars={vars_required}")
    
    # Test case 2: Template com chave malformada {var}
    print_info("Test 2.2: Template com chave malformada")
    template_malformed = "Momento {M_k} = 150 kN"  # ❌ {M_k} ao invés de {{M_k}}
    is_valid, vars_required = engine.validate_template(template_malformed)
    
    assert is_valid == False, "Template malformado não detectado!"
    print_success(f"Template malformado detectado: is_valid={is_valid}")
    
    # Test case 3: Template com chaves desbalanceadas
    print_info("Test 2.3: Template com chaves desbalanceadas")
    template_unbalanced = "Momento {{M_k} = 150 kN"  # ❌ Falta }
    is_valid, vars_required = engine.validate_template(template_unbalanced)
    
    assert is_valid == False, "Chaves desbalanceadas não detectadas!"
    print_success(f"Chaves desbalanceadas detectadas: is_valid={is_valid}")
    
    # Test case 4: Template complexo com múltiplas variáveis
    print_info("Test 2.4: Template complexo")
    template_complex = """
    Memorial de Cálculo:
    - Momento característico: {{M_k}} kN.m
    - Fator de segurança: {{gamma_s}}
    - Momento de cálculo: {{M_d}} kN.m
    - Resistência: {{M_Rd}} kN.m
    """
    is_valid, vars_required = engine.validate_template(template_complex)
    
    assert is_valid == True
    assert len(vars_required) == 4
    print_success(f"Template complexo OK: {len(vars_required)} variáveis detectadas")
    
    return True


# ============================================================================
# TEST 3: AUTO-DETECÇÃO DE VARIÁVEIS
# ============================================================================

def test_auto_detection():
    """Testa detecção automática de variáveis de engenharia."""
    print_test_header("TEST 3: AUTO-DETECÇÃO DE VARIÁVEIS")
    
    from pymemorial.recognition.text_processor import SmartTextEngine
    engine = SmartTextEngine(enable_auto_detect=True)
    
    # Test case 1: Variáveis simples
    print_info("Test 3.1: Detecção de variáveis simples")
    text = "O momento M_k = 150 kN e força N_Rd = 3650 kN"
    detected = engine._detect_engineering_variables(text)
    
    detected_names = [v.name for v in detected]
    assert 'M_k' in detected_names, "M_k não detectado!"
    assert 'N_Rd' in detected_names, "N_Rd não detectado!"
    print_success(f"Detectadas {len(detected)} variáveis: {detected_names}")
    
    # Test case 2: Variáveis gregas
    print_info("Test 3.2: Detecção de variáveis gregas")
    text_greek = "Coeficiente gamma_s = 1.4 e tensão sigma_max = 250 MPa"
    detected_greek = engine._detect_engineering_variables(text_greek)
    
    detected_greek_names = [v.name for v in detected_greek]
    assert 'gamma_s' in detected_greek_names, "gamma_s não detectado!"
    assert 'sigma_max' in detected_greek_names, "sigma_max não detectado!"
    
    # Verifica flag is_greek
    gamma_var = next(v for v in detected_greek if v.name == 'gamma_s')
    assert gamma_var.is_greek == True, "gamma_s não marcado como grego!"
    print_success(f"Variáveis gregas OK: {detected_greek_names}")
    
    # Test case 3: Falso positivos (palavras comuns NÃO devem ser detectadas)
    print_info("Test 3.3: Teste de falso positivos")
    text_false_positive = "O momento M_k para o pilar é calculado como 150 kN via norma."
    detected_fp = engine._detect_engineering_variables(text_false_positive)
    
    detected_fp_names = [v.name for v in detected_fp]
    assert 'M_k' in detected_fp_names, "M_k deveria ser detectado!"
    assert 'para' not in detected_fp_names, "Falso positivo: 'para' detectado!"
    assert 'como' not in detected_fp_names, "Falso positivo: 'como' detectado!"
    assert 'via' not in detected_fp_names, "Falso positivo: 'via' detectado!"
    print_success(f"Sem falso positivos: apenas {detected_fp_names} detectado")
    
    return True


# ============================================================================
# TEST 4: CONVERSÃO LATEX AUTOMÁTICA
# ============================================================================

def test_latex_conversion():
    """Testa conversão automática para LaTeX."""
    print_test_header("TEST 4: CONVERSÃO LATEX AUTOMÁTICA")
    
    from pymemorial.recognition.text_processor import SmartTextEngine
    engine = SmartTextEngine(enable_latex=True, enable_auto_detect=True)
    
    # Test case 1: Variável simples com subscrito
    print_info("Test 4.1: M_k → $M_{k}$")
    text = "Momento M_k = 150 kN"
    context = {'M_k': 150}
    processed = engine.process_text(text, context)
    
    assert '$M_{k}$' in processed, f"LaTeX não gerado! Resultado: {processed}"
    print_success(f"LaTeX gerado: {processed}")
    
    # Test case 2: Variável grega
    print_info("Test 4.2: gamma_s → $\\gamma_{s}$")
    text_greek = "Fator gamma_s = 1.4"
    context_greek = {'gamma_s': 1.4}
    processed_greek = engine.process_text(text_greek, context_greek)
    
    # Aceita tanto \gamma_{s} quanto γ_{s}
    has_latex = r'$\gamma_{s}$' in processed_greek or '$γ_{s}$' in processed_greek
    assert has_latex, f"LaTeX grego não gerado! Resultado: {processed_greek}"
    print_success(f"LaTeX grego gerado: {processed_greek}")
    
    # Test case 3: Múltiplas variáveis
    print_info("Test 4.3: Múltiplas variáveis")
    text_multi = "M_k = 150 kN, N_Rd = 3650 kN, gamma_s = 1.4"
    context_multi = {'M_k': 150, 'N_Rd': 3650, 'gamma_s': 1.4}
    processed_multi = engine.process_text(text_multi, context_multi)
    
    assert '$M_{k}$' in processed_multi
    assert '$N_{Rd}$' in processed_multi
    print_success(f"Múltiplas variáveis LaTeX OK")
    
    return True


# ============================================================================
# TEST 5: SAFE_EVAL_EXPRESSION() (SEGURANÇA)
# ============================================================================

def test_safe_eval():
    """Testa avaliação segura de expressões."""
    print_test_header("TEST 5: SAFE_EVAL_EXPRESSION() (SEGURANÇA)")
    
    from pymemorial.recognition.text_processor import SmartTextEngine
    engine = SmartTextEngine()
    
    # Test case 1: Expressão válida simples
    print_info("Test 5.1: Expressão válida")
    expr = "M_k * gamma_s"
    context = {'M_k': 150, 'gamma_s': 1.4}
    result = engine.safe_eval_expression(expr, context)
    
    expected = 150 * 1.4
    assert result == expected, f"Resultado incorreto: {result} != {expected}"
    print_success(f"Expressão avaliada: {expr} = {result}")
    
    # Test case 2: Expressão com potência
    print_info("Test 5.2: Expressão com potência")
    expr_pow = "M_k ** 2"
    context_pow = {'M_k': 10}
    result_pow = engine.safe_eval_expression(expr_pow, context_pow)
    
    assert result_pow == 100, f"Potência incorreta: {result_pow}"
    print_success(f"Potência OK: {expr_pow} = {result_pow}")
    
    # Test case 3: ⚠️ SEGURANÇA - Injection attack bloqueado
    print_info("Test 5.3: ⚠️ TESTE DE SEGURANÇA - Injection attack")
    malicious_expressions = [
        "__import__('os').system('ls')",
        "eval('1+1')",
        "exec('print(\"pwned\")')",
        "open('/etc/passwd').read()",
    ]
    
    for malicious in malicious_expressions:
        result_malicious = engine.safe_eval_expression(malicious, {})
        assert result_malicious is None, f"Injection NÃO bloqueado: {malicious}"
        print_success(f"Injection bloqueado: {malicious[:50]}")
    
    # Test case 4: Expressão com parênteses
    print_info("Test 5.4: Expressão com parênteses")
    expr_paren = "(M_k + N_Rd) / gamma_s"
    context_paren = {'M_k': 100, 'N_Rd': 200, 'gamma_s': 2.0}
    result_paren = engine.safe_eval_expression(expr_paren, context_paren)
    
    expected_paren = (100 + 200) / 2.0
    assert result_paren == expected_paren, f"Resultado incorreto: {result_paren}"
    print_success(f"Parênteses OK: {expr_paren} = {result_paren}")
    
    return True


# ============================================================================
# TEST 6: FORMATAÇÃO INTELIGENTE (PINT)
# ============================================================================

def test_formatting():
    """Testa formatação inteligente de valores."""
    print_test_header("TEST 6: FORMATAÇÃO INTELIGENTE")
    
    from pymemorial.recognition.text_processor import SmartTextEngine
    engine = SmartTextEngine()
    
    # Test case 1: Float
    print_info("Test 6.1: Formatação de float")
    formatted_float = engine._format_value(1.4)
    assert formatted_float == "1.40", f"Float mal formatado: {formatted_float}"
    print_success(f"Float OK: 1.4 → {formatted_float}")
    
    # Test case 2: Int
    print_info("Test 6.2: Formatação de int")
    formatted_int = engine._format_value(150)
    assert formatted_int == "150", f"Int mal formatado: {formatted_int}"
    print_success(f"Int OK: 150 → {formatted_int}")
    
    # Test case 3: String
    print_info("Test 6.3: Formatação de string")
    formatted_str = engine._format_value("característico")
    assert formatted_str == "característico"
    print_success(f"String OK: {formatted_str}")
    
    # Test case 4: Pint (se disponível)
    print_info("Test 6.4: Formatação de Pint Quantity (se disponível)")
    try:
        from pint import Quantity, UnitRegistry
        ureg = UnitRegistry()
        
        pint_value = ureg('150 kilonewton')
        formatted_pint = engine._format_value(pint_value)
        
        # Verifica se tem kN (abreviado) ou kilonewton
        assert 'kN' in formatted_pint or 'kilonewton' in formatted_pint, \
            f"Unidade não formatada: {formatted_pint}"
        assert '150' in formatted_pint, f"Magnitude não formatada: {formatted_pint}"
        print_success(f"Pint OK: ureg('150 kilonewton') → {formatted_pint}")
    
    except ImportError:
        print_warning("Pint não instalado - teste pulado")
    
    return True


# ============================================================================
# TEST 7: BACKWARD COMPATIBILITY
# ============================================================================

def test_backward_compatibility():
    """Testa compatibilidade com API v1.0."""
    print_test_header("TEST 7: BACKWARD COMPATIBILITY (API v1.0)")
    
    from pymemorial.recognition.text_processor import TextProcessor
    processor = TextProcessor()
    
    # Test case 1: render() método original
    print_info("Test 7.1: Método render() (API v1.0)")
    template = "Resistência {{fck}} = {{valor}} MPa"
    context = {"fck": "característica", "valor": 30}
    result = processor.render(template, context)
    
    expected = "Resistência característica = 30 MPa"
    assert result == expected, f"Render falhou: {result}"
    print_success(f"render() OK: {result}")
    
    # Test case 2: to_latex() método original
    print_info("Test 7.2: Método to_latex() (API v1.0)")
    text = "Resistência fck = 30 MPa com 10% de variação"
    latex_result = processor.to_latex(text)
    
    # Verifica escape de caracteres especiais
    assert r'\%' in latex_result or '%' not in latex_result, \
        f"% não escapado: {latex_result}"
    print_success(f"to_latex() OK")
    
    # Test case 3: extract_and_replace() método original
    print_info("Test 7.3: Método extract_and_replace() (API v1.0)")
    text_extract = "Valor {{x}} e {{y}}"
    replacements = {"x": "150", "y": "200"}
    replaced = processor.extract_and_replace(text_extract, replacements)
    
    assert "150" in replaced and "200" in replaced
    print_success(f"extract_and_replace() OK: {replaced}")
    
    # Test case 4: validate_template() método original (agora completo!)
    print_info("Test 7.4: Método validate_template() (API v1.0)")
    template_validate = "M = {{M_k}} * {{gamma}}"
    is_valid, vars_list = processor.validate_template(template_validate)
    
    assert is_valid == True
    assert set(vars_list) == {'M_k', 'gamma'}
    print_success(f"validate_template() OK: vars={vars_list}")
    
    return True


# ============================================================================
# TEST 8: PERFORMANCE (BENCHMARK)
# ============================================================================

def test_performance():
    """Testa performance (tempo de execução)."""
    print_test_header("TEST 8: PERFORMANCE (BENCHMARK)")
    
    from pymemorial.recognition.text_processor import SmartTextEngine
    engine = SmartTextEngine()
    
    # Test case 1: Tempo de import (já importado, mas testa reimport)
    print_info("Test 8.1: Tempo de processamento básico")
    text_short = "Momento M_k = {{M_k}} kN"
    context_short = {'M_k': 150}
    
    start = time.perf_counter()
    for _ in range(100):  # 100 iterações
        engine.process_text(text_short, context_short)
    end = time.perf_counter()
    
    avg_time_ms = ((end - start) / 100) * 1000
    print_success(f"Tempo médio (100 iterações): {avg_time_ms:.3f} ms")
    assert avg_time_ms < 10, f"Muito lento: {avg_time_ms:.3f} ms"
    
    return True


# ============================================================================
# TEST 9: CASOS EDGE (EDGE CASES)
# ============================================================================

def test_edge_cases():
    """Testa casos extremos e edge cases."""
    print_test_header("TEST 9: CASOS EDGE (EDGE CASES)")
    
    from pymemorial.recognition.text_processor import SmartTextEngine
    engine = SmartTextEngine()
    
    # Test case 1: Texto vazio
    print_info("Test 9.1: Texto vazio")
    result_empty = engine.process_text("", {})
    assert result_empty == "", "Texto vazio falhou!"
    print_success("Texto vazio OK")
    
    # Test case 2: Context vazio
    print_info("Test 9.2: Context vazio")
    text_no_context = "Momento M_k = valor"
    result_no_context = engine.process_text(text_no_context, {})
    assert "M_k" in result_no_context or "$M_{k}$" in result_no_context
    print_success("Context vazio OK")
    
    # Test case 3: Variável não encontrada no contexto
    print_info("Test 9.3: Variável não encontrada")
    text_missing_var = "Valor {{x}} e {{y}}"
    context_missing = {"x": 100}  # y está faltando
    result_missing = engine.process_text(text_missing_var, context_missing)
    assert "100" in result_missing
    assert "{{y}}" in result_missing
    print_success("Variável faltando OK")
    
    return True


# ============================================================================
# TEST 10: INTEGRAÇÃO COMPLETA (END-TO-END)
# ============================================================================

def test_integration():
    """Teste de integração completo (end-to-end real-world)."""
    print_test_header("TEST 10: INTEGRAÇÃO COMPLETA (REAL-WORLD)")
    
    from pymemorial.recognition.text_processor import SmartTextEngine
    engine = SmartTextEngine(enable_latex=True, enable_auto_detect=True)
    
    # Simula texto real de memorial de cálculo
    print_info("Test 10.1: Memorial de cálculo real (NBR 8800)")
    
    memorial_text = """
Memorial de Cálculo - Pilar de Aço

1. Dados de Entrada
O momento característico M_k = {{M_k}} kN.m atua no pilar.
A força axial de cálculo N_Sd = {{N_Sd}} kN.

2. Coeficientes de Segurança
Conforme NBR 8800:2024, o fator de segurança gamma_s = {{gamma_s}}.

3. Verificação de Resistência
A resistência de cálculo N_Rd = {{N_Rd}} kN supera a solicitação.

Verificação: N_Sd <= N_Rd ✅ OK
    """
    
    context_memorial = {
        'M_k': 150,
        'N_Sd': 2500,
        'gamma_s': 1.4,
        'N_Rd': 3650
    }
    
    start_integration = time.perf_counter()
    result_memorial = engine.process_text(memorial_text, context_memorial)
    end_integration = time.perf_counter()
    
    integration_time_ms = (end_integration - start_integration) * 1000
    
    # Verificações
    assert '$M_{k}$' in result_memorial, "LaTeX M_k não gerado!"
    assert '$N_{Sd}$' in result_memorial, "LaTeX N_Sd não gerado!"
    assert '150' in result_memorial, "Valor M_k não substituído!"
    assert '2500' in result_memorial, "Valor N_Sd não substituído!"
    
    print_success(f"Memorial processado em {integration_time_ms:.2f} ms")
    print_info(f"Preview (primeiros 150 chars):\n{result_memorial[:150]}...")
    
    return True


# ============================================================================
# RUNNER PRINCIPAL
# ============================================================================

def main():
    """Executa todos os testes."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║         PYMEMORIAL TEXT_PROCESSOR v2.0 - DEBUG COMPLETO                  ║")
    print("║                                                                           ║")
    print("║         Validando TODOS os ajustes implementados                         ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    tests = [
        ("Imports e Disponibilidade", test_imports),
        ("validate_template() Completo", test_validate_template),
        ("Auto-Detecção de Variáveis", test_auto_detection),
        ("Conversão LaTeX Automática", test_latex_conversion),
        ("safe_eval_expression() Seguro", test_safe_eval),
        ("Formatação Inteligente", test_formatting),
        ("Backward Compatibility", test_backward_compatibility),
        ("Performance (Benchmark)", test_performance),
        ("Casos Edge", test_edge_cases),
        ("Integração Completa (Real-World)", test_integration),
    ]
    
    results = []
    total_start = time.perf_counter()
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print_fail(f"ERRO no teste '{test_name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    total_end = time.perf_counter()
    total_time_s = total_end - total_start
    
    # ========================================================================
    # RELATÓRIO FINAL
    # ========================================================================
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                         RELATÓRIO FINAL                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    success_rate = (passed / len(results)) * 100 if results else 0
    
    print(f"{Colors.BOLD}Resumo dos Testes:{Colors.ENDC}")
    print(f"  Total de testes: {len(results)}")
    print(f"  {Colors.OKGREEN}✅ Passaram: {passed}{Colors.ENDC}")
    print(f"  {Colors.FAIL}❌ Falharam: {failed}{Colors.ENDC}")
    print(f"  Taxa de sucesso: {Colors.BOLD}{success_rate:.1f}%{Colors.ENDC}")
    print(f"  Tempo total: {Colors.BOLD}{total_time_s:.2f}s{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Detalhes:{Colors.ENDC}")
    for test_name, success in results:
        status = f"{Colors.OKGREEN}✅ PASS{Colors.ENDC}" if success else \
                 f"{Colors.FAIL}❌ FAIL{Colors.ENDC}"
        print(f"  {status} - {test_name}")
    
    # Veredicto final
    print(f"\n{Colors.BOLD}Veredicto Final:{Colors.ENDC}")
    if failed == 0:
        print(f"{Colors.OKGREEN}{Colors.BOLD}")
        print("╔═══════════════════════════════════════════════════════════════════════════╗")
        print("║                                                                           ║")
        print("║                   🎉 TODOS OS TESTES PASSARAM! 🎉                        ║")
        print("║                                                                           ║")
        print("║              text_processor.py v2.0 ESTÁ PRONTO PARA PRODUÇÃO!           ║")
        print("║                                                                           ║")
        print("╚═══════════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}")
        print("╔═══════════════════════════════════════════════════════════════════════════╗")
        print("║                                                                           ║")
        print(f"║              ⚠️  {failed} TESTE(S) FALHARAM - REVISAR CÓDIGO! ⚠️               ║")
        print("║                                                                           ║")
        print("╚═══════════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
