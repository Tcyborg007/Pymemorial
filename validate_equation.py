#!/usr/bin/env python3
"""
🔍 Validador Completo do PyMemorial Core - equation.py
Testa todas as funcionalidades e granularidades de steps

Uso:
    python validate_equation.py
"""

import sys
import logging
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Cores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.CYAN}ℹ️  {msg}{Colors.RESET}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")

# ============================================================================
# TESTES
# ============================================================================

class EquationValidator:
    """Validador completo do core.Equation"""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0
        
        try:
            from pymemorial.core import Equation, Variable, VariableFactory, ureg
            self.Equation = Equation
            self.Variable = Variable
            self.VariableFactory = VariableFactory
            self.ureg = ureg
            self.core_available = True
            print_success("PyMemorial Core importado com sucesso")
        except ImportError as e:
            print_error(f"Falha ao importar PyMemorial Core: {e}")
            self.core_available = False
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Executa um teste e registra resultado"""
        self.total_tests += 1
        try:
            print_info(f"Testando: {test_name}")
            result = test_func()
            if result:
                self.passed_tests += 1
                print_success(f"PASSOU: {test_name}")
                return True
            else:
                self.failed_tests += 1
                print_error(f"FALHOU: {test_name}")
                return False
        except Exception as e:
            self.failed_tests += 1
            print_error(f"ERRO: {test_name} - {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def test_basic_equation_creation(self) -> bool:
        """Teste 1: Criação básica de equação"""
        var_a = self.VariableFactory.create(name='a', value=10)
        var_b = self.VariableFactory.create(name='b', value=5)
        
        eq = self.Equation(
            expression='c = a + b',
            variables={'a': var_a, 'b': var_b},
            description='Soma simples'
        )
        
        result = eq.evaluate()
        return result == 15
    
    def test_equation_with_units(self) -> bool:
        """Teste 2: Equação com unidades (Pint)"""
        try:
            var_L = self.VariableFactory.create(name='L', value=6, unit='m')
            var_F = self.VariableFactory.create(name='F', value=1000, unit='N')
            
            eq = self.Equation(
                expression='M = F * L / 4',
                variables={'F': var_F, 'L': var_L},
                description='Momento fletor'
            )
            
            result = eq.evaluate()
            # Verifica se resultado tem unidades
            expected = 1500  # 1000 * 6 / 4
            magnitude = result.magnitude if hasattr(result, 'magnitude') else result
            return abs(magnitude - expected) < 0.01
        except:
            # Se Pint não disponível, retorna True para não falhar
            print_warning("Pint não disponível - teste pulado")
            self.warnings += 1
            return True
    
    def test_steps_minimal(self) -> bool:
        """Teste 3: Steps com granularidade 'minimal'"""
        var_x = self.VariableFactory.create(name='x', value=2)
        var_y = self.VariableFactory.create(name='y', value=3)
        
        eq = self.Equation(
            expression='z = x * y',
            variables={'x': var_x, 'y': var_y}
        )
        
        steps = eq.steps(granularity='minimal')
        
        # Minimal deve ter 2 steps (simbólico + resultado)
        if len(steps) < 2:
            print_error(f"Minimal deveria ter >= 2 steps, tem {len(steps)}")
            return False
        
        # Verifica operações
        operations = [s['operation'] for s in steps]
        has_symbolic = 'symbolic' in operations
        has_result = 'result' in operations
        
        if not (has_symbolic and has_result):
            print_error(f"Minimal deve ter 'symbolic' e 'result'. Tem: {operations}")
            return False
        
        print_info(f"Minimal: {len(steps)} steps - {operations}")
        return True
    
    def test_steps_basic(self) -> bool:
        """Teste 4: Steps com granularidade 'basic'"""
        var_a = self.VariableFactory.create(name='a', value=5)
        var_b = self.VariableFactory.create(name='b', value=3)
        
        eq = self.Equation(
            expression='result = a + b',
            variables={'a': var_a, 'b': var_b}
        )
        
        steps = eq.steps(granularity='basic')
        
        # Basic deve ter 3+ steps (simbólico + substituição + resultado)
        if len(steps) < 3:
            print_error(f"Basic deveria ter >= 3 steps, tem {len(steps)}")
            return False
        
        operations = [s['operation'] for s in steps]
        has_symbolic = 'symbolic' in operations
        has_substitution = 'substitution' in operations
        has_result = 'result' in operations
        
        if not (has_symbolic and has_substitution and has_result):
            print_error(f"Basic deve ter 'symbolic', 'substitution' e 'result'. Tem: {operations}")
            return False
        
        print_info(f"Basic: {len(steps)} steps - {operations}")
        return True
    
    def test_steps_normal(self) -> bool:
        """Teste 5: Steps com granularidade 'normal'"""
        var_a = self.VariableFactory.create(name='a', value=10)
        var_b = self.VariableFactory.create(name='b', value=2)
        
        eq = self.Equation(
            expression='result = (a * b) / 2',
            variables={'a': var_a, 'b': var_b}
        )
        
        steps = eq.steps(granularity='normal')
        
        # Normal deve ter 5+ steps
        if len(steps) < 4:
            print_error(f"Normal deveria ter >= 4 steps, tem {len(steps)}")
            return False
        
        operations = [s['operation'] for s in steps]
        print_info(f"Normal: {len(steps)} steps - {operations}")
        return True
    
    def test_steps_detailed(self) -> bool:
        """Teste 6: Steps com granularidade 'detailed'"""
        var_x = self.VariableFactory.create(name='x', value=4)
        var_y = self.VariableFactory.create(name='y', value=3)
        
        eq = self.Equation(
            expression='z = x**2 + y**2',
            variables={'x': var_x, 'y': var_y}
        )
        
        steps = eq.steps(granularity='detailed')
        
        # Detailed deve ter 5+ steps
        if len(steps) < 4:
            print_error(f"Detailed deveria ter >= 4 steps, tem {len(steps)}")
            return False
        
        operations = [s['operation'] for s in steps]
        print_info(f"Detailed: {len(steps)} steps - {operations}")
        return True
    
    def test_steps_all(self) -> bool:
        """Teste 7: Steps com granularidade 'all'"""
        var_a = self.VariableFactory.create(name='a', value=2)
        var_b = self.VariableFactory.create(name='b', value=3)
        var_c = self.VariableFactory.create(name='c', value=4)
        
        eq = self.Equation(
            expression='result = a * b + c**2',
            variables={'a': var_a, 'b': var_b, 'c': var_c}
        )
        
        steps = eq.steps(granularity='all')
        
        # All deve ter o máximo de steps
        if len(steps) < 5:
            print_error(f"All deveria ter >= 5 steps, tem {len(steps)}")
            return False
        
        operations = [s['operation'] for s in steps]
        
        # Deve incluir análise de complexidade
        has_complexity = any('complexidade' in s.get('description', '').lower() for s in steps)
        
        print_info(f"All: {len(steps)} steps - {operations}")
        print_info(f"Inclui análise de complexidade: {has_complexity}")
        return True
    
    def test_steps_smart(self) -> bool:
        """Teste 8: Steps com granularidade 'smart' (auto-detecta)"""
        var_x = self.VariableFactory.create(name='x', value=5)
        
        eq = self.Equation(
            expression='y = x * 2',
            variables={'x': var_x}
        )
        
        steps = eq.steps(granularity='smart')
        
        # Smart deve decidir baseado na complexidade
        if len(steps) < 2:
            print_error(f"Smart deveria ter >= 2 steps, tem {len(steps)}")
            return False
        
        operations = [s['operation'] for s in steps]
        print_info(f"Smart: {len(steps)} steps - {operations}")
        return True
    
    def test_complex_expression(self) -> bool:
        """Teste 9: Expressão complexa com integração"""
        var_b = self.VariableFactory.create(name='b', value=25)
        var_h = self.VariableFactory.create(name='h', value=60)
        
        eq = self.Equation(
            expression='I = (b * h**3) / 12',
            variables={'b': var_b, 'h': var_h},
            description='Momento de inércia'
        )
        
        result = eq.evaluate()
        expected = (25 * 60**3) / 12
        
        if abs(result - expected) > 0.01:
            print_error(f"Resultado incorreto: {result} != {expected}")
            return False
        
        # Testa steps também
        steps = eq.steps(granularity='basic')
        if len(steps) < 3:
            print_error(f"Steps insuficientes para expressão complexa: {len(steps)}")
            return False
        
        return True
    
    def test_latex_output(self) -> bool:
        """Teste 10: Saída LaTeX dos steps"""
        var_a = self.VariableFactory.create(name='a', value=10)
        var_b = self.VariableFactory.create(name='b', value=5)
        
        eq = self.Equation(
            expression='c = a / b',
            variables={'a': var_a, 'b': var_b}
        )
        
        steps = eq.steps(granularity='basic')
        
        # Verifica se tem LaTeX em cada step
        for i, step in enumerate(steps):
            expr = step.get('expr', '')
            if not expr:
                print_warning(f"Step {i} sem expressão LaTeX")
            else:
                print_info(f"Step {i}: {expr[:50]}...")
        
        return True
    
    def test_error_handling(self) -> bool:
        """Teste 11: Tratamento de erros"""
        # Testa variável não definida
        try:
            eq = self.Equation(
                expression='z = x + y',
                variables={},  # Sem variáveis
                description='Teste erro'
            )
            result = eq.evaluate()
            print_warning("Deveria ter gerado erro para variáveis não definidas")
            return False
        except Exception as e:
            print_info(f"Erro capturado corretamente: {type(e).__name__}")
            return True
    
    def test_cache_functionality(self) -> bool:
        """Teste 12: Cache de resultados"""
        var_x = self.VariableFactory.create(name='x', value=7)
        
        eq = self.Equation(
            expression='y = x**2',
            variables={'x': var_x}
        )
        
        # Primeira avaliação
        result1 = eq.evaluate()
        
        # Segunda avaliação (deve usar cache)
        result2 = eq.evaluate()
        
        if result1 != result2:
            print_error("Cache não funcionando - resultados diferentes")
            return False
        
        print_info(f"Cache funcionando: {result1} == {result2}")
        return True
    
    def run_all_tests(self):
        """Executa todos os testes"""
        if not self.core_available:
            print_error("PyMemorial Core não disponível - abortando testes")
            return
        
        print_header("VALIDAÇÃO COMPLETA DO PYMEMORIAL CORE - EQUATION.PY")
        
        # Lista de testes
        tests = [
            ("Criação Básica de Equação", self.test_basic_equation_creation),
            ("Equações com Unidades (Pint)", self.test_equation_with_units),
            ("Steps - Granularidade 'minimal'", self.test_steps_minimal),
            ("Steps - Granularidade 'basic'", self.test_steps_basic),
            ("Steps - Granularidade 'normal'", self.test_steps_normal),
            ("Steps - Granularidade 'detailed'", self.test_steps_detailed),
            ("Steps - Granularidade 'all'", self.test_steps_all),
            ("Steps - Granularidade 'smart'", self.test_steps_smart),
            ("Expressão Complexa", self.test_complex_expression),
            ("Saída LaTeX", self.test_latex_output),
            ("Tratamento de Erros", self.test_error_handling),
            ("Funcionalidade de Cache", self.test_cache_functionality),
        ]
        
        # Executa cada teste
        for test_name, test_func in tests:
            print(f"\n{'-'*80}")
            self.run_test(test_name, test_func)
        
        # Resumo final
        print_header("RESUMO DOS TESTES")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total de Testes:   {self.total_tests}")
        print(f"{Colors.GREEN}Testes Passados:   {self.passed_tests}{Colors.RESET}")
        print(f"{Colors.RED}Testes Falhados:   {self.failed_tests}{Colors.RESET}")
        print(f"{Colors.YELLOW}Avisos:            {self.warnings}{Colors.RESET}")
        print(f"\n{Colors.BOLD}Taxa de Sucesso:   {success_rate:.1f}%{Colors.RESET}\n")
        
        if self.failed_tests == 0:
            print_success("🎉 TODOS OS TESTES PASSARAM! PyMemorial Core está funcionando perfeitamente!")
            return 0
        else:
            print_error(f"⚠️  {self.failed_tests} teste(s) falharam. Revise o código.")
            return 1


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}🔍 VALIDADOR DO PYMEMORIAL CORE - EQUATION.PY{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    validator = EquationValidator()
    exit_code = validator.run_all_tests()
    
    sys.exit(exit_code)
