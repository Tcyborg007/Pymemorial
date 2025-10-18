"""
Teste de integração completo do MemorialBuilder com Equation.
Exemplo: Cálculo de viga biapoiada.
"""

from pymemorial.core import Equation, VariableFactory

# Simulação do MemorialBuilder (substitua pela implementação real quando estiver pronta)
class MemorialBuilder:
    """Builder simplificado para teste de integração."""
    
    def __init__(self, title: str):
        self.title = title
        self.variables = {}
        self.sections = []
        self.equations = []
        self.steps_config = {
            "mode": "detailed",
            "complexity_threshold": 3
        }
        
    def configure_steps(self, mode: str = "smart", complexity_threshold: int = 3):
        """Configura modo de exibição de passos."""
        self.steps_config["mode"] = mode
        self.steps_config["complexity_threshold"] = complexity_threshold
        print(f"✓ Configurado: mode={mode}, threshold={complexity_threshold}")
    
    def add_variable(self, name: str, value: float, unit: str = None):
        """Adiciona variável ao memorial."""
        self.variables[name] = VariableFactory.create(name, value, unit)
        print(f"✓ Variável: {name} = {value} {unit if unit else ''}")
    
    def add_section(self, title: str):
        """Adiciona seção ao memorial."""
        self.sections.append({"title": title, "equations": []})
        print(f"\n📋 SEÇÃO: {title}")
    
    def add_equation(self, expression: str, description: str = ""):
        """
        Adiciona equação e decide automaticamente se mostra steps.
        
        ✨ NOVIDADE: Resultados são automaticamente adicionados como variáveis
        para uso em equações subsequentes.
        """
        # Extrair nome da variável resultado (lado esquerdo do =)
        if '=' in expression:
            result_name = expression.split('=')[0].strip()
        else:
            result_name = None
        
        # Criar equação com variáveis atuais
        eq = Equation(expression, variables=self.variables, description=description)
        
        # Contar operações
        complexity = self._count_operations(expression)
        
        # Decidir granularidade
        if self.steps_config["mode"] == "minimal":
            granularity = "minimal"
            show_steps = False
        elif self.steps_config["mode"] == "detailed":
            granularity = "detailed"
            show_steps = True
        else:  # smart
            if complexity > self.steps_config["complexity_threshold"]:
                granularity = "detailed"
                show_steps = True
            else:
                granularity = "minimal"
                show_steps = False
        
        # Gerar steps
        steps = eq.steps(granularity=granularity)
        
        # ✅ SOLUÇÃO: Armazenar resultado como nova variável
        if result_name and steps and steps[-1]['numeric'] is not None:
            result_value = steps[-1]['numeric']
            # Criar variável com o resultado para uso futuro
            self.variables[result_name] = VariableFactory.create(result_name, result_value)
            # Também atualizar a equação com a nova variável
            eq.variables = self.variables.copy()
        
        # Mostrar resultado
        print(f"\n  Equação: {expression}")
        print(f"  Complexidade: {complexity} operações")
        print(f"  Granularidade: {granularity}")
        
        if show_steps and len(steps) > 2:
            print(f"  📊 PASSOS DE CÁLCULO:")
            for step in steps:
                if step["numeric"] is not None:
                    print(f"    {step['step_number']}. {step['description']}: {step['latex']} = {step['numeric']:.4f}")
                else:
                    print(f"    {step['step_number']}. {step['description']}: {step['latex']}")
        else:
            if steps and steps[-1]['numeric'] is not None:
                print(f"  ✓ Resultado direto: {steps[-1]['numeric']:.4f}")
                if result_name:
                    print(f"  ✓ {result_name} armazenado para uso futuro")
        
        self.equations.append({
            "equation": eq,
            "steps": steps,
            "show_steps": show_steps
        })
    
    def _count_operations(self, expression: str) -> int:
        """Conta operações matemáticas na expressão."""
        ops = ['+', '-', '*', '/', '**']
        count = sum(expression.count(op) for op in ops)
        count += expression.count('(')
        return count
    
    def export_summary(self):
        """Exporta resumo do memorial."""
        print("\n" + "="*70)
        print(f"📄 MEMORIAL: {self.title}")
        print("="*70)
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"  - Total de variáveis: {len(self.variables)}")
        print(f"  - Total de equações: {len(self.equations)}")
        print(f"  - Equações com steps: {sum(1 for eq in self.equations if eq['show_steps'])}")
        print(f"  - Modo configurado: {self.steps_config['mode']}")
        
        print(f"\n📝 VARIÁVEIS FINAIS:")
        for name, var in self.variables.items():
            if hasattr(var, 'value') and var.value is not None:
                unit_str = f" {var.unit}" if hasattr(var, 'unit') and var.unit else ""
                print(f"  - {name} = {var.value}{unit_str}")
        
        print("\n" + "="*70)


# ============================================================================
# TESTE 1: Modo SMART (padrão inteligente)
# ============================================================================

def test_smart_mode():
    """Testa modo smart que decide automaticamente quando mostrar steps."""
    print("\n" + "🔷" * 35)
    print("TESTE 1: MODO SMART")
    print("🔷" * 35)
    
    memorial = MemorialBuilder("Viga Biapoiada - Modo Smart")
    memorial.configure_steps(mode="smart", complexity_threshold=3)
    
    # Adicionar variáveis
    memorial.add_variable("L", 6.0, "m")
    memorial.add_variable("q", 15.0, "kN/m")
    memorial.add_variable("b", 0.2, "m")
    memorial.add_variable("h", 0.5, "m")
    
    memorial.add_section("Geometria")
    
    # Equação SIMPLES (2 operações) → SEM steps
    memorial.add_equation("A = b * h", "Área da seção")
    
    # Equação MÉDIA (3 operações) → SEM steps (threshold=3)
    memorial.add_equation("I = b * h**3 / 12", "Momento de inércia")
    
    memorial.add_section("Esforços")
    
    # Equação COMPLEXA (4+ operações) → COM steps
    memorial.add_equation("M = q * L**2 / 8", "Momento fletor máximo")
    
    memorial.add_section("Tensões")
    
    # Equação MUITO COMPLEXA (6+ operações) → COM steps detalhados
    memorial.add_equation("sigma = M * (h/2) / I", "Tensão máxima")
    
    memorial.export_summary()


# ============================================================================
# TESTE 2: Modo DETAILED (sempre mostra steps)
# ============================================================================

def test_detailed_mode():
    """Testa modo detailed que sempre mostra todos os passos."""
    print("\n" + "🔶" * 35)
    print("TESTE 2: MODO DETAILED")
    print("🔶" * 35)
    
    memorial = MemorialBuilder("Pilar - Modo Detailed")
    memorial.configure_steps(mode="detailed")
    
    memorial.add_variable("N", 1000.0, "kN")
    memorial.add_variable("A", 0.04, "m²")
    memorial.add_variable("fck", 30.0, "MPa")
    
    memorial.add_section("Verificação")
    
    # Mesmo equações simples mostram steps
    memorial.add_equation("sigma_c = N / A", "Tensão de compressão")
    memorial.add_equation("taxa = sigma_c / fck", "Taxa de utilização")
    
    memorial.export_summary()


# ============================================================================
# TESTE 3: Modo MINIMAL (nunca mostra steps)
# ============================================================================

def test_minimal_mode():
    """Testa modo minimal que nunca mostra passos intermediários."""
    print("\n" + "🔷" * 35)
    print("TESTE 3: MODO MINIMAL")
    print("🔷" * 35)
    
    memorial = MemorialBuilder("Laje - Modo Minimal")
    memorial.configure_steps(mode="minimal")
    
    memorial.add_variable("lx", 4.0, "m")
    memorial.add_variable("ly", 5.0, "m")
    memorial.add_variable("q", 12.0, "kN/m²")
    
    memorial.add_section("Momentos")
    
    # Mesmo equações complexas não mostram steps
    memorial.add_equation("lambda_val = ly / lx", "Relação de lados")
    memorial.add_equation("mx = q * lx**2 / 8", "Momento na direção x")
    memorial.add_equation("my = mx * (lambda_val**2)", "Momento na direção y")
    
    memorial.export_summary()


# ============================================================================
# TESTE 4: Comparação de Modos
# ============================================================================

def test_comparison():
    """Compara os três modos lado a lado."""
    print("\n" + "🔶" * 35)
    print("TESTE 4: COMPARAÇÃO DE MODOS")
    print("🔶" * 35)
    
    # Mesma equação, modos diferentes
    expression = "resultado = (a**2 + b**2)**0.5 / c"
    
    for mode in ["minimal", "smart", "detailed"]:
        print(f"\n--- Modo: {mode.upper()} ---")
        
        memorial = MemorialBuilder(f"Teste {mode}")
        memorial.configure_steps(mode=mode, complexity_threshold=4)
        
        memorial.add_variable("a", 3.0)
        memorial.add_variable("b", 4.0)
        memorial.add_variable("c", 5.0)
        
        memorial.add_section("Cálculo")
        memorial.add_equation(expression)


# ============================================================================
# EXECUTAR TODOS OS TESTES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 TESTE DE INTEGRAÇÃO: MemorialBuilder + Equation")
    print("="*70)
    
    test_smart_mode()
    test_detailed_mode()
    test_minimal_mode()
    test_comparison()
    
    print("\n" + "="*70)
    print("✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
    print("="*70)
