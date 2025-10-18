"""Representação de equações simbólicas e numéricas."""

from typing import Dict, Optional, Union, Any
from dataclasses import dataclass, field
import sympy as sp

from .variable import Variable


@dataclass
class Equation:
    """
    Equação de memorial de cálculo.
    
    Attributes:
        expression: expressão SymPy ou string
        variables: dicionário de variáveis
        result: resultado numérico (após avaliação)
        description: descrição da equação
    """
    expression: Union[sp.Expr, str]
    variables: Dict[str, Variable] = field(default_factory=dict)
    result: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        """Converte string para expressão SymPy após inicialização."""
        if isinstance(self.expression, str):
            # Remover parte esquerda da equação se houver (K = ...)
            expr_str = self.expression
            if '=' in expr_str:
                expr_str = expr_str.split('=', 1)[1].strip()
            
            # Converter para SymPy, usando símbolos das variáveis
            local_dict = {name: var.symbol for name, var in self.variables.items()}
            
            try:
                # ✅ evaluate=False para manter a estrutura da expressão
                self.expression = sp.sympify(expr_str, locals=local_dict, evaluate=False)
            except (sp.SympifyError, ValueError, TypeError) as e:
                raise ValueError(f"Erro ao converter expressão '{expr_str}': {e}")

    def substitute(self, **kwargs) -> sp.Expr:
        """
        Substitui variáveis na expressão.
        
        Args:
            **kwargs: pares nome=valor
            
        Returns:
            Expressão com substituições
            
        Raises:
            KeyError: Se variável não for encontrada
        """
        subs_dict = {}
        
        # Tentar mapear nomes para símbolos registrados primeiro
        for name, value in kwargs.items():
            if name in self.variables:
                # Variável registrada - usar seu símbolo
                var = self.variables[name]
                subs_dict[var.symbol] = value
            else:
                # Variável não registrada - tentar encontrar símbolo na expressão
                # Isso permite usar substitute() mesmo sem variables definidas
                symbols_in_expr = {str(s): s for s in self.expression.free_symbols}
                
                if name in symbols_in_expr:
                    # Símbolo existe na expressão
                    subs_dict[symbols_in_expr[name]] = value
                else:
                    # Símbolo não existe - avisar usuário
                    available = list(self.variables.keys()) if self.variables else list(symbols_in_expr.keys())
                    raise KeyError(
                        f"Variável '{name}' não encontrada. "
                        f"Disponíveis: {available}"
                    )
        
        try:
            return self.expression.subs(subs_dict)
        except Exception as e:
            raise ValueError(f"Erro ao substituir variáveis: {e}")


    def evaluate(self) -> float:
        """
        Avalia numericamente usando valores das variáveis.
        
        Returns:
            Resultado numérico
            
        Raises:
            ValueError: Se variável não tem valor ou avaliação falha
        """
        # Verificar se todas as variáveis usadas têm valores
        used_symbols = self.expression.free_symbols
        missing_vars = []
        
        subs = {}
        for var_name, var in self.variables.items():
            if var.symbol in used_symbols:
                if var.value is None:
                    missing_vars.append(var_name)
                else:
                    # Extrair magnitude se for Quantity do Pint
                    if hasattr(var.value, 'magnitude'):
                        subs[var.symbol] = var.value.magnitude
                    else:
                        subs[var.symbol] = float(var.value)
        
        if missing_vars:
            raise ValueError(f"Variáveis sem valor: {missing_vars}")
        
        try:
            expr_sub = self.expression.subs(subs)
            result = float(expr_sub.evalf())
            self.result = result
            return result
        except (TypeError, ValueError, ZeroDivisionError) as e:
            raise ValueError(f"Erro ao avaliar expressão: {e}")

    def simplify(self) -> sp.Expr:
        """
        Retorna expressão simplificada.
        
        Returns:
            Expressão simplificada (SymPy Expr)
        """
        try:
            simplified = sp.simplify(self.expression, rational=True)
            return simplified
        except Exception as e:
            # Se simplificação falhar, retornar expressão original
            return self.expression

    def latex(self) -> str:
        """
        Retorna representação LaTeX da expressão.
        
        Returns:
            String LaTeX
        """
        try:
            return sp.latex(self.expression)
        except Exception as e:
            # Fallback: retornar string simples
            return str(self.expression)

    def steps(
        self,
        granularity: str = "detailed",
        show_units: bool = True,
        max_steps: int = 20
    ) -> list[dict]:
        """
        Gera passos intermediários com granularidade controlada.
        
        Args:
            granularity: nível de detalhe
                - "minimal": apenas expressão simbólica e resultado
                - "normal": substituição + simplificação final
                - "detailed": cada operação separada (recomendado)
                - "all": todos os micro-passos possíveis
            show_units: incluir unidades em cada passo
            max_steps: limite de passos (proteção contra loops)
        
        Returns:
            Lista de dicionários com:
                - step_number: número do passo
                - expression: expressão SymPy
                - latex: representação LaTeX
                - numeric: valor numérico (se totalmente resolvido)
                - description: descrição do passo
                - operation: tipo de operação
        
        Raises:
            ValueError: Se granularidade inválida
        """
        valid_granularities = ["minimal", "normal", "detailed", "all"]
        if granularity not in valid_granularities:
            raise ValueError(f"Granularidade '{granularity}' inválida. Use: {valid_granularities}")
        
        steps = []
        step_num = 1
        
        # Passo 1: Expressão simbólica original
        steps.append({
            "step_number": step_num,
            "expression": self.expression,
            "latex": sp.latex(self.expression),
            "numeric": None,
            "description": "Expressão simbólica",
            "operation": "symbolic"
        })
        step_num += 1
        
        if granularity == "minimal":
            result = self.evaluate()
            steps.append({
                "step_number": step_num,
                "expression": sp.Number(result),
                "latex": self._format_result(result, show_units),
                "numeric": result,
                "description": "Resultado final",
                "operation": "result"
            })
            return steps
        
        # Passo 2: Substituir variáveis por valores SEM simplificar automaticamente
        substitutions = self._build_substitutions()
        
        def replace_no_eval(expr):
            """Substitui recursivamente sem avaliar."""
            if expr.is_Symbol and expr in substitutions:
                return substitutions[expr]
            elif expr.is_Atom:
                return expr
            else:
                try:
                    new_args = [replace_no_eval(arg) for arg in expr.args]
                    return expr.func(*new_args, evaluate=False)
                except Exception:
                    return expr
        
        expr_with_values = replace_no_eval(self.expression)
        
        steps.append({
            "step_number": step_num,
            "expression": expr_with_values,
            "latex": sp.latex(expr_with_values),
            "numeric": None,
            "description": "Substituição de valores",
            "operation": "substitution"
        })
        step_num += 1
        
        if granularity == "normal":
            try:
                simplified = sp.simplify(expr_with_values, rational=True)
                final_value = float(simplified.evalf())
                
                if simplified != expr_with_values:
                    steps.append({
                        "step_number": step_num,
                        "expression": simplified,
                        "latex": self._format_result(final_value, show_units),
                        "numeric": final_value,
                        "description": "Resultado final",
                        "operation": "simplify"
                    })
                else:
                    steps.append({
                        "step_number": step_num,
                        "expression": sp.Number(final_value),
                        "latex": self._format_result(final_value, show_units),
                        "numeric": final_value,
                        "description": "Resultado final",
                        "operation": "result"
                    })
            except Exception as e:
                # Fallback: avaliar diretamente
                final_value = float(expr_with_values.evalf())
                steps.append({
                    "step_number": step_num,
                    "expression": sp.Number(final_value),
                    "latex": self._format_result(final_value, show_units),
                    "numeric": final_value,
                    "description": "Resultado final",
                    "operation": "result"
                })
            return steps
        
        # Granularidade "detailed" ou "all"
        current = expr_with_values
        iteration = 0
        
        # Flag para garantir que sempre temos um resultado final
        reached_final = False
        
        while iteration < max_steps:
            try:
                next_expr, operation = self._simplify_one_step(current, granularity)
                
                # Se não houve mudança, terminamos a simplificação
                if next_expr == current or operation == "done":
                    # Se current já é um número, marcar como finalizado
                    if current.is_Number:
                        reached_final = True
                    break
                
                iteration += 1
                description = self._operation_description(operation)
                
                # ✅ usar is_Number (maiúsculo)
                is_numeric = next_expr.is_Number
                numeric_value = float(next_expr.evalf()) if is_numeric else None
                
                steps.append({
                    "step_number": step_num,
                    "expression": next_expr,
                    "latex": sp.latex(next_expr),
                    "numeric": numeric_value,
                    "description": description,
                    "operation": operation
                })
                step_num += 1
                current = next_expr
                
                if is_numeric:
                    reached_final = True
                    break
                    
            except Exception as e:
                # Se ocorrer erro, parar iteração
                break
        
        # ✅ GARANTIR que último passo sempre tem valor numérico e operation="result"
        if steps:
            # Se o último passo não é numérico, avaliar
            if steps[-1]['numeric'] is None:
                try:
                    final_value = float(steps[-1]['expression'].evalf())
                    steps[-1]['numeric'] = final_value
                    steps[-1]['latex'] = self._format_result(final_value, show_units)
                except Exception:
                    pass
            
            # ✅ GARANTIR que o último passo sempre tem description="Resultado final"
            if steps[-1]['description'] != "Resultado final":
                steps[-1]['description'] = "Resultado final"
            
            # ✅ GARANTIR que o último passo sempre tem operation="result"
            # APENAS se tiver valor numérico
            if steps[-1]['numeric'] is not None and steps[-1]['operation'] != "result":
                # Se o último passo é "substitution" mas já tem valor numérico,
                # adicionar um passo final explícito
                if steps[-1]['operation'] == "substitution" and steps[-1]['numeric'] is not None:
                    # Já é o resultado final, apenas atualizar operation
                    steps[-1]['operation'] = "result"
        
        # Garantir unidades no último passo
        if steps and show_units and steps[-1]['numeric'] is not None:
            steps[-1]['latex'] = self._format_result(
                steps[-1]['numeric'],
                show_units=True
            )
        
        return steps


    def _build_substitutions(self) -> dict:
        """
        Constrói dicionário de substituições variável → valor.
        
        Returns:
            Dicionário {símbolo: valor}
        """
        substitutions = {}
        for name, var in self.variables.items():
            if var.value is not None:
                try:
                    if hasattr(var.value, 'magnitude'):
                        value = var.value.magnitude
                    else:
                        value = float(var.value)
                    
                    substitutions[var.symbol] = sp.Float(value)
                except (TypeError, ValueError):
                    # Pular variável se conversão falhar
                    continue
        
        return substitutions

    def _simplify_one_step(self, expr, granularity: str) -> tuple:
        """
        Simplifica uma operação matemática por vez.
        
        Args:
            expr: expressão SymPy
            granularity: nível de detalhe
        
        Returns:
            (expressão_simplificada, tipo_operação)
        """
        try:
            # Se já é número, terminou
            if expr.is_Number:
                return expr, "done"
            
            # 1. EXPONENCIAÇÃO - se a expressão raiz é Pow
            if isinstance(expr, sp.Pow):
                if expr.base.is_Number and expr.exp.is_Number:
                    result = sp.Number(float(expr.base) ** float(expr.exp))
                    return result, "power"
                
                if isinstance(expr.base, sp.Add):
                    for i, arg in enumerate(expr.base.args):
                        if isinstance(arg, sp.Pow) and arg.base.is_Number and arg.exp.is_Number:
                            pow_result = sp.Number(float(arg.base) ** float(arg.exp))
                            new_add_args = list(expr.base.args)
                            new_add_args[i] = pow_result
                            new_add = sp.Add(*new_add_args, evaluate=False)
                            new_expr = sp.Pow(new_add, expr.exp, evaluate=False)
                            return new_expr, "power"
                    
                    if all(arg.is_Number for arg in expr.base.args):
                        sum_val = sum(float(arg) for arg in expr.base.args)
                        new_base = sp.Number(sum_val)
                        new_expr = sp.Pow(new_base, expr.exp, evaluate=False)
                        return new_expr, "add"
                
                if isinstance(expr.base, sp.Mul):
                    for i, mul_arg in enumerate(expr.base.args):
                        if isinstance(mul_arg, sp.Pow) and mul_arg.base.is_Number and mul_arg.exp.is_Number:
                            pow_result = sp.Number(float(mul_arg.base) ** float(mul_arg.exp))
                            new_mul_args = list(expr.base.args)
                            new_mul_args[i] = pow_result
                            new_mul = sp.Mul(*new_mul_args, evaluate=False)
                            new_expr = sp.Pow(new_mul, expr.exp, evaluate=False)
                            return new_expr, "power"
                    
                    if all(arg.is_Number for arg in expr.base.args):
                        prod_val = 1.0
                        for arg in expr.base.args:
                            prod_val *= float(arg)
                        new_base = sp.Number(prod_val)
                        new_expr = sp.Pow(new_base, expr.exp, evaluate=False)
                        return new_expr, "multiply"
            
            # 2. MULTIPLICAÇÃO - se a expressão raiz é Mul
            if isinstance(expr, sp.Mul):
                for i, arg in enumerate(expr.args):
                    if isinstance(arg, sp.Pow):
                        if arg.base.is_Number and arg.exp.is_Number:
                            pow_result = sp.Number(float(arg.base) ** float(arg.exp))
                            new_args = list(expr.args)
                            new_args[i] = pow_result
                            new_expr = sp.Mul(*new_args, evaluate=False)
                            return new_expr, "power"
                        
                        if isinstance(arg.base, sp.Add):
                            for j, add_arg in enumerate(arg.base.args):
                                if isinstance(add_arg, sp.Pow) and add_arg.base.is_Number and add_arg.exp.is_Number:
                                    pow_result = sp.Number(float(add_arg.base) ** float(add_arg.exp))
                                    new_add_args = list(arg.base.args)
                                    new_add_args[j] = pow_result
                                    new_add = sp.Add(*new_add_args, evaluate=False)
                                    new_pow = sp.Pow(new_add, arg.exp, evaluate=False)
                                    new_mul_args = list(expr.args)
                                    new_mul_args[i] = new_pow
                                    new_expr = sp.Mul(*new_mul_args, evaluate=False)
                                    return new_expr, "power"
                            
                            if all(a.is_Number for a in arg.base.args):
                                sum_val = sum(float(a) for a in arg.base.args)
                                new_base = sp.Number(sum_val)
                                new_pow = sp.Pow(new_base, arg.exp, evaluate=False)
                                new_mul_args = list(expr.args)
                                new_mul_args[i] = new_pow
                                new_expr = sp.Mul(*new_mul_args, evaluate=False)
                                return new_expr, "add"
                        
                        if isinstance(arg.base, sp.Mul):
                            for j, mul_arg in enumerate(arg.base.args):
                                if isinstance(mul_arg, sp.Pow) and mul_arg.base.is_Number and mul_arg.exp.is_Number:
                                    pow_result = sp.Number(float(mul_arg.base) ** float(mul_arg.exp))
                                    new_mul_args = list(arg.base.args)
                                    new_mul_args[j] = pow_result
                                    new_mul = sp.Mul(*new_mul_args, evaluate=False)
                                    new_pow = sp.Pow(new_mul, arg.exp, evaluate=False)
                                    new_expr_args = list(expr.args)
                                    new_expr_args[i] = new_pow
                                    new_expr = sp.Mul(*new_expr_args, evaluate=False)
                                    return new_expr, "power"
                            
                            if all(a.is_Number for a in arg.base.args):
                                prod_val = 1.0
                                for a in arg.base.args:
                                    prod_val *= float(a)
                                new_base = sp.Number(prod_val)
                                new_pow = sp.Pow(new_base, arg.exp, evaluate=False)
                                new_expr_args = list(expr.args)
                                new_expr_args[i] = new_pow
                                new_expr = sp.Mul(*new_expr_args, evaluate=False)
                                return new_expr, "multiply"
                
                nums = [a for a in expr.args if a.is_Number]
                others = [a for a in expr.args if not a.is_Number]
                
                if len(nums) >= 2:
                    product = sp.Number(float(nums[0]) * float(nums[1]))
                    new_args = [product] + nums[2:] + others
                    new_expr = sp.Mul(*new_args, evaluate=False) if len(new_args) > 1 else new_args[0]
                    return new_expr, "multiply"
            
            # 3. ADIÇÃO - se a expressão raiz é Add
            if isinstance(expr, sp.Add):
                for i, arg in enumerate(expr.args):
                    if isinstance(arg, sp.Pow) and arg.base.is_Number and arg.exp.is_Number:
                        pow_result = sp.Number(float(arg.base) ** float(arg.exp))
                        new_args = list(expr.args)
                        new_args[i] = pow_result
                        new_expr = sp.Add(*new_args, evaluate=False)
                        return new_expr, "power"
                    
                    if isinstance(arg, sp.Mul):
                        for j, mul_arg in enumerate(arg.args):
                            if isinstance(mul_arg, sp.Pow) and mul_arg.exp.is_Number and mul_arg.base.is_Number:
                                pow_result = sp.Number(float(mul_arg.base) ** float(mul_arg.exp))
                                new_mul_args = list(arg.args)
                                new_mul_args[j] = pow_result
                                new_mul = sp.Mul(*new_mul_args, evaluate=False)
                                new_add_args = list(expr.args)
                                new_add_args[i] = new_mul
                                new_expr = sp.Add(*new_add_args, evaluate=False)
                                return new_expr, "power"
                        
                        nums = [a for a in arg.args if a.is_Number]
                        others = [a for a in arg.args if not a.is_Number]
                        
                        if len(nums) >= 2:
                            product = sp.Number(float(nums[0]) * float(nums[1]))
                            new_mul_args = [product] + nums[2:] + others
                            new_mul = sp.Mul(*new_mul_args, evaluate=False) if len(new_mul_args) > 1 else new_mul_args[0]
                            new_add_args = list(expr.args)
                            new_add_args[i] = new_mul
                            new_expr = sp.Add(*new_add_args, evaluate=False)
                            return new_expr, "multiply"
                
                nums = [a for a in expr.args if a.is_Number]
                others = [a for a in expr.args if not a.is_Number]
                
                if len(nums) >= 2:
                    sum_val = sp.Number(float(nums[0]) + float(nums[1]))
                    new_args = [sum_val] + nums[2:] + others
                    new_expr = sp.Add(*new_args, evaluate=False) if len(new_args) > 1 else new_args[0]
                    return new_expr, "add"
            
            return expr, "done"
            
        except Exception:
            # Se qualquer erro ocorrer, retornar expressão sem mudanças
            return expr, "done"

    def _operation_description(self, operation: str) -> str:
        """Gera descrição em português da operação."""
        descriptions = {
            "power": "Cálculo de potência",
            "multiply": "Multiplicação",
            "divide": "Divisão",
            "add": "Adição/subtração",
            "simplify": "Simplificação algébrica",
            "substitution": "Substituição de valores",
            "result": "Resultado final",
            "symbolic": "Expressão simbólica",
            "done": "Expressão simplificada"
        }
        
        return descriptions.get(operation, "Passo de cálculo")

    def _format_result(self, value: float, show_units: bool) -> str:
        """Formata resultado final com unidades."""
        if not show_units:
            return sp.latex(sp.Number(value))
        
        # Por enquanto, retornar valor numérico
        # TODO: Adicionar suporte a unidades na Fase 5
        return sp.latex(sp.Number(value))
    
    def __repr__(self) -> str:
        """Representação da equação."""
        vars_info = f"{len(self.variables)} vars" if self.variables else "no vars"
        result_info = f", result={self.result}" if self.result is not None else ""
        return f"Equation({self.expression}, {vars_info}{result_info})"
