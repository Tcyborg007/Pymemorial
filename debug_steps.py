"""Debug passo a passo do _simplify_one_step."""
import sympy as sp

# Simular a expressão: 2.0 + 4.0*2.0*0.5
expr = sp.Add(
    sp.Float(2.0),
    sp.Mul(sp.Float(4.0), sp.Float(2.0), sp.Float(0.5), evaluate=False),
    evaluate=False
)

print("=" * 70)
print("DEBUG: _simplify_one_step")
print("=" * 70)
print(f"\nExpressão inicial: {expr}")
print(f"Repr: {sp.srepr(expr)}")

iteration = 0
current = expr

while iteration < 10:
    print(f"\n--- Iteração {iteration + 1} ---")
    print(f"Current: {current}")
    print(f"is_Number: {current.is_Number}")
    
    if current.is_Number:
        print("✓ É número, terminou")
        break
    
    # Verificar se é Mul
    if isinstance(current, sp.Mul):
        print("✓ É Mul")
        nums = [a for a in current.args if a.is_Number and not isinstance(a, sp.Pow)]
        print(f"  Números: {nums}")
        
        if len(nums) >= 2:
            product = sp.Number(float(nums[0]) * float(nums[1]))
            print(f"  Produto: {nums[0]} * {nums[1]} = {product}")
            
            others = [a for a in current.args if not (a.is_Number and not isinstance(a, sp.Pow))]
            new_args = [product] + nums[2:] + others
            print(f"  New args: {new_args}")
            
            new_expr = sp.Mul(*new_args, evaluate=False) if len(new_args) > 1 else new_args[0]
            print(f"  New expr: {new_expr}")
            print(f"  New repr: {sp.srepr(new_expr)}")
            
            current = new_expr
            iteration += 1
            continue
    
    # Verificar se é Add e tem args que são Mul
    if isinstance(current, sp.Add):
        print("✓ É Add")
        print(f"  Args: {current.args}")
        
        # Processar cada arg
        modified = False
        for i, arg in enumerate(current.args):
            if isinstance(arg, sp.Mul):
                print(f"\n  Arg {i} é Mul: {arg}")
                nums = [a for a in arg.args if a.is_Number and not isinstance(a, sp.Pow)]
                print(f"    Números no Mul: {nums}")
                
                if len(nums) >= 2:
                    product = sp.Number(float(nums[0]) * float(nums[1]))
                    print(f"    Produto: {nums[0]} * {nums[1]} = {product}")
                    
                    others = [a for a in arg.args if not (a.is_Number and not isinstance(a, sp.Pow))]
                    new_mul_args = [product] + nums[2:] + others
                    new_mul = sp.Mul(*new_mul_args, evaluate=False) if len(new_mul_args) > 1 else new_mul_args[0]
                    
                    print(f"    Novo Mul: {new_mul}")
                    
                    # Reconstruir o Add
                    new_add_args = list(current.args)
                    new_add_args[i] = new_mul
                    current = sp.Add(*new_add_args, evaluate=False)
                    
                    print(f"    Novo Add: {current}")
                    modified = True
                    break
        
        if modified:
            iteration += 1
            continue
        
        # Se não modificou, tentar somar números
        nums = [a for a in current.args if a.is_Number]
        print(f"  Números no Add: {nums}")
        
        if len(nums) >= 2:
            sum_val = sp.Number(float(nums[0]) + float(nums[1]))
            print(f"  Soma: {nums[0]} + {nums[1]} = {sum_val}")
            
            others = [a for a in current.args if not a.is_Number]
            new_args = [sum_val] + nums[2:] + others
            current = sp.Add(*new_args, evaluate=False) if len(new_args) > 1 else new_args[0]
            
            print(f"  Novo Add: {current}")
            iteration += 1
            continue
    
    print("✗ Não conseguiu simplificar")
    break

print("\n" + "=" * 70)
print(f"RESULTADO FINAL: {current}")
print("=" * 70)