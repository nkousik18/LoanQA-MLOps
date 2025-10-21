import re
import sympy as sp

def evaluate_math(text):
    """
    Extract and evaluate valid math expressions from text.
    Ignores standalone numbers (e.g., 'Step 1') and computes only
    expressions containing at least one operator.
    """
    results = []
    matches = re.findall(r'([0-9\.\+\-\*/\(\)\s]+[+\-\*/][0-9\.\+\-\*/\(\)\s]+)', text)
    for expr in matches:
        expr = expr.strip()
        try:
            value = sp.sympify(expr).evalf()
            results.append(f"{expr} = {float(value):.2f}")
        except Exception:
            continue
    return results
