import re
import sympy as sp

"""
LLMquery/prompts/math_utils.py
SymPy-based numeric evaluator with error handling and % parsing.
"""
import re
from sympy import sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def evaluate_math(text: str):
    """
    Extract and safely evaluate any simple math expressions (e.g. '5000 * 0.05 * 1')
    from LLM output. Returns a list of formatted numeric results.
    """
    results = []
    if not text:
        return results

    # Normalize common math text
    text = text.replace("^", "**").replace("×", "*").replace("÷", "/")

    # Replace percentages with decimals: 5% -> 0.05
    text = re.sub(r"(\d+(\.\d+)?)\s*%", lambda m: str(float(m.group(1)) / 100), text)

    # Find math-like patterns (numbers + operators)
    pattern = r"(\d+(\.\d+)?(\s*[\+\-\*/]\s*\d+(\.\d+)?){1,3})"
    matches = re.findall(pattern, text)

    for m in matches:
        expr_str = m[0]
        try:
            expr = parse_expr(expr_str, transformations=(standard_transformations + (implicit_multiplication_application,)))
            value = float(expr.evalf())
            results.append(f"{expr_str} = {value:.2f}")
        except Exception:
            # Skip malformed expressions
            continue

    return results
