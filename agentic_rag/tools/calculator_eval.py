"""AST-only safe evaluation for the calculator tool."""

import ast
import math
import operator


_CALC_ALLOWED_FUNCS: dict[str, object] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_CALC_ALLOWED_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_CALC_BIN_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_CALC_UNARY_OPS: dict[type, object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_calculator_ast(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_calculator_ast(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _CALC_ALLOWED_CONSTANTS:
            return _CALC_ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Unknown identifier: {node.id}")

    if isinstance(node, ast.UnaryOp):
        op_fn = _CALC_UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Expression not allowed: {type(node).__name__}")
        return op_fn(_eval_calculator_ast(node.operand))  # type: ignore[arg-type]

    if isinstance(node, ast.BinOp):
        op_fn = _CALC_BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Expression not allowed: {type(node).__name__}")
        left = _eval_calculator_ast(node.left)
        right = _eval_calculator_ast(node.right)
        return op_fn(left, right)  # type: ignore[arg-type, operator]

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        func = _CALC_ALLOWED_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed")
        args = [_eval_calculator_ast(arg) for arg in node.args]
        return func(*args)  # type: ignore[operator]

    raise ValueError(f"Expression not allowed: {type(node).__name__}")


def evaluate_calculator_expression(expression: str | None) -> str:
    """Parse and evaluate; returns numeric string or ``Error: ...`` (never raises)."""
    try:
        parsed = ast.parse((expression or "").strip(), mode="eval")
        return str(_eval_calculator_ast(parsed))
    except Exception as e:
        return f"Error: {e}"
