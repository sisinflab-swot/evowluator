import ast
import operator as op
from typing import Dict

_UNARY_OPS: Dict = {
    ast.USub: op.neg
}

_BINARY_OPS: Dict = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow
}


def evaluate_expression(expression: str) -> int | float:
    try:
        return _eval(ast.parse(expression, mode='eval').body)
    except TypeError:
        raise ValueError(f'Not a valid arithmetic expression: \"{expression}\"')


def _eval(node: ast.expr) -> int | float:
    if isinstance(node, ast.Num) and isinstance(node.n, (int, float)):
        return node.n
    elif isinstance(node, ast.BinOp):
        return _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    else:
        raise TypeError(node)
