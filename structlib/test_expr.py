
from io import StringIO
import typing as t
import operator
from copy import deepcopy

import pytest

from .expr import BinaryOp, BinaryOrUnaryOp, Parser, ParseState, NUMERIC_PARSER, NUMERIC_OPS
from .expr import Token, ValueToken, OpToken, GroupOpenToken, GroupCloseToken, WhitespaceToken
from .expr import Expr, GroupExpr, ValueExpr, BinaryExpr, UnaryExpr


def iter_from_func(f, *args, **kwargs):
    while True:
        val = f(*args, **kwargs)
        if val is None:
            return
        yield val


def test_tokenize():
    state = ParseState(NUMERIC_PARSER, "5 + (6 * 3)")
    tokens = list(iter_from_func(state.next))
    
    assert tokens == [
        ValueToken('5', 1, (1, 2), 5),
        WhitespaceToken(' ', 1, (2, 3)),
        OpToken('+', 1, (3, 4), NUMERIC_OPS[1]),
        WhitespaceToken(' ', 1, (4, 5)),
        GroupOpenToken('(', 1, (5, 6)),
        ValueToken('6', 1, (6, 7), 6),
        WhitespaceToken(' ', 1, (7, 8)),
        OpToken('*', 1, (8, 9), NUMERIC_OPS[2]),
        WhitespaceToken(' ', 1, (9, 10)),
        ValueToken('3', 1, (10, 11), 3),
        GroupCloseToken(')', 1, (11, 12))
    ]


def test_tokenize_error():
    state = ParseState(NUMERIC_PARSER, "5 + (foo * 3)")

    with pytest.raises(ValueError, match="Syntax error at 1:6-9: Unexpected token 'foo'"):
        tokens = list(iter_from_func(state.next))


def test_parse_binary():
    s = "6*3*9"
    expr = NUMERIC_PARSER.parse(s)
    assert expr == BinaryExpr(OpToken('*', 1, (4, 5), NUMERIC_OPS[2]),
        BinaryExpr(OpToken('*', 1, (2, 3), NUMERIC_OPS[2]),
            ValueExpr(ValueToken('6', 1, (1, 2), 6)),
            ValueExpr(ValueToken('3', 1, (3, 4), 3))
        ),
        ValueExpr(ValueToken('9', 1, (5, 6), 9))
    )

    s = "6+3*9"
    expr = NUMERIC_PARSER.parse(s)
    assert expr == BinaryExpr(OpToken('+', 1, (2, 3), NUMERIC_OPS[1]),
        ValueExpr(ValueToken('6', 1, (1, 2), 6)),
        BinaryExpr(OpToken('*', 1, (4, 5), NUMERIC_OPS[2]),
            ValueExpr(ValueToken('3', 1, (3, 4), 3)),
            ValueExpr(ValueToken('9', 1, (5, 6), 9))
        ),
    )


def format_precedence(expr: Expr) -> str:
    if isinstance(expr, BinaryExpr):
        return f"({format_precedence(expr.lhs)}{expr.op_token}{format_precedence(expr.rhs)})"
    elif isinstance(expr, UnaryExpr):
        return f"{expr.op_token}{format_precedence(expr.inner)}"
    elif isinstance(expr, GroupExpr):
        return f"{expr.open}{format_precedence(expr.inner)}{expr.close}"
    elif isinstance(expr, ValueExpr):
        return str(expr.token)
    else:
        return str(expr)


def test_precedence():
    expr = NUMERIC_PARSER.parse("5 + 6 * 3 / 2")
    assert format_precedence(expr) == "(5+((6*3)/2))"

    expr = NUMERIC_PARSER.parse("2 * 3 * 4 * 5")
    assert format_precedence(expr) == "(((2*3)*4)*5)"

    ops = deepcopy(NUMERIC_OPS)
    ops[2].right_assoc = True
    parser = Parser(ops)
    expr = parser.parse("2 * 3 * 4 * 5")
    assert format_precedence(expr) == "(2*(3*(4*5)))"

def test_parse():
    s = "5 + (6 * 3 * 9)"
    expr = NUMERIC_PARSER.parse(s)
    print(f"{expr!s}")

    assert expr == BinaryExpr(OpToken('+', 1, (3, 4), NUMERIC_OPS[1]),
        ValueExpr(ValueToken('5', 1, (1, 2), 5), rspace=" "),
        GroupExpr(
            GroupOpenToken('(', 1, (5, 6)),
            BinaryExpr(OpToken('*', 1, (12, 13), NUMERIC_OPS[2]),
                BinaryExpr(OpToken('*', 1, (8, 9), NUMERIC_OPS[2]),
                    ValueExpr(ValueToken('6', 1, (6, 7), 6), rspace=" "),
                    ValueExpr(ValueToken('3', 1, (10, 11), 3), lspace=" ", rspace=" ")
                ),
                ValueExpr(ValueToken('9', 1, (14, 15), 9), lspace=" ")
            ),
            GroupCloseToken(')', 1, (15, 16)),
            lspace=" "
        )
    )

    assert expr.format() == s