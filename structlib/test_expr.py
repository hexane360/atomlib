
from io import StringIO
import typing as t
import re
from copy import deepcopy
from dataclasses import replace

import pytest
import numpy

from .expr import Parser, ParseState, NUMERIC_PARSER, NUMERIC_OPS, VECTOR_PARSER, VECTOR_OPS
from .expr import ValueToken, OpToken, GroupOpenToken, GroupCloseToken, WhitespaceToken
from .expr import Expr, GroupExpr, ValueExpr, BinaryExpr, NaryExpr, UnaryExpr
from .expr import interleave, parse_numeric


def iter_from_func(f, *args, **kwargs):
    while True:
        val = f(*args, **kwargs)
        if val is None:
            return
        yield val


def test_tokenize():
    state = ParseState(NUMERIC_PARSER, StringIO("5 + (6 * 3)"))
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
    state = ParseState(NUMERIC_PARSER, StringIO("5 + (foo * 3)"))

    with pytest.raises(ValueError, match="Syntax error at 1:6: Unexpected token 'foo'"):
        tokens = list(iter_from_func(state.next))


def test_parse_binary():
    s = "6*3*9"
    expr = NUMERIC_PARSER.parse(StringIO(s))
    assert expr == BinaryExpr(OpToken('*', 1, (4, 5), NUMERIC_OPS[2]),
        BinaryExpr(OpToken('*', 1, (2, 3), NUMERIC_OPS[2]),
            ValueExpr(ValueToken('6', 1, (1, 2), 6)),
            ValueExpr(ValueToken('3', 1, (3, 4), 3))
        ),
        ValueExpr(ValueToken('9', 1, (5, 6), 9))
    )

    s = "6+3*9"
    expr = NUMERIC_PARSER.parse(StringIO(s))
    assert expr == BinaryExpr(OpToken('+', 1, (2, 3), NUMERIC_OPS[1]),
        ValueExpr(ValueToken('6', 1, (1, 2), 6)),
        BinaryExpr(OpToken('*', 1, (4, 5), NUMERIC_OPS[2]),
            ValueExpr(ValueToken('3', 1, (3, 4), 3)),
            ValueExpr(ValueToken('9', 1, (5, 6), 9))
        ),
    )


def test_parse_nary():
    s = "(6,9+5)*1,2,3"
    expr = VECTOR_PARSER.parse(StringIO(s))
    assert expr == NaryExpr([OpToken(',', 1, (10, 11), VECTOR_OPS[-1]), OpToken(',', 1, (12, 13), VECTOR_OPS[-1])], [
        BinaryExpr(OpToken('*', 1, (8, 9), NUMERIC_OPS[2]),
            GroupExpr(GroupOpenToken('(', 1, (1, 2)),
                NaryExpr([OpToken(',', 1, (3, 4), op=VECTOR_OPS[-1])], [
                    ValueExpr(ValueToken('6', 1, (2, 3), numpy.array(6))),
                    BinaryExpr(OpToken('+', 1, (5, 6), NUMERIC_OPS[1]),
                        ValueExpr(ValueToken('9', 1, (4, 5), numpy.array(9))),
                        ValueExpr(ValueToken('5', 1, (6, 7), numpy.array(5)))
                    )
                ]), GroupCloseToken(')', 1, (7, 8))
            ),
            ValueExpr(ValueToken('1', 1, (9, 10), numpy.array(1)))
        ),
        ValueExpr(ValueToken('2', 1, (11, 12), numpy.array(2))),
        ValueExpr(ValueToken('3', 1, (13, 14), numpy.array(3)))
    ])


def format_precedence(expr: Expr) -> str:
    if isinstance(expr, BinaryExpr):
        return f"({format_precedence(expr.lhs)}{expr.op_token}{format_precedence(expr.rhs)})"
    elif isinstance(expr, NaryExpr):
        inner = "".join(interleave(
            map(format_precedence, expr.args),
            map(str, expr.op_tokens)
        ))
        return f"({inner})"
    elif isinstance(expr, UnaryExpr):
        return f"{expr.op_token}{format_precedence(expr.inner)}"
    elif isinstance(expr, GroupExpr):
        return f"{expr.open}{format_precedence(expr.inner)}{expr.close}"
    elif isinstance(expr, ValueExpr):
        return str(expr.token)
    else:
        return str(expr)


def test_precedence():
    expr = NUMERIC_PARSER.parse(StringIO("5 + 6 * 3 / 2"))
    assert format_precedence(expr) == "(5+((6*3)/2))"

    expr = NUMERIC_PARSER.parse(StringIO("2 * 3 * 4 * 5"))
    assert format_precedence(expr) == "(((2*3)*4)*5)"

    ops = deepcopy(NUMERIC_OPS)
    ops[2].right_assoc = True  # type: ignore
    parser = Parser(ops)
    expr = parser.parse(StringIO("2 * 3 * 4 * 5"))
    assert format_precedence(expr) == "(2*(3*(4*5)))"


def test_nary_precedence():
    ops = [
        *NUMERIC_OPS,
        replace(VECTOR_OPS[-1], precedence=8)
    ]
    parser = Parser(ops, lambda v: numpy.array(parse_numeric(v)))

    expr = parser.parse(StringIO("2 , 3 , 5 * 5 , 8"))
    assert format_precedence(expr) == "((2,3,5)*(5,8))"

    ops.append(replace(VECTOR_OPS[-1], precedence=8))
    with pytest.raises(ValueError, match=re.escape('N-ary operators must have distinct precedences.')):
        parser = Parser(ops, lambda v: numpy.array(parse_numeric(v)))


def test_parse():
    s = "5 + (6 * 3 * 9)"
    expr = NUMERIC_PARSER.parse(StringIO(s))
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