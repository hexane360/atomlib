from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import TextIOBase
from itertools import zip_longest
import logging
import operator
import re
import typing as t

import numpy

V = t.TypeVar('V')
T_co = t.TypeVar('T_co', covariant=True)

WSPACE_RE = re.compile(r"\s+")


class VariadicCallable(t.Protocol, t.Generic[V]):
    def __call__(self, *args: V) -> V:
        ...


@dataclass
class Op(ABC, t.Generic[V]):
    aliases: t.List[str]
    call: t.Callable = field(repr=False)

    @property
    @abstractmethod
    def requires_whitespace(self) -> bool:
        ...


@dataclass
class NaryOp(Op[V]):
    call: VariadicCallable[V] = field(repr=False)  # type: ignore
    precedence: int = 5
    requires_whitespace: bool = False  # type: ignore

    def __call__(self, *args: V) -> V:
        return self.call(*args)

    def precedes(self, other: t.Union[BinaryOp, NaryOp, int]) -> bool:
        if isinstance(other, (BinaryOp, NaryOp)):
            other = other.precedence

        # default to lower precedence for Nary
        return self.precedence > other


@dataclass
class BinaryOp(Op[V]):
    call: t.Callable[[V, V], V] = field(repr=False)  # type: ignore
    precedence: int = 5
    right_assoc: bool = False
    requires_whitespace: bool = False  # type: ignore

    def __call__(self, lhs: V, rhs: V) -> V:
        return self.call(lhs, rhs)

    def precedes(self, other: t.Union[BinaryOp, NaryOp, int]) -> bool:
        """Returns true if self is higher precedence than other.
        Handles a tie using the associativity of self.
        """
        if isinstance(other, (BinaryOp, NaryOp)):
            other = other.precedence

        if self.precedence == other:
            return self.right_assoc
        return self.precedence > other


@dataclass
class UnaryOp(Op[V]):
    call: t.Callable[[V], V] = field(repr=False)  # type: ignore
    requires_whitespace: bool = False  # type: ignore

    def __call__(self, inner: V) -> V:
        return self.call(inner)


@dataclass
class BinaryOrUnaryOp(BinaryOp[V], UnaryOp[V]):
    call: t.Callable[[V, t.Optional[V]], V] = field(repr=False)  # type: ignore

    def __call__(self, lhs: V, rhs: t.Optional[V] = None) -> V:
        return self.call(lhs, rhs)


@dataclass
class Token(ABC, t.Generic[T_co, V]):
    raw: str
    line: int
    span: t.Tuple[int, int]

    def __str__(self):
        return self.raw


@dataclass
class OpToken(Token[T_co, V]):
    op: Op[V]


@dataclass
class GroupOpenToken(Token):
    ...


@dataclass
class GroupCloseToken(Token):
    ...


class WhitespaceToken(Token):
    ...


@dataclass
class ValueToken(Token[T_co, V]):
    val: T_co


class Expr(ABC, t.Generic[T_co, V]):
    @abstractmethod
    def eval(self, map_f: t.Callable[[T_co], V]) -> V:
        ...

    @abstractmethod
    def format(self,
               format_scalar: t.Callable[[ValueToken[T_co, V]], str] = str,
               format_op: t.Callable[[OpToken[T_co, V]], str] = str) -> str:
        ...

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        ...


@dataclass
class UnaryExpr(Expr[T_co, V]):
    op_token: OpToken[T_co, V]
    op: UnaryOp[V] = field(init=False)
    inner: Expr[T_co, V]
    lspace: str = ""

    def __post_init__(self):
        if not isinstance(self.op_token.op, UnaryOp):
            raise TypeError()
        self.op = self.op_token.op

    def eval(self, map_f: t.Callable[[T_co], V]) -> V:
        return self.op.call(self.inner.eval(map_f))

    def format(self,
               format_scalar: t.Callable[[ValueToken[T_co, V]], str] = str,
               format_op: t.Callable[[OpToken[T_co, V]], str] = str) -> str:
        return f"{self.lspace}{format_op(self.op_token)}{self.inner.format(format_scalar, format_op)}"


@dataclass
class BinaryExpr(Expr[T_co, V]):
    op_token: OpToken[T_co, V]
    op: BinaryOp[V] = field(init=False)
    lhs: Expr[T_co, V]
    rhs: Expr[T_co, V]

    def __post_init__(self):
        if not isinstance(self.op_token.op, BinaryOp):
            raise TypeError()
        self.op = self.op_token.op

    def eval(self, map_f: t.Callable[[T_co], V]) -> V:
        return self.op.call(self.lhs.eval(map_f), self.rhs.eval(map_f))

    def format(self,
               format_scalar: t.Callable[[ValueToken[T_co, V]], str] = str,
               format_op: t.Callable[[OpToken[T_co, V]], str] = str) -> str:
        return f"{self.lhs.format(format_scalar, format_op)}{format_op(self.op_token)}{self.rhs.format(format_scalar, format_op)}"


@dataclass
class NaryExpr(Expr[T_co, V]):
    op_tokens: t.Sequence[OpToken[T_co, V]]
    op: NaryOp[V] = field(init=False)
    args: t.Sequence[Expr[T_co, V]]
    rspace: str = ""

    def __post_init__(self):
        op = next(token.op for token in self.op_tokens)
        if not all(token.op == op for token in self.op_tokens[1:]):
            raise ValueError("All `op`s must be identical inside a NaryExpr")
        if not isinstance(op, NaryOp):
            raise TypeError()
        self.op = op

        assert len(self.op_tokens) == len(self.args) - 1

    def eval(self, map_f: t.Callable[[T_co], V]) -> V:
        return self.op.call(*map(lambda expr: expr.eval(map_f), self.args))

    def format(self,
               format_scalar: t.Callable[[ValueToken[T_co, V]], str] = str,
               format_op: t.Callable[[OpToken[T_co, V]], str] = str) -> str:
        return "".join(interleave(
            map(lambda expr: expr.format(format_scalar, format_op), self.args),
            map(format_op, self.op_tokens)
        )) + self.rspace


@dataclass
class GroupExpr(Expr[T_co, V]):
    open: GroupOpenToken
    inner: Expr[T_co, V]
    close: GroupCloseToken
    lspace: str = ""
    rspace: str = ""

    def eval(self, map_f: t.Callable[[T_co], V]) -> V:
        return self.inner.eval(map_f)
    
    def format(self,
               format_scalar: t.Callable[[ValueToken[T_co, V]], str] = str,
               format_op: t.Callable[[OpToken[T_co, V]], str] = str) -> str:
        return f"{self.lspace}{self.open}{self.inner.format(format_scalar, format_op)}{self.close}{self.rspace}"


@dataclass
class ValueExpr(Expr[T_co, V]):
    token: ValueToken[T_co, V]
    lspace: str = ""
    rspace: str = ""

    def eval(self, map_f: t.Callable[[T_co], V]) -> V:
        return map_f(self.token.val)

    def format(self,
               format_scalar: t.Callable[[ValueToken[T_co, V]], str] = str,
               format_op: t.Callable[[OpToken[T_co, V]], str] = str) -> str:
        return f"{self.lspace}{format_scalar(self.token)}{self.rspace}"


@dataclass(init=False)
class Parser(t.Generic[T_co, V]):
    parse_scalar: t.Callable[[str], T_co]
    ops: t.Dict[str, Op[V]]
    group_open: t.Dict[str, int]
    group_close: t.Dict[str, int]

    token_re: re.Pattern = field(init=False)
    """Regex matching operators, brackets, and whitespace"""

    @t.overload
    def __init__(self: Parser[str, V], ops: t.Sequence[Op[V]],
                 parse_scalar: t.Optional[t.Callable[[str], str]] = None,
                 groups: t.Optional[t.Sequence[t.Tuple[str, str]]] = None):
        ...

    @t.overload
    def __init__(self: Parser[T_co, V], ops: t.Sequence[Op[V]],
                 parse_scalar: t.Callable[[str], T_co],
                 groups: t.Optional[t.Sequence[t.Tuple[str, str]]] = None):
        ...

    def __init__(self, ops: t.Sequence[Op[V]],
                 parse_scalar: t.Optional[t.Callable[[str], t.Union[str, T_co]]] = None,
                 groups: t.Optional[t.Sequence[t.Tuple[str, str]]] = None):
        self.parse_scalar = t.cast(t.Callable[[str], T_co], parse_scalar or (lambda s: s))

        if groups is None:
            groups = [('(', ')'), ('[', ']')]

        match_dict: t.Dict[str, t.Optional[Op[V]]] = {}

        self.group_open = {}
        self.group_close = {}
        for (i, (group_open, group_close)) in enumerate(groups):
            if group_open in match_dict:
                raise ValueError(f"Group open token '{group_open}' already defined")
            if group_close in match_dict:
                raise ValueError(f"Group close token '{group_close}' already defined")
            self.group_open[group_open] = i
            self.group_close[group_close] = i
            match_dict[group_open] = None
            match_dict[group_close] = None

        nary_precedences = set()
        for op in ops:
            if isinstance(op, NaryOp):
                if op.precedence in nary_precedences:
                    raise ValueError("N-ary operators must have distinct precedences. "
                                     f"Precedence {op.precedence} conflicts with {op!r}")
                nary_precedences.add(op.precedence)
            for alias in op.aliases:
                if alias in match_dict:
                    raise ValueError(f"Alias '{alias}' already defined")
                match_dict[alias] = op

        match_list = list(match_dict.items())
        match_list.sort(key=lambda a: -len(a[0]))  #longer operators match first
        self.ops = {k: v for (k, v) in match_list if v is not None}

        def op_to_regex(tup: t.Tuple[str, t.Optional[Op[V]]]):
            alias, op = tup
            s = re.escape(alias)  #escape operator for use in regex
            if op is not None and op.requires_whitespace:
                #assert operator must be surrounded by whitespace
                s = r"(?<=\s){}(?=\s)".format(s)
            return s
 
        op_alternation = "|".join(map(op_to_regex, match_list))
        self.token_re = re.compile(f"(\\s+|{op_alternation})")

    def parse(self, reader: TextIOBase) -> Expr[T_co, V]:
        state = ParseState(self, reader)
        expr = state.parse_expr()
        if not state.empty():
            raise ValueError(f"At {state.line}:{state.char}, expected binary operator or end of expression, instead got token {state.peek()}")
        return expr


class ParseState(t.Generic[T_co, V]):
    def __init__(self, parser: Parser[T_co, V], reader: TextIOBase):
        self.parser: Parser[T_co, V] = parser
        self._reader = reader
        self._buf: t.Optional[str] = None
        self._peek: t.List[Token[T_co, V]] = []
        self.line = 0
        self.char = 1

    def empty(self) -> bool:
        return self.peek() is None
    
    def _get_buf(self) -> t.Optional[str]:
        if self._buf is not None:
            return self._buf
        try:
            self._buf = next(self._reader)
            self.line += 1
            self.char = 1
        except StopIteration:
            pass
        return self._buf

    def _refill_peek(self):
        if len(self._peek) > 0:
            return
        try:
            buf = next(self._reader)
            self.line += 1
            self.char = 1
        except StopIteration:
            return None
        if buf is None or len(buf) == 0:  # type: ignore
            return None

        split = self.parser.token_re.split(buf)
        for s in split:
            if len(s) == 0:
                continue
            span = (self.char, self.char + len(s))
            self.char += len(s)
            self._peek.append(self.make_token(s, self.line, span))

        self._peek.reverse()

    def peek(self) -> t.Optional[Token[T_co, V]]:
        self._refill_peek()
        return self._peek[-1] if len(self._peek) > 0 else None

    def collect_wspace(self) -> str:
        wspace = ""
        while True:
            token = self.peek()
            if not isinstance(token, WhitespaceToken):
                break
            wspace += token.raw
            self.next()
        return wspace

    def make_token(self, s: str, line: int, span: t.Tuple[int, int]) -> Token[T_co, V]:
        if s in self.parser.group_open:
            return GroupOpenToken(s, line, span)
        if s in self.parser.group_close:
            return GroupCloseToken(s, line, span)
        if s in self.parser.ops:
            return OpToken(s, line, span, self.parser.ops[s])
        if WSPACE_RE.fullmatch(s):
            return WhitespaceToken(s, line, span)

        try:
            return ValueToken(s, line, span, self.parser.parse_scalar(s))
        except (ValueError, TypeError):
            raise ValueError(f"Syntax error at {line}:{span[0]}: Unexpected token '{s}'") from None

    def next(self) -> t.Optional[Token[T_co, V]]:
        token = self.peek()
        if token is not None:
            self._peek.pop()
        return token

    def parse_expr(self) -> Expr[T_co, V]:
        """
            EXPR := PRIMARY, [ BINARY ]
        """
        logging.debug("parse_expr()")
        lhs = self.parse_primary()
        return self.parse_nary(lhs)

    def parse_nary(self, lhs: Expr[T_co, V], level: t.Optional[int] = None) -> Expr[T_co, V]:
        """
            NARY := { BINARY_OP | NARY_OP, ( PRIMARY, ? higher precedence NARY ? ) }
        """
        logging.debug(f"parse_nary({lhs}, level={level})")
        token = self.peek()
        logging.debug(f"token: '{token!r}'")
        
        while token is not None:
            if not isinstance(token, OpToken) or \
               not isinstance(token.op, (NaryOp, BinaryOp)):
                break

            if level is not None and not token.op.precedes(level):
                # next op has lower precedence, it needs to be parsed at a higher level
                break

            self.next()
            rhs = self.parse_primary()
            logging.debug(f"rhs: '{rhs}'")

            inner = self.peek()
            if not inner is None and isinstance(inner, OpToken):
                inner_op = inner.op
                if isinstance(inner_op, (NaryOp, BinaryOp)) and \
                    inner_op.precedes(token.op):
                    #rhs is actually lhs of an inner expression
                    rhs = self.parse_nary(rhs, token.op.precedence)

            # append rhs to lhs and loop
            if isinstance(token.op, NaryOp):
                if isinstance(lhs, NaryExpr) and token.op == lhs.op:
                    # append to existing n-ary node
                    lhs = NaryExpr(list(lhs.op_tokens) + [token], list(lhs.args) + [rhs])
                else:
                    # make new n-ary expression
                    lhs = NaryExpr([token], [lhs, rhs])
            else:
                # or make binary expression
                lhs = BinaryExpr(token, lhs, rhs)

            token = self.peek()

        return lhs

    def parse_primary(self) -> Expr[T_co, V]:
        """
            PRIMARY := GROUP_OPEN, EXPR, GROUP_CLOSE | UNARY_OP, PRIMARY | SCALAR
        """

        logging.debug("parse_primary()")
        lspace = self.collect_wspace()
        token = self.peek()
        logging.debug(f"token: '{token!r}'")
        if token is None:
            raise ValueError("Unexpected EOF while parsing expression")

        if isinstance(token, GroupOpenToken):
            self.next()
            inner = self.parse_expr()
            close = self.next()
            rspace = self.collect_wspace()
            if close is None:
                raise ValueError(f"Unclosed delimeter '{token.raw}' opened at {token.line}:{token.span[0]}")
            if not isinstance(close, GroupCloseToken):
                raise ValueError(f"At {close.line}:{close.span[0]}: Expected operator or group close, instead got '{close.raw}'")
            if self.parser.group_open[token.raw] != self.parser.group_close[close.raw]:
                raise ValueError(f"At {token.line}:{token.span[0]}-{close.span[1]}: Mismatched delimeters: '{token.raw}' closed with '{close.raw}'")
            return GroupExpr(token, inner, close, lspace, rspace)

        if isinstance(token, GroupCloseToken):
            raise ValueError(f"At {token.line}:{token.span[0]}: Unexpected delimeter '{token.raw.strip()}'")

        if isinstance(token, OpToken):
            self.next()
            if not isinstance(token.op, UnaryOp):
                raise ValueError(f"At {token.line}:{token.span[0]}: Unexpected operator '{token.raw}'. Expected a value or prefix operator.")
            inner = self.parse_primary()
            return UnaryExpr(token, inner, lspace)

        if isinstance(token, ValueToken):
            self.next()
            rspace = self.collect_wspace()
            return ValueExpr(token, lspace, rspace)

        raise TypeError(f"Unknown token type '{type(token)}'")


def interleave(l1: t.Iterable[T_co], l2: t.Iterable[T_co]) -> t.Iterator[T_co]:
    for (v1, v2) in zip_longest(l1, l2):
        yield v1
        if v2 is not None:
            yield v2


SupportsBoolSelf = t.TypeVar('SupportsBoolSelf', bound='SupportsBool')
SupportsNumSelf = t.TypeVar('SupportsNumSelf', bound='SupportsNum')


class SupportsBool(t.Protocol):
    def __and__(self: SupportsBoolSelf, other: SupportsBoolSelf) -> SupportsBoolSelf:
        ...

    def __or__(self: SupportsBoolSelf, other: SupportsBoolSelf) -> SupportsBoolSelf:
        ...

    def __xor__(self: SupportsBoolSelf, other: SupportsBoolSelf) -> SupportsBoolSelf:
        ...

    def __invert__(self: SupportsBoolSelf) -> SupportsBoolSelf:
        ...


class SupportsNum(t.Protocol):
    def __add__(self: SupportsNumSelf, other: SupportsNumSelf) -> SupportsNumSelf:
        ...

    def __sub__(self: SupportsNumSelf, other: SupportsNumSelf) -> SupportsNumSelf:
        ...

    def __mul__(self: SupportsNumSelf, other: SupportsNumSelf) -> SupportsNumSelf:
        ...

    def __truediv__(self: SupportsNumSelf, other: SupportsNumSelf) -> SupportsNumSelf:
        ...

    def __floordiv__(self: SupportsNumSelf, other: SupportsNumSelf) -> SupportsNumSelf:
        ...

    def __mod__(self: SupportsNumSelf, other: SupportsNumSelf) -> SupportsNumSelf:
        ...

    def __pow__(self: SupportsNumSelf, other: SupportsNumSelf) -> SupportsNumSelf:
        ...

    def __neg__(self: SupportsNumSelf) -> SupportsNumSelf:
        ...


def parse_numeric(s: str) -> t.Union[int, float]:
    try:
        return int(s)
    except ValueError:
        pass
    return float(s)


def sub(lhs: SupportsNum, rhs: t.Optional[SupportsNum] = None):
    if rhs is None:
        return -lhs
    return lhs-rhs


def parse_boolean(s: str) -> bool:
    if s.lower() in ("0", "false", "f"):
        return False
    elif s.lower() in ("1", "true", "t"):
        return True
    raise ValueError(f"Can't parse '{s}' as boolean")


NUMERIC_OPS: t.Sequence[Op[SupportsNum]] = [
    BinaryOrUnaryOp(['-'], sub, False, 5),
    BinaryOp(['+'], operator.add, 5),
    BinaryOp(['*'], operator.mul, 6),
    BinaryOp(['/'], operator.truediv, 6),
    BinaryOp(['//'], operator.floordiv, 6),
    BinaryOp(['^', '**'], operator.pow, 7)
]
NUMERIC_PARSER = Parser(NUMERIC_OPS, parse_numeric)


BOOLEAN_OPS: t.Sequence[Op[SupportsBool]] = [
    BinaryOp(['=', '=='], operator.eq, 3),
    BinaryOp(['!=', '<>'], operator.ne, 3),
    BinaryOp(['|', '||'], operator.or_, 4),
    BinaryOp(['&', '&&'], operator.and_, 5),
    BinaryOp(['^'], operator.xor, 6),
    UnaryOp(['!', '~'], operator.invert)
]
BOOLEAN_PARSER = Parser(BOOLEAN_OPS, parse_boolean)
"""Parser for boolean expressions ([1 || false && true])"""


def stack(*vs: numpy.ndarray) -> numpy.ndarray:
    return numpy.stack(vs, axis=0)


VECTOR_OPS: t.Sequence[Op[numpy.ndarray]] = [
    *(t.cast(Op[numpy.ndarray], op) for op in NUMERIC_OPS),
    NaryOp([','], call=stack, precedence=3),
]
VECTOR_PARSER = Parser(VECTOR_OPS, lambda v: numpy.array(parse_numeric(v)))