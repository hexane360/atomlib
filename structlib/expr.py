from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO, TextIOBase
import logging
import operator
import re
import typing as t

V = t.TypeVar('V')
T = t.TypeVar('T', covariant=True)

WSPACE_RE = re.compile(r"\s+")


@dataclass
class Op(ABC, t.Generic[V]):
    aliases: t.List[str]
    call: t.Callable = field(repr=False)

    @property
    @abstractmethod
    def requires_whitespace(self):
        ...


@dataclass
class BinaryOp(Op[V]):
    call: t.Callable[[V, V], V] = field(repr=False)
    precedence: int = 5
    right_assoc: bool = False
    requires_whitespace: bool = False

    def __call__(self, lhs: V, rhs: V) -> V:
        return self.call(lhs, rhs)

    def precedes(self, other: t.Union[BinaryOp, int]) -> bool:
        """Returns true if self is higher precedence than other.
        Handles a tie using the associativity of self.
        """
        if isinstance(other, BinaryOp):
            other = other.precedence

        if self.precedence == other:
            return self.right_assoc
        return self.precedence > other


@dataclass
class UnaryOp(Op[V]):
    call: t.Callable[[V], V] = field(repr=False)
    requires_whitespace: bool = False

    def __call__(self, inner: V) -> V:
        return self.call(inner)


@dataclass
class BinaryOrUnaryOp(BinaryOp[V], UnaryOp[V]):
    call: t.Callable[[V, t.Optional[V]], V] = field(repr=False)

    def __call__(self, lhs: V, rhs: t.Optional[V] = None) -> V:
        return self.call(lhs, rhs)


@dataclass
class Token(ABC, t.Generic[T, V]):
    raw: str
    line: int
    span: t.Tuple[int, int]

    def __str__(self):
        return self.raw


@dataclass
class OpToken(Token[T, V]):
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
class ValueToken(Token[T, V]):
    val: T


class Expr(ABC, t.Generic[T, V]):
    @abstractmethod
    def eval(self, map_f: t.Callable[[T], V]) -> V:
        ...

    @abstractmethod
    def format(self,
               format_scalar: t.Callable[[ValueToken[T, V]], str] = str,
               format_op: t.Callable[[OpToken[T, V]], str] = str) -> str:
        ...

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        ...


@dataclass
class UnaryExpr(Expr[T, V]):
    op_token: OpToken[T, V]
    op: UnaryOp[V] = field(init=False)
    inner: Expr[T, V]
    lspace: str = ""

    def __post_init__(self):
        if not isinstance(self.op_token.op, UnaryOp):
            raise TypeError()
        self.op = self.op_token.op

    def eval(self, map_f: t.Callable[[T], V]) -> V:
        return self.op.call(self.inner.eval(map_f))

    def format(self,
               format_scalar: t.Callable[[ValueToken[T, V]], str] = str,
               format_op: t.Callable[[OpToken[T, V]], str] = str) -> str:
        return f"{self.lspace}{format_op(self.op_token)}{self.inner.format(format_scalar, format_op)}"


@dataclass
class BinaryExpr(Expr[T, V]):
    op_token: OpToken[T, V]
    op: BinaryOp[V] = field(init=False)
    lhs: Expr[T, V]
    rhs: Expr[T, V]

    def __post_init__(self):
        if not isinstance(self.op_token.op, BinaryOp):
            raise TypeError()
        self.op = self.op_token.op

    def eval(self, map_f: t.Callable[[T], V]) -> V:
        return self.op.call(self.lhs.eval(map_f), self.rhs.eval(map_f))

    def format(self,
               format_scalar: t.Callable[[ValueToken[T, V]], str] = str,
               format_op: t.Callable[[OpToken[T, V]], str] = str) -> str:
        return f"{self.lhs.format(format_scalar, format_op)}{format_op(self.op_token)}{self.rhs.format(format_scalar, format_op)}"


@dataclass
class GroupExpr(Expr[T, V]):
    open: GroupOpenToken
    inner: Expr[T, V]
    close: GroupCloseToken
    lspace: str = ""
    rspace: str = ""

    def eval(self, map_f: t.Callable[[T], V]) -> V:
        return self.inner.eval(map_f)
    
    def format(self,
               format_scalar: t.Callable[[ValueToken[T, V]], str] = str,
               format_op: t.Callable[[OpToken[T, V]], str] = str) -> str:
        return f"{self.lspace}{self.open}{self.inner.format(format_scalar, format_op)}{self.close}{self.rspace}"


@dataclass
class ValueExpr(Expr[T, V]):
    token: ValueToken[T, V]
    lspace: str = ""
    rspace: str = ""

    def eval(self, map_f: t.Callable[[T], V]) -> V:
        return map_f(self.token.val)

    def format(self,
               format_scalar: t.Callable[[ValueToken[T, V]], str] = str,
               format_op: t.Callable[[OpToken[T, V]], str] = str) -> str:
        return f"{self.lspace}{format_scalar(self.token)}{self.rspace}"


@dataclass(init=False)
class Parser(t.Generic[T, V]):
    parse_scalar: t.Callable[[str], T]
    ops: t.Dict[str, Op[V]]
    binary_ops: t.Dict[str, BinaryOp[V]]
    unary_ops: t.Dict[str, UnaryOp[V]]
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
    def __init__(self: Parser[T, V], ops: t.Sequence[Op[V]],
                 parse_scalar: t.Optional[t.Callable[[str], T]],
                 groups: t.Optional[t.Sequence[t.Tuple[str, str]]] = None):
        ...

    def __init__(self, ops: t.Sequence[Op[V]],
                 parse_scalar: t.Optional[t.Callable[[str], T]] = None,
                 groups: t.Optional[t.Sequence[t.Tuple[str, str]]] = None):
        self.parse_scalar = parse_scalar or t.cast(t.Callable[[str], T], lambda s: s)

        if groups is None:
            groups = [('(', ')'), ('[', ']')]

        self.group_open = {}
        self.group_close = {}
        for (i, (group_open, group_close)) in enumerate(groups):
            if group_open in self.group_open or group_open in self.group_close:
                raise ValueError(f"Group open token '{group_open}' already defined")
            if group_close in self.group_open or group_open in self.group_close:
                raise ValueError(f"Group close token '{group_close}' already defined")
            self.group_open[group_open] = i
            self.group_close[group_close] = i

        self.ops = {}
        self.binary_ops = {}
        self.unary_ops = {}
        for op in ops:
            is_binary = isinstance(op, BinaryOp)
            is_unary = isinstance(op, UnaryOp)
            if not (is_binary or is_unary):
                raise TypeError(f"Unknown operator type '{type(op)}'")
            for alias in op.aliases:
                if alias in self.ops:
                    raise ValueError(f"Alias '{alias}' already defined")
                if is_binary:
                    self.binary_ops[alias] = op
                if is_unary:
                    self.unary_ops[alias] = op
                self.ops[alias] = op

        match_list: t.List[t.Tuple[str, t.Optional[Op[V]]]] = list(self.ops.items())
        match_list.extend((t, None) for t in self.group_open.keys())
        match_list.extend((t, None) for t in self.group_close.keys())
        match_list.sort(key=lambda a: -len(a[0]))  #longer operators match first

        def op_to_regex(op):
            alias, op = op
            s = re.escape(alias)  #escape operator for use in regex
            if op is not None and op.requires_whitespace:
                #assert operator must be surrounded by whitespace
                s = r"(?<=\s){}(?=\s)".format(s)
            return s
 
        op_alternation = "|".join(map(op_to_regex, match_list))
        self.token_re = re.compile(f"(\\s+|{op_alternation})")

    def parse(self, reader: TextIOBase) -> Expr[T, V]:
        state = ParseState(self, reader)
        expr = state.parse_expr()
        if not state.empty():
            raise ValueError(f"At {state.line}:{state.char}, expected binary operator or end of expression, instead got token {state.peek()}")
        return expr


class ParseState(t.Generic[T, V]):
    def __init__(self, parser: Parser[T, V], reader: TextIOBase):
        self.parser: Parser[T, V] = parser
        self._reader = reader
        self._buf: t.Optional[str] = None
        self._peek: t.List[Token[T, V]] = []
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
        if buf is None or len(buf) == 0:
            return None

        split = self.parser.token_re.split(buf)
        for s in split:
            if len(s) == 0:
                continue
            span = (self.char, self.char + len(s))
            self.char += len(s)
            self._peek.append(self.make_token(s, self.line, span))

        self._peek.reverse()

    def peek(self) -> t.Optional[Token[T, V]]:
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

    def make_token(self, s, line, span) -> Token[T, V]:
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
            raise ValueError(f"Syntax error at {line}:{span[0]}-{span[1]}: Unexpected token '{s}'") from None

    def next(self) -> t.Optional[Token[T, V]]:
        token = self.peek()
        if token is not None:
            self._peek.pop()
        return token

    def parse_expr(self) -> Expr[T, V]:
        """
            EXPR := PRIMARY, [ BINARY ]
        """
        logging.debug("parse_expr()")
        lhs = self.parse_primary()
        return self.parse_binary(lhs)

    def parse_binary(self, lhs: Expr[T, V], level: t.Optional[int] = None) -> Expr[T, V]:
        """
            BINARY := { BINARY_OP, ( PRIMARY, ? higher precedence BINARY ? ) }
        """
        logging.debug(f"parse_binary({lhs}, level={level})")
        token = self.peek()
        logging.debug(f"token: '{token!r}'")
        
        while token is not None:
            if not isinstance(token, OpToken) or \
               not isinstance(token.op, BinaryOp):
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
                if isinstance(inner_op, BinaryOp) and \
                    inner_op.precedes(token.op):
                    #rhs is actually lhs of an inner expression
                    rhs = self.parse_binary(rhs, token.op.precedence)

            # append rhs to lhs and loop
            lhs = BinaryExpr(token, lhs, rhs)
            token = self.peek()

        return lhs

    def parse_primary(self) -> Expr[T, V]:
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
                raise ValueError(f"At {token.line}:{token.span[0]}: Unclosed delimeter '{token.raw}'")
            if not isinstance(close, GroupCloseToken):
                raise ValueError(f"Expected operator or group close, instead got '{close.raw}'")
            if self.parser.group_open[token.raw] != self.parser.group_close[close.raw]:
                raise ValueError(f"Mismatched delimeters: Opener '{token.raw}' closed with '{close.raw}'")
            return GroupExpr(token, inner, close, lspace, rspace)
        
        if isinstance(token, GroupCloseToken):
            raise ValueError(f"Unexpected close delimeter: '{token.raw.strip()}'")
        
        if isinstance(token, OpToken):
            self.next()
            if not isinstance(token.op, UnaryOp):
                raise ValueError(f"Unexpected binary operator '{token.raw}'. Expected a value or prefix operator.")
            inner = self.parse_primary()
            return UnaryExpr(token, inner, lspace)

        if isinstance(token, ValueToken):
            self.next()
            rspace = self.collect_wspace()
            return ValueExpr(token, lspace, rspace)
        
        raise TypeError(f"Unknown token type '{type(token)}'")


def parse_numeric(s: str) -> t.Union[int, float]:
    try:
        return int(s)
    except ValueError:
        pass
    return float(s)


def sub(lhs, rhs=None):
    if rhs is None:
        return -lhs
    return lhs-rhs


def parse_boolean(s) -> bool:
    if s.lower() in ("0", "false", "f"):
        return False
    elif s.lower() in ("1", "true", "t"):
        return True
    raise ValueError(f"Can't parse '{s}' as boolean")


NUMERIC_OPS: t.Sequence[Op[t.Union[int, float]]] = [
    BinaryOrUnaryOp(['-'], sub, False, 5),
    BinaryOp(['+'], operator.add, 5),
    BinaryOp(['*'], operator.mul, 6),
    BinaryOp(['/'], operator.truediv, 6),
    BinaryOp(['//'], operator.floordiv, 6),
    BinaryOp(['^', '**'], operator.pow, 7)
]
NUMERIC_PARSER: Parser[t.Union[int, float], t.Union[int, float]] = Parser(NUMERIC_OPS, parse_numeric)


BOOLEAN_OPS: t.Sequence[Op[bool]] = [
    BinaryOp(['=', '=='], operator.eq, 3),
    BinaryOp(['!=', '<>'], operator.ne, 3),
    BinaryOp(['|', '||'], operator.or_, 4),
    BinaryOp(['&', '&&'], operator.and_, 5),
    BinaryOp(['^'], operator.xor, 6),
    UnaryOp(['!', '~'], operator.not_)
]
BOOLEAN_PARSER = Parser(BOOLEAN_OPS, parse_boolean)
"""Parser for boolean expressions ([1 || false && true])"""