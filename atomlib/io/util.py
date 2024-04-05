from __future__ import annotations

from itertools import islice
from io import TextIOBase
import typing as t

import polars


class LineBuffer:
    def __init__(self, f: TextIOBase, start_line: int = 0):
        self._inner = f
        self.start_line: int = start_line
        self.line: int = start_line
        # invariants:
        # - len(self.line_pos) == self.line - start_line + 1
        # - line_pos[-1] is the current 'logical' position
        #   which is equal to f.tell() - len(self._peek)
        self.line_pos: t.List[int] = [f.tell()]
        self._peek: str = ""

    def __iter__(self) -> LineBuffer:
        return self

    def __next__(self) -> str:
        if (line := self.next_line()) is None:
            raise StopIteration()
        return line

    def get_inner(self) -> t.Tuple[TextIOBase, int]:
        if self._peek:
            self._inner.seek(self.line_pos[-1])
            self._peek = ""
        return (self._inner, self.line)

    def peek_line(self) -> t.Optional[str]:
        if not self._peek:
            self._peek = self._inner.readline()
        return self._peek if len(self._peek) else None

    def next_line(self) -> t.Optional[str]:
        if (ret := self.peek_line()) is not None:
            self.line_pos.append(self.line_pos[-1] + len(self._peek))
            self.line += 1
            self._peek = ""
        return ret

    def prev_line(self) -> None:
        try:
            self.line_pos.pop()
            self._inner.seek(self.line_pos[-1])
        except IndexError:
            raise ValueError("prev_line before start_line")
        self.line -= 1
        self._peek = ""

    def goto_line(self, line: int) -> None:
        i = line - self.start_line
        if i < 0:
            raise ValueError("line before start_line")
        try:
            self._inner.seek(self.line_pos[i])
        except IndexError:
            raise ValueError("line after current line")
        self.line_pos = self.line_pos[:i+1]
        self.line = line
        self._peek = ""

    def get_line(self, line: int) -> str:
        i = line - self.start_line
        if i < 0:
            raise ValueError("line before start_line")
        ret_pos = self._inner.tell()
        try:
            self._inner.seek(self.line_pos[i])
        except IndexError:
            raise ValueError("line after current line")
        ret = self._inner.readline()
        self._inner.seek(ret_pos)
        return ret


def parse_whitespace_separated(
    f: t.Union[t.Sequence[str], LineBuffer, TextIOBase], schema: t.Dict[str, t.Type[polars.DataType]], *,
    n_rows: t.Optional[int] = None,
    start_line: int = 0,
    allow_extra_cols: bool = False,
    allow_initial_whitespace: bool = True,
    allow_comment: bool = True
) -> polars.DataFrame:
    get_line: t.Callable[[int], str]

    if isinstance(f, t.Sequence):
        it = iter(f)
        get_line = lambda i: f[i]
    else:
        if isinstance(f, LineBuffer):
            buf = f
            start_line = buf.line
        else:
            buf = LineBuffer(f, start_line=start_line)

        it = buf
        get_line = buf.get_line

    if n_rows is not None:
        it = islice(it, n_rows)

    return _parse_rows_whitespace_separated(
        it, get_line, schema,
        start_line=start_line,
        allow_extra_cols=allow_extra_cols,
        allow_initial_whitespace=allow_initial_whitespace,
        allow_comment=allow_comment
    )


def _parse_rows_whitespace_separated(
    it: t.Iterator[str], get_line: t.Callable[[int], str],
    schema: t.Dict[str, t.Type[polars.DataType]], *,
    start_line: int = 0,
    allow_extra_cols: bool = False,
    allow_initial_whitespace: bool = True,
    allow_comment: bool = True
) -> polars.DataFrame:

    regex = "".join((
        "^",
        r"\s*" if allow_initial_whitespace else "",
        r"\s+".join(rf"(?<{col}>\S+)" for col in schema.keys()),
        '' if allow_extra_cols else (r'\s*(?:#|$)' if allow_comment else r'\s*$')
    ))

    df = polars.LazyFrame({'s': it}, schema={'s': polars.Utf8}).select(polars.col('s').str.extract_groups(regex)).unnest('s').collect()

    if len(df.lazy().filter(polars.col(df.columns[0]).is_null()).first().collect()):
        # failed to match
        first_failed_row = df.lazy().with_row_index('_row_index').filter(polars.col(df.columns[0]).is_null()).first().collect()['_row_index'][0]
        row = get_line(first_failed_row + start_line)
        #row = next(islice(make_row_gen(), first_failed_row, None))
        if allow_comment:
            row = row.split('#', maxsplit=1)[0]
        raise ValueError(f"At line {start_line + first_failed_row}: Failed to parse line '{row.strip()}'. Expected {len(schema)} columns.")

    # TODO better error handling here
    df = df.select(polars.col(col).cast(dtype) for (col, dtype) in schema.items())

    return df