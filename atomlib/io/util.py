
from itertools import islice
from io import TextIOBase
import typing as t

import polars


def parse_file_whitespace_separated(
    f: TextIOBase, schema: t.Dict[str, t.Type[polars.DataType]], *,
    n_rows: t.Optional[int] = None, start_row: int = 0,
    allow_extra_cols: bool = False,
    allow_initial_whitespace: bool = True,
    allow_comment: bool = True
) -> polars.DataFrame:
    start_pos = f.tell()

    def make_row_gen() -> t.Iterator[str]:
        f.seek(start_pos)
        for line in f if n_rows is None else islice(f, n_rows):
            yield line

    return _parse_rows_whitespace_separated(
        make_row_gen, schema, start_row=start_row,
        allow_extra_cols=allow_extra_cols,
        allow_initial_whitespace=allow_initial_whitespace,
        allow_comment=allow_comment
    )


def _parse_rows_whitespace_separated(
    make_row_gen: t.Callable[[], t.Iterator[str]],
    schema: t.Dict[str, t.Type[polars.DataType]], *,
    start_row: int = 0,
    allow_extra_cols: bool = False,
    allow_initial_whitespace: bool = True,
    allow_comment: bool = True
) -> polars.DataFrame:

    regex = "".join((
        "^",
        r"\s*" if allow_initial_whitespace else "",
        r"\s+".join(rf"(?<{col}>\S+)" for col in schema.keys()),
        '' if allow_extra_cols else (r'\s*#' if allow_comment else r'\s*$')
    ))

    df = polars.LazyFrame({'s': make_row_gen()}, schema={'s': polars.Utf8}).select(polars.col('s').str.extract_groups(regex)).unnest('s').collect()

    if len(df.lazy().filter(polars.col(df.columns[0]).is_null()).first().collect()):
        # failed to match
        first_failed_row = df.lazy().with_row_index('i').filter(polars.col(df.columns[0]).is_null()).first().collect()['i'][0]
        row = next(islice(make_row_gen(), first_failed_row, None))
        if allow_comment:
            row = row.split('#', maxsplit=1)[0]
        raise ValueError(f"At line {start_row + first_failed_row}: Failed to parse line '{row}'. Expected {len(schema)} columns.")

    # TODO better error handling here
    df = df.select(polars.col(col).cast(dtype) for (col, dtype) in schema.items())

    return df