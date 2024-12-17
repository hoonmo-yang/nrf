from io import StringIO
import json
import pandas as pd
from pathlib import Path


def to_json(target: Path, *args, **kwargs) -> None:
    data = [
        dataframe.to_json(orient="split", **kwargs)
        for dataframe in args
    ]

    with target.open("wt") as fout:
        json.dump(data, fout)


def from_json(source: Path, **kwargs) -> list[pd.DataFrame]:
    with source.open("rt") as fin:
        data = json.load(fin)

    dataframes = [
        pd.read_json(StringIO(serial), orient="split", **kwargs)
        for serial in data
    ]

    return dataframes
