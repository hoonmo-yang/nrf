from typing import Any, Hashable, Sequence, Union

import hashlib
import json
from io import BytesIO
from operator import attrgetter, itemgetter
from pathlib import Path
import random
import re
import textwrap
import yaml
import zipfile


def load_json_yaml(src: Path, **kwargs) -> dict[Hashable, Any]:
    '''
    Loads json or yaml file to a dict and returns it.
    It is determined by file extension whether the src is
    a json or yaml file.
    '''
    if not src.is_file() or src.suffix not in [".json", ".yaml", ".yml"]:
        raise TypeError(f"{src} isn't either yaml or json file")

    with src.open() as fin:
        deserial = (
            json.load(fin, **kwargs) if src.suffix == ".json" else
            yaml.load(fin, Loader=yaml.Loader, **kwargs)
        )
    return deserial


def dump_json_yaml(content: dict[str, Any], dst: Path, **kwargs) -> None:
    '''
    Dumps the content dict to the dst file. The file serialization format
    (either json or yaml) is deterimed by the file extension of the dst file.
    '''
    if dst.suffix not in [".json', '.yaml', '.yml"]:
        raise TypeError(f"{dst} isn't either yaml or json file")

    with dst.open("wt", encoding="utf-8") as fout:
        if dst.suffix == ".json":
            json.dump(content, fout, ensure_ascii=False, indent=4, **kwargs)
        else:
            yaml.dump(content, fout, sort_keys=False, **kwargs)


def concat_texts(
        chunks: Sequence[Any],
        field: str,
        seperator: str = "\n"
) -> str:
    if not chunks:
        return ""

    sample = chunks[0]

    selector = (
        itemgetter(field)
        if hasattr(sample, "__getitem__") and not isinstance(sample, str) else
        attrgetter(field) if hasattr(sample, field) else
        None
    )

    if selector is None:
        return ""

    return seperator.join([selector(chunk) for chunk in chunks])


def concat_ranges(*args) -> list[int]:
    out = []
    for arg in args:
        if isinstance(arg, int):
            out.append(arg)
        elif isinstance(arg, range):
            out.extend(list(arg))
        else:
            raise TypeError(
                f"{Q(arg)} isn't int or range"
            )
    return out


def Q(text: str) -> str:
    return f"'{text}'"


def QQ(text: str) -> str:
    return f"\"{text}\""


def dedent(source: str) -> str:
    return textwrap.dedent(source).strip()


def ordinal(n: int) -> str:
    suffix = (
        "th" if 10 <= n % 100 <= 13 else
        "st" if n % 10 == 1 else
        "nd" if n % 10 == 2 else
        "rd" if n % 10 == 3 else
        "th"
    )

    return f"{n}{suffix}"


class Struct:
    def __init__(self, data: dict[str, Any]):
        self.__dict__.update(data)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __or__(self, other):
        return self.to_dict() | other

    def __ror__(self, other):
        return other | self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    def get(self, key, alternative=None):
        return self.to_dict().get(key, alternative)

    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()


def to_camel(literal: str) -> str:
    return "".join([
        word[1:-1].upper() if word[0] == "*" else word.capitalize()
        for word in literal.split("_")
    ])


def to_snake(literal: str) -> str:
    snake = re.sub(r"(?<=[a-z])([A-Z])", r'_\1', literal)
    snake = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", snake)
    return snake.lower()


def optional(value: Any, other: Any) -> Any:
    return other if value is None else value


def load_text(src: Path,
              input_variables: dict[str, str],
              ) -> tuple[str, list[str]]:
    deserial = load_json_yaml(src)

    content = deserial.pop("content", "")
    required_variables = deserial.pop("input_variables", [])

    variables = {}
    missings = []

    for key in required_variables:
        value = input_variables.get(key)

        if value is None:
            value = ""
            missings.append()

        variables[key] = value

    result = content.format(**variables)
    return result, missings


def proportionalize(total_size: int,
                    weights: Sequence[int]
                    ) -> list[int]:
    total_sum = sum(weights)

    proportionals = [
        int(total_size * (weight / total_sum)) for weight in weights 
    ]

    distributed_sum = sum(proportionals)
    delta = total_size - distributed_sum

    k = proportionals.index(max(proportionals))
    proportionals[k] += delta

    return proportionals


def divide_evenly(total: int, n: int) -> list[int]:
    q, r = divmod(total, n)

    splits = [q] * n

    indices = set(random.sample(range(n), r))

    one_zeros = [1 if i in indices else 0 for i in range(n)]

    splits = [
        split + one_zero for split, one_zero in zip(splits, one_zeros)
    ]

    return splits


def split_by_length(text: str, n: int, pad: str = "") -> list[str]:
    if len(pad) > 1:
        raise ValueError(
            f"length of pad must be less than 2: {Q(pad)}"
        )

    r = len(text) % n
    if r:
        text += pad * (n - r)

    return [text[i:i+n] for i in range(0, len(text), n)]


def pad_list(texts: list[str], n: int, padding="") -> list[str]:
    texts = texts[:]
    texts.extend([padding] * (n - len(texts)))

    return texts[:n]


def get_empty_html() -> str:
    html = """
    <!DOCTYPE html>
    <html lang="kr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title></title>
    </head>
    <body>
    </body>
    </html>
    """

    return dedent(html)


def generate_checksum(src: Union[str, bytes], algorithm: str) -> str:
    '''
    Gets string or binary arrays, generate a checksum from it and
    returns it. Checksum algorithm is either sha224, sha256 or md5.
    '''
    if algorithm not in ["sha224", "sha256", "md5"]:
        raise RuntimeError(
            f"algorithm {algorithm} isn't supported"
        )
    h = hashlib.new(algorithm)

    if not isinstance(src, (str, bytes)):
        raise TypeError(f"type of {src} isn't str or bytes ({type(src)})")

    if isinstance(src, str):
        src = src.encode("utf-8")

    h.update(src)
    return h.hexdigest()


def generate_checksum_file(
    src: Path, algorithm: str, block_size: int = 1048576
) -> str:
    '''
    Generate a checksum from a file and returns it. The supported
    checksum algorithms are sha224, sha256 and md5.
    '''
    if algorithm not in ["sha224", "sha256", "md5"]:
        raise RuntimeError(
            f"algorithm {algorithm} isn't supported"
        )
    h = hashlib.new(algorithm)

    with src.open("rb") as fin:
        for byte_block in iter(lambda: fin.read(block_size), b''):
            h.update(byte_block)
    return h.hexdigest()


def generate_zip_buffer(files: list[Path]) -> BytesIO:
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_ in files:
            zip_file.write(file_, file_.name)

    zip_buffer.seek(0)
    return zip_buffer
