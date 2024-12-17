import argparse
import os
from pathlib import Path
import sys

from cenai_core.grid.runner import BaseRunner
from cenai_core.logger import Logger
from cenai_core.dataman import Q
from cenai_core.system import cenai_path


class GridCLI(Logger):
    profile_dir = Path("profile")
    logger_name = "cenai.system"

    def __init__(self,
                 title: str,
                 runner: BaseRunner,
                 search_name: str,
                 search_dir: str,
                 search_pattern: str = "*",
                 search_type: str = "",
                 argvs: list[str] = [],
                 environ: dict[str, str] = {}
                 ):

        super().__init__()

        self._runner = runner

        os.environ |= environ

        if not argvs:
            argvs = sys.argv[1:]

        self._title = title
        self._name = search_name

        self._search_dir = Path(
            cenai_path(search_dir[1:])
            if search_dir.startswith("%") else
            search_dir
        )

        self._search_pattern = search_pattern

        self._search_checker = (
            lambda x: getattr(x, "is_dir")
            if search_type == "dir" else

            lambda x: getattr(x, "is_file")
            if search_type == "file" else

            lambda _: True
        )

        self._option = self._get_option(argvs)

    def _get_option(self,
                    argvs: list[str]
                    ) -> argparse.Namespace:

        parser = argparse.ArgumentParser(
            description=f"{self._title}"
        )

        parser.add_argument(
            self._name,
            nargs="*",
            default=[],
            help=f"list of {self._name}"
        )

        return parser.parse_args(argvs)

    def _choose_paths(self) -> list[Path]:
        extension = Path(self._search_pattern).suffix

        paths = [
            self._search_dir / f"{name}{extension}"
            for name in getattr(self._option, self._name)
        ]

        if paths:
            return paths

        paths = [
            path for path in self._search_dir.glob(self._search_pattern)
            if self._search_checker(path)
        ]

        if not paths:
            raise RuntimeError(
                f"no {self._name} in {Q(self._search_dir)}"
            )

        items = [
            (f"[{i + 1}] {path.stem}")
            for i, path in enumerate(paths)
        ]

        while True:
            answer = input(
                f"\n{'\n'.join(items)}\n\n"
                f"Choose a {self._name} for {Q(self._title)} "
                "by number (q for exit): "
            )

            answer = answer.strip()

            if answer.lower() == "q":
                return []

            if answer.isdigit():
                k = int(answer) - 1
                if k < len(items):
                    path = paths[k]
                    break

            self.ERROR(f"\nwrong selection - {Q(answer)}\n")

        return [path]

    def invoke(self) -> None:
        paths = self._choose_paths()

        for path in paths:
            self.INFO(
                f"{self._name.upper()} {Q(path.name)} "
                f"for {Q(self._title)} proceed ...."
            )

            runner = self._runner(path)
            
            runner.invoke()
            runner.export_tables()
            runner.export_documents()

            self.INFO(
                f"{self._name.upper()} {Q(path.name)} "
                f"for {Q(self._title)} proceed DONE"
            )

        self.INFO("bye")
