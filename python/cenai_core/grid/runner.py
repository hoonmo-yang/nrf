from __future__ import annotations
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import copy
import importlib
from itertools import product
import itertools
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
import shutil
from sklearn.model_selection import train_test_split

from cenai_core.dataman import (
    load_json_yaml, get_empty_html, optional, ordinal, Q, Struct, to_camel
)

from cenai_core.logger import Logger
from cenai_core.pandas_helper import from_json, to_json
from cenai_core.grid.runnable import GridRunnable
from cenai_core.render import html_to_pdf, html_to_docx
from cenai_core.system import cenai_path, load_dotenv

class Gridsuite(BaseModel):
    prefix: str = Field(description="prefix of gridsuite name")
    label: str = Field(description="label of grid suite")
    artifact_dir: Path = Field(description="dir of gridsuite artifact")
    profile_file: str = Field(description="config profile file of gridsuite")

    @property
    def id(self) -> str:
        label = f"_{self.label}" if self.label else ""
        return f"{self.prefix}{label}"

    @property
    def prefix_dir(self) -> str:
        return self.artifact_dir / self.prefix


class BaseRunner(Logger):
    logger_name = "cenai.system"

    def __init__(self):
        super().__init__()


class GridRunner(BaseRunner):
    version = "v1"

    data_dir = cenai_path("data")
    artifact_dir = cenai_path("artifact")

    def __init__(self, profile: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        if profile is not None:
            self.update(profile)

    def update(self, profile: Union[Path, dict[str, Any]]) -> None:
        self._recipe = self._load_gridsuite_recipe(profile)
        self._suite = self._update_gridsuite()

        self._datastore_dir = self.suite.prefix_dir / "datastore"

        self._corpus_dir = self.suite.prefix_dir / "corpus"
        self._corpus_dir.mkdir(parents=True, exist_ok=True)

        self._export_dir = self.suite.prefix_dir / "export"
        self._export_dir.mkdir(parents=True, exist_ok=True)

        source_dir = Path(
            self.data_dir /
            self.recipe.metadata.task
        )

        self._source_corpus_dir = source_dir / "corpus"
        self._html_dir = source_dir / "html"

    @classmethod
    def _load_gridsuite_recipe(
            cls,
            profile: Union[Path, dict[str, Any]]
        ) -> Struct:

        if isinstance(profile, Path):
            profile_file = profile.resolve()
            profile = load_json_yaml(profile_file)
            location = f"file {Q(profile_file)}"
        else:
            profile_file = ""
            profile = copy.deepcopy(profile)
            location = f"var {Q('profile')}",

        cls._check_profile(profile, location)

        entity = {
            "cases": Struct({
                "keyword": "module",
                "implicit_param_keys": [
                    "models",
                ],
                "all_params": [],
            }),
            "corpora": Struct({
                "keyword": "mode",
                "implicit_param_keys": [],
                "all_params": [],
            }),
        }

        for key, context in entity.items():
            templates = profile.pop(key)

            for i, template in enumerate(templates):
                params = {}
                keyword_values = template.pop(context.keyword, [])

                if keyword_values is None:
                    raise ValueError(f"{Q(context.keyword)} key does not exist "
                                     f"or is empty in {ordinal(i + 1)} element"
                                     f"of {Q(key)} key in {location}")

                params[context.keyword] = keyword_values

                if key in ["cases",]:
                    for category, keys in template.items():
                        if category not in profile:
                            raise KeyError(f"{Q(category)} node missing"
                                           f"in {location}")

                        if keys is None:
                            keys = profile[category].keys()

                        for key in keys:
                            branch = profile[category]

                            if key not in branch:
                                raise KeyError(f"{Q(key)} key missing in "
                                               f"{Q(category)} branch "
                                               f"in {location}")

                            if key in params:
                                raise KeyError(f"{Q(category)} key contains "
                                               "a duplicate name for "
                                               f"{Q(key)} key. Change the "
                                               "duplicate keys to resolve it")

                            params[key] = branch[key]

                elif key in ["corpora",]:
                    if len(keyword_values) > 1:
                        raise ValueError(f"{Q(context.keyword)} key has "
                                         "a list with more than 1 element."
                                         f"in {ordinal(i + 1)} element of "
                                         f"{Q(key)} key in {location}")

                    params.update(template)

            implicit_params = {
                key: profile[key] for key in context.implicit_param_keys
            }

            context.all_params.append(implicit_params | params)

        recipe = {
            key: context.all_params
            for key, context in entity.items()
        } | {
            key: Struct(profile[key]) for key in [
                "metadata",
                "directive",
            ]
        } | {
            f"export_{key}": profile["export"][key] for key in [
                "table",
                "document",
            ]
        } | {
            "profile_file": str(profile_file),
        }

        return Struct(recipe)

    @classmethod
    def _check_profile(
            cls,
            profile: dict[str, Any],
            location: str
        ) -> None:

        type_checks = [
            ["metadata", "", dict],
            ["version", "metadata", str],
            ["name", "metadata", str],
            ["label", "metadata", str],
            ["task", "metadata", str],
            ["tags", "metadata", list],
            ["directive", "", dict],
            ["export", "", dict],
            ["models", "", list],
            ["corpora", "", list],
            ["cases", "", list],
        ]

        profile[""] = profile

        for key, node, type_ in type_checks:
            node_name = Q(node) if node else "root"

            if key not in profile[node]:
                raise KeyError(f"{Q(key)} key missing on {node_name} node"
                               f"in {location}")

            if not isinstance(profile[node][key], type_):
                raise ValueError(f"value of {Q(key)} key not {Q(type_)} type "
                                 f"on {node_name} node in {location}")

        profile.pop("")

        version = profile["metadata"].get("version", "").lower()

        if version != cls.version:
            raise ValueError("Profile version not matched: "
                             f"{Q(version)} in {location} != {Q(cls.version)}")

    def _update_gridsuite(self) -> Gridsuite:
        prefix = self.recipe.metadata.name
        datastore_dir = self.artifact_dir / prefix / "datastore"
        datastore_dir.mkdir(parents=True, exist_ok=True)
        label = self.recipe.metadata.label

        suite = Gridsuite(
            prefix=prefix,
            label=label,
            artifact_dir=self.artifact_dir,
            profile_file=self.recipe.profile_file,
        )

        return suite

    def _prepare_corpora(self) -> list[dict[str, str]]:
        all_corpus_args = []

        corpora = self.recipe.corpora

        for i, corpus in enumerate(corpora):
            mode = corpus["mode"][0]

            if mode in ["aggregate", "none",]:

                if mode in ["aggregate",]:
                    some_corpus_args = self._prepare_aggregate(
                        order=i + 1,
                        corpus=corpus,
                    )

                else:
                    some_corpus_args = [{
                        "corpus_mode": "none",
                        "corpus_prefix": "",
                        "corpus_stem": [],
                        "corpus_ext": [],
                    }]

                all_corpus_args.extend(some_corpus_args)
                continue

            for values in product(*corpus.values()):
                corpus_args = dict(zip(corpus.keys(), values))

                some_corpus_args = (
                    self._prepare_dataset
                    if corpus_args["mode"] in ["dataset",] else

                    self._prepare_document
                    # if corpus_args["mode"] in ["document",] else
                )(**corpus_args)

                all_corpus_args.extend(some_corpus_args)

        return all_corpus_args

    def _prepare_aggregate(
            self,
            order: int,
            corpus: dict[str, Any]
        ) -> list[dict[str, Any]]:

        if len(corpus["prefix"]) > 1:
            raise ValueError(
                f"{Q('prefix')} key has "
                "a list with more than 1 element "
                f"in {ordinal(order)} element of {Q('corpora')} key"
                )

        mode = corpus.pop("mode")[0]
        prefix = corpus.pop("prefix")[0]

        source_dir = self.source_corpus_dir / prefix

        target_dir = self.corpus_dir / prefix

        if target_dir.is_dir():
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        for values in product(*corpus.values()):
            corpus_args = dict(zip(corpus.keys(), values))

            stem = corpus_args["stem"]
            extension = corpus_args["extension"]

            for file_ in source_dir.glob(f"{stem}{extension}"):
                target = target_dir / file_.name
                shutil.copyfile(file_, target)

                self.INFO(f"File {Q(file_)} copied to {Q(target.parent)} DONE")

        return [{
            "corpus_mode": mode,
            "corpus_prefix": prefix,
            "corpus_stem": corpus["stem"],
            "corpus_ext": corpus["extension"],
        }]

    def _prepare_dataset(
            self,
            mode: str,
            prefix: str,
            stem: str,
            extension: str,
            test_size: float,
            keywords: Union[str, list[str]],
            seeds: Union[int, list[Union[int, list[int]]]]
        ) -> list[dict[str, Any]]:

        source_corpus_dir = self.source_corpus_dir / prefix 
        corpus_dir = self.corpus_dir / prefix

        seeds = self._fanout_seeds(seeds)

        test = int(test_size * 10)
        train = 10 - test

        keywords = keywords if isinstance(keywords, list) else [keywords]

        all_corpus_args = []

        for tag in ["train", "test"]:
            target_dir = corpus_dir / tag

            if target_dir.is_dir():
                shutil.rmtree(target_dir)

            target_dir.mkdir(parents=True, exist_ok=True)

        for file_ in source_corpus_dir.glob(f"{stem}{extension}"):
            records = load_json_yaml(file_)
            source_df = pd.DataFrame(records)

            for seed in seeds:
                corpus_prefix = "/".join([
                    token for token in [
                        prefix,
                        file_.stem,
                    ] if token
                ])

                corpus_prefix += f"_{train}-{test}"
                corpus_stem = f"{corpus_prefix}_{seed:02d}"

                target_df = {key: pd.DataFrame() for key in ["train", "test"]}

                for _, dataframe in source_df.groupby(keywords):
                    trainset_df, testset_df = train_test_split(
                        dataframe,
                        test_size=test_size,
                        random_state=seed,
                    )

                    target_df["train"] = pd.concat(
                        [target_df["train"], trainset_df], axis=0
                    )

                    target_df["test"] = pd.concat(
                        [target_df["test"], testset_df], axis=0
                    )

                for tag in ["train", "test"]:
                    dataframe = target_df[tag].reset_index().rename(
                        columns={"index": "sample"}
                    )

                    dataframe["sample"] = dataframe["sample"].astype(int)

                    target_dir = corpus_dir / tag
                    target_file = target_dir / f"{Path(corpus_stem).name}{extension}"
                    dataframe.to_json(target_file)

                    self.INFO(
                        f"File {Q(target_file)} copied to "
                        f"{Q(target_file.parent)} DONE"
                    )

                all_corpus_args.append({
                    "corpus_mode": mode,
                    "corpus_prefix": corpus_prefix,
                    "corpus_stem": corpus_stem,
                    "corpus_ext": extension,
                })

        return all_corpus_args

    def _fanout_seeds(
            self,
            seeds: Union[int, list[Union[int, list[int]]]]
        ) -> list[int]:

        if isinstance(seeds, int):
            return [seeds]

        targets = []

        for seed in seeds:
            if isinstance(seed, int):
                targets.append(seed)

            elif isinstance(seed, list):
                if len(seed) > 2:
                    seed[1] += seed[0]

                targets.extend(range(*seed[:3]))

        return list(set(targets))

    def _prepare_document(
            self,
            mode: str,
            prefix: str,
            stem: str,
            extension: str,
        ) -> list[dict[str, Any]]:

        source_corpus_dir = self.source_corpus_dir / prefix 

        all_corpus_args = []

        target_dir = self.corpus_dir / prefix

        if target_dir.is_dir():
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        for file_ in source_corpus_dir.glob(f"{stem}{extension}"):
            target = target_dir / file_.name
            shutil.copyfile(file_, target)

            self.INFO(f"File {Q(file_)} copied to {Q(target.parent)} DONE")

            all_corpus_args.append({
                "corpus_mode": mode,
                "corpus_prefix": prefix,
                "corpus_stem": file_.stem,
                "corpus_ext": file_.suffix,
            })

        return all_corpus_args

    def get_instance(
            self,
            case_args: dict[str, Any],
            corpus_args: dict[str, Any],
            ) -> GridRunnable:

        module = case_args.pop("module")

        class_name = to_camel(module.replace("-", "_"))
        module_name = module.replace("*", "").replace("-", "_")

        try:
            module = importlib.import_module(module_name)
            Class = getattr(module, class_name)
        
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Can't import the module {Q(module_name)}"
            )

        except AttributeError:
            raise AttributeError(
                f"Can't import the class {module_name}.{class_name}"
            )

        metadata = Struct({
            "module": module_name.replace("_", "-"),
            "suite": self.suite,
        } | self.recipe.metadata | corpus_args)

        return Class(
            metadata=metadata,
            **case_args
        )

    def yield_result(self) -> pd.DataFrame:
        self.invoke()
        return pd.DataFrame(self.generate_htmls())

    def invoke(self) -> None:
        if not self.recipe.directive["force"] and self.data_json.is_file():
            return

        self.INFO(f"{self.header} proceed ....")

        all_corpus_args = self._prepare_corpora()

        load_dotenv(self.recipe.directive.get("langsmith"))

        try:
            self._truncate()

            for case in self.recipe.cases:
                for values in product(*case.values()):
                    case_args = dict(zip(case.keys(), values))

                    for corpus_args in all_corpus_args:
                        instance = self.get_instance(
                            case_args=dict(case_args),
                            corpus_args=corpus_args,
                        )

                        instance.run(**self.recipe.directive)

                        self._save(instance.result_df, instance.metadata_df)

        except Exception:
            self._restore(True)
            raise
        else:
            self._restore(False)

        self.INFO(f"{self.header} proceed DONE")

    def _truncate(self) -> None:
        backup_file = self.data_json.with_suffix(".bak")

        if self.data_json.is_file():
            shutil.copy2(self.data_json, backup_file)

            if self.recipe.directive.get("truncate", False):
                self.data_json.unlink()

    def _restore(self, error: bool) -> None:
        backup_file = self.data_json.with_suffix(".bak")

        if backup_file.is_file():
            if error:
                self.data_json.unlink()
                shutil.copy2(backup_file, self.data_json)

            backup_file.unlink()

    def _save(self,
              result_df: pd.DataFrame,
              metadata_df: pd.DataFrame
              ) -> None:
        self.INFO(f"{self.header} DATA saved ....")

        if self.data_json.is_file():
            old_result_df, old_metadata_df, *_ = from_json(self.data_json)
            columns = ["suite_id", "case_id"]

            if not old_result_df.empty:
                old_result_df = old_result_df.set_index(columns)
                result_df = result_df.set_index(columns)

                result_df = (
                    result_df.combine_first(old_result_df).reset_index()
                )

            if not old_metadata_df.empty:
                old_metadata_df = old_metadata_df.set_index(columns)
                metadata_df = metadata_df.set_index(columns)

                metadata_df = (
                    metadata_df.combine_first(old_metadata_df).reset_index()
                )

        to_json(self.data_json, result_df, metadata_df)

        self.INFO(f"{self.header} DATA saved DONE")

    def export_tables(self) -> tuple[Path, list[str]]:
        self.INFO(f"{self.header} TABLE EXPORT proceed ....")

        export = self.recipe.export_table

        if not export.pop("enable", False) or not self.data_json.is_file():
            self.INFO(f"{self.header} TABLE EXPORT proceed DONE")
            return self.export_dir

        result_df, metadata_df, *_ = from_json(self.data_json)

        stem = optional(export.pop("stem", None), self.suite_id)
        columns = optional(export.pop("columns", None), [])
        extensions = optional(export.pop("extension", None), [])

        export_df = pd.merge(
            result_df,
            metadata_df,
            on=["suite_id", "case_id"],
            how="outer",
        )

        columns = [
            column for column in columns
            if column in export_df.columns
        ] if columns else (
            export_df.columns
        )

        dup_columns = [
            column for column in columns
            if column not in [
                "summary_pv",
                "summary_gt",
                "corpus_stem",
                "corpus_ext",
                "html_args",
            ]
        ]

        export_df = export_df[columns].drop_duplicates(dup_columns)

        if ".csv" in extensions:
            target = self.export_dir / f"{stem}.csv"

            export_df.to_csv(
                target,
                index=False,
                encoding="utf-8",
            )
            self.INFO(f"File {Q(target)} saved Done")

        if ".json" in extensions:
            target = self.export_dir / f"{stem}.json"

            export_df.to_json(
                target,
                orient="records",
                force_ascii=False,
            )
            self.INFO(f"File {Q(target)} saved Done")

        if ".xlsx" in extensions:
            target = self.export_dir / f"{stem}.xlsx"

            with pd.ExcelWriter(target) as writer:
                export_df.to_excel(writer)

            self.INFO(f"File {Q(target)} saved Done")

        self.INFO(f"{self.header} TABLE EXPORT proceed DONE")
        return self.export_dir, extensions

    def generate_htmls(self) -> pd.DataFrame:
        self.INFO(f"{self.header} HTML GENERATE proceed ....")

        if self.data_json.is_file():
            result_df, *_ = from_json(self.data_json)
        else:
            result_df = pd.DataFrame()

        if not result_df.empty:
            result_df["html"] = result_df.apply(
                self._generate_html_foreach,
                count=itertools.count(1),
                total=result_df.shape[0],
                axis=1
            )
        else:
            result_df["html"] = get_empty_html()

        self.INFO(f"{self.header} HTML GENERATE proceed DONE")
        return result_df

    def _generate_html_foreach(
            self,
            result: pd.Series,
            count: Callable[..., Iterator[int]],
            total: int
        ) -> pd.Series:
        if pd.isna(result.html_file):
            html = None

            self.INFO(
                f"{self.header} HTML [{next(count):02d}/{total:02d}] SKIP"
            )

        else:
            html_file = Path(result.html_file)
            html_text = html_file.read_text()

            css_file = Path(result.css_file)
            css_content = f"<style>\n{css_file.read_text()}</style>"

            html_args = result.html_args | {
                "css_content": css_content,
            }

            html = html_text.format(**html_args)

            self.INFO(
                f"{self.header} HTML [{next(count):02d}/{total:02d}] DONE"
            )

        return pd.Series({"html": html})

    def export_documents(self) -> tuple[Path, list[str]]:
        self.INFO(f"{self.header} DOCUMENT EXPORT proceed ....")

        export = self.recipe.export_document

        if not export.pop("enable", False) or not self.data_json.is_file():
            self.INFO(f"{self.header} DOCUMENT EXPORT proceed DONE")
            return

        keywords = export.pop("keywords", [])
        extensions = export.pop("extension", [])

        result_df = self.generate_htmls()

        if result_df.empty:
            self.INFO(f"{self.header} DOCUMENT EXPORT proceed SKIP")
            return self.export_dir

        if keywords:
            total = result_df.groupby(
                keywords, sort=False
            ).html.size().shape[0]

            result_df.groupby(keywords, sort=False).html.apply(
                self._export_document_foreach,
                extensions=extensions,
                count=itertools.count(1),
                total=total,
            )
        else:
            result_df = result_df[result_df.html.notna()]

            self._export_document_foreach(
                result_df.html,
                count=itertools.count(1),
                total=1,
            )

        self.INFO(f"{self.header} DOCUMENT EXPORT proceed DONE")
        return self.export_dir, extensions

    def _export_document_foreach(
            self,
            htmls: pd.Series,
            extensions: list[str],
            count: Callable[..., Iterator[int]],
            total: int
        ) -> None:

        if htmls.name == "html":
            suffix = ""
        else:
            parts = list(htmls.name) if isinstance(htmls.name, tuple) else [htmls.name]
            suffix = "-".join([f"{part}" for part in parts])

        stem = "_".join([part for part in [self.suite_id, suffix] if part])

        if ".pdf" in extensions:
            pdf_file = self.export_dir / f"{stem}.pdf"
            self.INFO(f"*{Q(pdf_file)} DOCUMENT EXPORT ....")

            html_to_pdf(htmls, pdf_file)

            self.INFO(f"*{Q(pdf_file)} DOCUMENT EXPORT DONE")

        if ".docx" in extensions:
            docx_file = self.export_dir / f"{stem}.docx"
            self.INFO(f"*{Q(docx_file)} DOCUMENT EXPORT ....")

            html_to_docx(htmls, docx_file)

            self.INFO(f"*{Q(docx_file)} DOCUMENT EXPORT DONE")

        self.INFO(
            f"{self.header} DOCUMENT EXPORT "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

    @property
    def recipe(self) -> Struct:
        return self._recipe

    @property
    def suite(self) -> Gridsuite:
        return self._suite

    @property
    def suite_id(self) -> str:
        return self.suite.id

    @property
    def header(self) -> str:
        return f"SUITE {Q(self.suite_id)}"

    @property
    def source_corpus_dir(self) -> Path:
        return self._source_corpus_dir

    @property
    def html_dir(self) -> Path:
        return self._html_dir

    @property
    def datastore_dir(self) -> Path:
        return self._datastore_dir

    @property
    def corpus_dir(self) -> Path:
        return self._corpus_dir

    @property
    def export_dir(self) -> Path:
        return self._export_dir

    @property
    def data_json(self) -> Path:
        return self.datastore_dir / f"{self.suite_id}.json"

