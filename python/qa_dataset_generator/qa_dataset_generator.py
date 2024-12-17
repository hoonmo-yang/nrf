from __future__ import annotations
from typing import Callable, Iterator, Optional, Sequence

import itertools
import pandas as pd
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from cenai_core import Timer

from cenai_core.dataman import (
    divide_evenly, proportionalize, Q, Struct
)

from cenai_core.grid import GridRunnable
from cenai_core.langchain_helper import get_document_length, load_documents


class QADatasetGenerator(GridRunnable):
    logger_name = "cenai.qa_dataset_generator"

    P_RESPONSE = re.compile(
        r"문제\s*:(.*?)\s정답\s*:(.*?)(?=\s*문제\s*:|$)", re.S
    )

    def __init__(self,
                 models: Sequence[str],
                 chunk_size: int,
                 chunk_overlap: int,
                 num_datasets: int,
                 max_tokens: int,
                 case_suffix: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            case_suffix,
            f"cs{chunk_size}",
            f"co{chunk_overlap}",
            f"n{num_datasets}",
            f"tk{max_tokens}",
        ])

        corpus_stem = metadata.corpus_stem
        corpus_ext = metadata.corpus_ext

        if isinstance(corpus_stem, str):
            corpus_stem = [corpus_stem]

        if isinstance(corpus_ext, str):
            corpus_ext = [corpus_ext]

        corpus_part = "_".join([
            metadata.corpus_prefix,
            "-".join(
                [stem for stem in corpus_stem if stem]
            ),
            "-".join([
                extension[1:] for extension in corpus_ext
                if extension]),
        ])

        super().__init__(
            models=models,
            case_suffix=case_suffix,
            corpus_part=corpus_part,
            metadata=metadata,
        )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._num_datasets = num_datasets
        self._max_tokens = max_tokens

        self.metadata_df.loc[
            0, 
            [
                "chunk_size",
                "chunk_overlap",
                "num_datasets",
                "max_tokens",
            ]
        ] = [
            chunk_size,
            chunk_overlap,
            num_datasets,
            max_tokens,
        ]

    def run(self, **directive) -> None:
        self._generate_qa_dataset(**directive)

    def _generate_qa_dataset(
            self,
            num_tries: Optional[int] = None,
            recovery_time: Optional[int] = None,
            **kwargs
            ) -> None:

        self.INFO(f"{self.header} QA-DATASET GENERATE proceed ....")

        document_df = self.document_df.copy()

        document_df["length"] = document_df.apply(
            lambda field: get_document_length(field.file),
            axis=1
        )

        document_df["num_per_file"] = proportionalize(
            self._num_datasets, document_df["length"]
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        columns = ["num_per_chunk", "document"]

        document_df[columns] = document_df.apply(
            self._split_document_foreach,
            count=itertools.count(1),
            total=document_df.shape[0],
            splitter=splitter,
            axis=1
        )
        
        document_df = document_df.explode(columns).reset_index( drop=True)

        columns = ["문제", "정답"]

        self.result_df = document_df.apply(
            self._generate_qa_dataset_foreach,
            count=itertools.count(1),
            total=document_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            axis=1
        ).dropna(how="all").explode(columns).reset_index(drop=True)

        self.result_df[["suite_id", "case_id"]] = [
            self.suite_id, self.case_id
        ]

        self.INFO(f"{self.header} QA-DATASET GENERATE proceed DONE")

    def _split_document_foreach(self,
                                field: pd.Series,
                                count: Callable[..., Iterator[int]],
                                total: int,
                                splitter: TextSplitter
                                ) -> pd.Series:

        documents = load_documents(field.file)

        split_documents = splitter.split_documents(
            documents=documents,
        )

        total = len(split_documents)
        split_sizes = divide_evenly(field.num_per_file, total)

        self.INFO(
            f"load and split documents from FILE {Q(field.file)} DONE"
            f"[{next(count):02d}/{total:02d}] proceed DONE"
        )

        return pd.Series({
            "num_per_chunk": split_sizes,
            "document": split_documents,
        })

    def _generate_qa_dataset_foreach(
        self,
        field: pd.Series,
        count: Callable[..., Iterator[int]],
        total: int,
        num_tries: int,
        recovery_time: int,
    ) -> pd.Series:

        num_per_chunk = field.num_per_chunk
        document = field.document

        if num_per_chunk == 0:
            self.INFO(
                f"FILE {Q(field.file)} DOCUMENT [{next(count)}/{total}] "
                f"SIZE {field.num_per_chunk} SKIP"
            )
            return pd.Series()

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.main_chain.invoke(
                    input={
                        "num_datasets": num_per_chunk,
                        "max_tokens": self._max_tokens,
                        "context": document.page_content,
                    },
                    config=self.chain_config,
                )
            except KeyboardInterrupt:
                raise

            except BaseException:
                self.ERROR(f"LLM({self.model[0].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
                recovery_time *= 2
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            self.INFO(
                f"FILE {Q(field.file)} DOCUMENT [{next(count)}/{total}] "
                f"SIZE {field.num_per_chunk} SKIP"
            )

            return pd.Series()

        timer.lap()

        questions, answers = self._parse_response(response, num_per_chunk)

        entry = pd.Series({
            "문제": questions,
            "정답": answers,
            "file": str(field.file),
            "time": timer.seconds / len(questions),
        })

        self.INFO(
            f"FILE {Q(field.file)} DOCUMENT [{next(count)}/{total}] "
            f"SIZE {field.num_per_chunk} TIME {timer.seconds:.1f} sec DONE"
        )
        return entry

    def _parse_response(self,
                        response: str,
                        num: int
                        ) -> tuple[list, list]:
        questions = []
        answers = []

        if not response:
            return questions, answers

        matches = self.P_RESPONSE.findall(response)

        for match in matches:
            questions.append(match[0].strip())
            answers.append(match[1].strip())

        if len(questions) != num:
            self.WARNING(
                f"numbers of generated Q&A: {Q(len(questions))} "
                f"not matched to {Q(num)}"
            )

        return questions, answers
