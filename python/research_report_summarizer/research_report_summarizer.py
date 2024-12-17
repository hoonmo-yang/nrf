from __future__ import annotations
from typing import Callable, Iterator, Optional, Union

import itertools
import json
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from cenai_core import Timer

from cenai_core.dataman import (
    load_json_yaml, optional, pad_list, Q, QQ, Struct
)

from cenai_core.grid import GridRunnable

from cenai_core.langchain_helper import (
    LineTextSplitter, load_documents, load_prompt
)

from research_report_template import (
    ResearchReportItemFail, ResearchReportIdentity,
    ResearchReportSummaryTemplate, ResearchReportSimilarity
)


class ResearchReportSummarizer(GridRunnable):
    logger_name = "cenai.research_report_summarizer"

    def __init__(self,
                 models: list[str],
                 num_keywords: int,
                 max_tokens: int,
                 extract_header_prompt: str,
                 extract_summary_prompt: str,
                 similarity_prompt: str,
                 case_suffix: str,
                 metadata: Struct
                 ):
        
        case_suffix = "_".join([
            extract_header_prompt.split(".")[0],
            extract_summary_prompt.split(".")[0],
            similarity_prompt.split(".")[0],
            case_suffix,
            f"kw{num_keywords}",
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

        self._num_keywords = num_keywords
        self._max_tokens = max_tokens

        self.metadata_df.loc[
            0,
            [
                "extract_header_prompt",
                "extract_summary_prompt",
                "similarity_prompt",
                "num_keywords",
                "max_tokens",
             ]
        ] = [
            extract_header_prompt,
            extract_summary_prompt,
            similarity_prompt,
            num_keywords,
            max_tokens,
        ]

        self._layout = self._get_layout()

        self._extract_header_chain = self._build_extract_header_chain(
            extract_header_prompt=extract_header_prompt,
        )

        self._extract_summary_chain = self._build_extract_summary_chain(
            extract_summary_prompt=extract_summary_prompt,
        )

        self._similarity_chain = self._build_similarity_chain(
            similarity_prompt=similarity_prompt,
        )

        self._css_file = self.html_dir / "styles.css"
        self._html_file = self.html_dir / "html_template.html"

    def _build_extract_header_chain(self,
                                    extract_header_prompt: str,
                                    ) -> Runnable:
        self.INFO(f"{self.header} EXTRACT-HEADER CHAIN prepared ....")

        parser = PydanticOutputParser(
            pydantic_object=ResearchReportIdentity,
        )

        prompt_args, partials = load_prompt(
            self.content_dir / extract_header_prompt
        )

        full_args = prompt_args | {
            "partial_variables": {
                partials[0]: parser.get_format_instructions(),
            },
        }

        prompt = PromptTemplate(**full_args)

        chain = prompt | self.model[1] | parser

        self.INFO(f"{self.header} EXTRACT-HEADER CHAIN prepared DONE")
        return chain

    def _build_extract_summary_chain(self,
                                     extract_summary_prompt: str,
                                     ) -> Runnable:
        self.INFO(f"{self.header} EXTRACT-SUMMARY CHAIN prepared ....")

        parser = PydanticOutputParser(
            pydantic_object=ResearchReportSummaryTemplate,
        )

        prompt_args, partials = load_prompt(
            self.content_dir / extract_summary_prompt
        )

        full_args = prompt_args | {
            "partial_variables": {
                partials[0]: parser.get_format_instructions(),
            },
        }

        prompt = PromptTemplate(**full_args)

        chain = prompt | self.model[1] | parser

        self.INFO(f"{self.header} EXTRACT-SUMMARY CHAIN prepared DONE")
        return chain

    def _build_similarity_chain(self,
                                similarity_prompt: str,
                                ) -> Runnable:
        self.INFO(f"{self.header} SIMILARITY CHAIN prepared ....")

        parser = PydanticOutputParser(
            pydantic_object=ResearchReportSimilarity,
        )

        prompt_args, partials = load_prompt(
            self.content_dir / similarity_prompt
        )

        full_args = prompt_args | {
            "partial_variables": {
                partials[0]: parser.get_format_instructions(),
            },
        }

        prompt = PromptTemplate(**full_args)

        chain = prompt | self.model[1] | parser

        self.INFO(f"{self.header} SIMILARITY CAHIN prepared DONE")
        return chain

    def _get_layout(self) -> Struct:
        layout_file = self.content_dir / "report-layout.yaml"
        layout = load_json_yaml(layout_file)

        return Struct({
            "source": layout["source_template"],
            "summary": layout["summary_template"],
        })

    def run(self, **directive) -> None:
        self._summarize(**directive)

    def _summarize(
            self,
            num_tries: Optional[int] = None,
            recovery_time: Optional[int] = None,
            **kwargs
            ) -> None:

        self.INFO(f"{self.header} REPORT SUMMARY proceed ....")

        num_tries = optional(num_tries, 5)
        recovery_time = optional(recovery_time, 0.5)

        report_df = self._split_reports_by_section()

        summary_pv_df = report_df.pipe(
            self._compile_reports_by_section,
        ).pipe(
            self._summarize_reports,
            num_tries=num_tries,
            recovery_time=recovery_time,
        )

        summary_gt_df = report_df[
            report_df.section == "summary"
        ].pipe(
            self._extract_report_summaries,
            num_tries=num_tries,
            recovery_time=recovery_time,
        )

        header_df = report_df[
            report_df.section == "header"
        ].pipe(
            self._extract_report_headers,
            num_tries=num_tries,
            recovery_time=recovery_time,
        )

        self.result_df = pd.merge(
            summary_pv_df,
            summary_gt_df,
            on=["file"],
            how="outer",
            suffixes=["_pv", "_gt"],
        ).pipe(
            pd.merge,
            header_df,
            on=["file"],
            how="outer",
        ).pipe(
            self._calculate_similarity,
            num_tries=num_tries,
            recovery_time=recovery_time,
        ).pipe(
            self._prepare_htmls,
        )

        self.INFO(f"{self.header} REPORT SUMMARY proceed DONE")

    def _split_reports_by_section(self) -> pd.DataFrame:
        self.INFO(f"{self.header} REPORT SPLIT proceed ....")

        splitter = LineTextSplitter(chunk_size=50)
        input_df = self.document_df

        stage_df = input_df.apply(
            self._split_report_by_section_foreach,
            splitter=splitter,
            count=itertools.count(1),
            total=input_df.shape[0],
            axis=1
        ).explode(
            ["document", "section"],
        ).reset_index(
            drop=True,
        )
        
        total = stage_df.groupby(
            ["file", "section"], sort=False
        ).size().shape[0]

        output_df = stage_df.groupby(
            ["file", "section"], sort=False,
        )["document"].apply(
            self._merge_report_by_section_foreach,
            count=itertools.count(1),
            total=total,
        ).reset_index()

        self.INFO(f"{self.header} REPORT SPLIT proceed DONE")
        return output_df

    def _split_report_by_section_foreach(
            self,
            report: pd.Series,
            splitter: LineTextSplitter,
            count: Callable[..., Iterator[int]],
            total: int
        ) -> pd.Series:
        file_ = report.file

        documents = load_documents(file_)
        content = "\n".join([document.page_content for document in documents])

        merged_documents = [
            Document(
                page_content=content,
                metadata={"source": str(file_)}
            )
        ]

        split_documents = splitter.split_documents(merged_documents)
        documents, sections = self._annotate_report_by_section(split_documents)

        self.INFO(
            f"{self.header} REPORT SPLIT FILE {Q(file_.name)} "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return pd.Series({
            "file": file_,
            "section": sections,
            "document": documents,
        })

    def _annotate_report_by_section(
            self,
            documents: list[Document]
        ) -> tuple[list[Document], list[str]]:

        section = ""

        targets = []
        sections = []

        for document in documents:
            target, section = self._annotate_document_by_section(
                document, section
            )

            targets.append(target)
            sections.append(section)

        return targets, sections

    def _annotate_document_by_section(
            self,
            document: Document,
            section: str
        ) -> tuple[Document, str]:

        sections = list(self.layout.source.keys())
        titles = list(self.layout.source.values())

        results = process.extract(
            document.page_content,
            titles,
            scorer=fuzz.partial_ratio,
        )

        _, score, k = results[0]
        section = sections[k] if score > 80.0 else section
        document.metadata["section"] = section

        return document, section

    def _merge_report_by_section_foreach(
            self,
            document: pd.Series,
            count: Callable[..., Iterator[int]],
            total: int
        ) -> pd.Series:
        file_, section = document.name

        page_content = "\n".join(
            document.apply(lambda document: document.page_content)
        )

        self.INFO(
            f"{self.header} REPORT MERGE FILE {Q(file_.name)} "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return pd.Series({
            "document": Document(
                            page_content=page_content,
                            metadata={
                                "file": str(file_),
                                "section": section,
                            },
                        ),
        })

    def _compile_reports_by_section(
            self,
            report_df: pd.DataFrame
        ) -> pd.DataFrame:

        self.INFO(f"{self.header} REPORT COMPILE proceed ....")

        total = report_df.groupby(["file"], sort=False).size().shape[0]

        output_df = report_df.groupby(["file"], sort=False)[
            ["section", "document"]
        ].apply(
            self._compile_report_by_section_foreach,
            count=itertools.count(1),
            total=total,
        ).explode(
            ["item", "title", "content",]
        ).reset_index()

        self.INFO(f"{self.header} REPORT COMPILE proceed DONE")
        return output_df

    def _compile_report_by_section_foreach(
            self,
            report_df: pd.DataFrame,
            count: Callable[..., Iterator[int]],
            total: int
        ) -> pd.Series:
        file_ = report_df.name

        items = []
        titles = []
        contents = []

        for item, record in self.layout.summary.items():
            title, sections = [
                record[key] for key in ["title", "sections"]
            ]

            page_contents = pd.Series()

            for section in sections:
                some_page_contents = report_df[
                    report_df.section == section
                ].document.apply(
                    lambda document: document.page_content
                )

                page_contents = pd.concat(
                    [page_contents, some_page_contents]
                )

            items.append(item)
            titles.append(title)
            contents.append("\n".join(page_contents))

            self.INFO(
                f"** ITEM:{Q(item)} TITLE:{Q(title)} "
                f"SECTIONS: [{','.join(sections)}]"
            )

        entry = pd.Series({
            "item": items,
            "title": titles,
            "content": contents,
        })

        self.INFO(
            f"{self.header} REPORT COMPILE FILE {Q(file_.name)} "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return entry

    def _summarize_reports(self,
                           report_df: pd.DataFrame,
                           num_tries: int,
                           recovery_time: int
                           ) -> pd.DataFrame:

        self.INFO(f"{self.header} REPORT SUMMARIZATION proceed ....")

        total = report_df.groupby(["file"], sort=False).size().shape[0]

        summary_df = report_df.groupby(["file"], sort=False)[
            ["item", "title", "content"]
        ].apply(
            self._summarize_report_foreach,
            count=itertools.count(1),
            total=total,
            num_tries=num_tries,
            recovery_time=recovery_time,
        ).reset_index()
        
        self.INFO(f"{self.header} REPORT SUMMARIZATION proceed DONE")

        return summary_df

    def _summarize_report_foreach(self,
                                  report_df: pd.DataFrame,
                                  count: Callable[..., Iterator[int]],
                                  total: int,
                                  num_tries: int,
                                  recovery_time: int
                                  ) -> pd.Series:
        file_ = report_df.name
        total2 = report_df.groupby(["item", "title"], sort=False).size().shape[0]

        timer = Timer()

        items = report_df.groupby(["item", "title"], sort=False)["content"].apply(
            self._summarize_report_item_foreach,
            file_=file_,
            count=itertools.count(1),
            total=total2,
            num_tries=num_tries,
            recovery_time=recovery_time,
        ).reset_index().to_dict(orient="records")

        summary = {
            item["item"]: {
                "title": item["title"],
                "content": item["content"],
            } for item in items
        }

        timer.lap()

        self.INFO(
            f"{self.header} REPORT SUMMARIZATION "
            f"FILE {Q(file_.name)} TIME {timer.seconds:.1f}s "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return pd.Series({
            "summary": summary,
            "time": timer.seconds,
        })

    def _summarize_report_item_foreach(
            self,
            content: pd.Series,
            file_: Path,
            count: Callable[..., Iterator[int]],
            total: int,
            num_tries: int,
            recovery_time: int
        ) -> Union[str, list[list[str]]]:

        item, title = content.name

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.main_chain.invoke(
                    input={
                        "item": item,
                        "title": title,
                        "content": content.iloc[0],
                        "num_keywords": self._num_keywords,
                        "max_tokens": self._max_tokens,
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

            response = ResearchReportItemFail(
                message="LLM internal error during summarization",
            )

        timer.lap()

        if not response.error:
            summary = (
                [response.keyword_kr, response.keyword_en]
                if item == "keyword" else

                response.summary
            )

        else:
            summary = [[], []] if item == "keyword" else ""

        self.INFO(
            f"{self.header} REPORT SUMMARIZATION "
            f"FILE {Q(file_.name)} ITEM: {Q(item)} TIME {timer.seconds:.1f}s "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return summary

    def _extract_report_summaries(
            self,
            report_df: pd.DataGrame,
            num_tries: int,
            recovery_time: int
        ) -> pd.DataFrame:

        self.INFO(f"{self.header} REPORT SUMMARY EXTRACTION proceed ....")

        total = report_df.groupby(["file"], sort=False).size().shape[0]

        summary_df = report_df.groupby(["file"], sort=False)[["document"]].apply(
            self._extract_report_summary_foreach,
            count=itertools.count(1),
            total=total,
            num_tries=num_tries,
            recovery_time=recovery_time,
        ).reset_index()

        self.INFO(f"{self.header} REPORT SUMMARY EXTRACTION proceed DONE")

        return summary_df

    def _extract_report_summary_foreach(
            self,
            report_df: pd.DataFrame,
            count: Callable[..., Iterator[int]],
            total: int,
            num_tries: int,
            recovery_time: int
        ) -> pd.Series:

        file_ = report_df.name

        content = "\n".join(
            report_df.document.apply(lambda document: document.page_content)
        )

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.extract_summary_chain.invoke(
                    input={
                        "content": content,
                    },
                )

            except KeyboardInterrupt:
                raise

            except BaseException:
                self.ERROR(f"LLM({self.model[1].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
                recovery_time *= 2
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            response = ResearchReportSummaryTemplate(
                abstract="",
                outcome="",
                expectation="",
                keyword_kr=[],
                keyword_en=[],
            )

        summary = self._generate_summary(response)

        timer.lap()

        self.INFO(
            f"{self.header} REPORT SUMMARY EXTRACTION "
            f"FILE {Q(file_.name)} TIME {timer.seconds:.1f}s "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return pd.Series({
            "summary": summary,
            "time": timer.seconds
        })

    def _generate_summary(self,
                          summary: ResearchReportSummaryTemplate
                          ) -> dict[str, dict[str, str]]:
        summary_dict = summary.model_dump()

        summary_dict["keyword"] = []

        for name in ["keyword_kr", "keyword_en"]:
            summary_dict["keyword"].append(summary_dict[name])
            summary_dict.pop(name)

        new_summary =  {
            key: {
                "title": self.layout.summary[key]["title"],
                "content": value,
            } for key, value in summary_dict.items()
        }

        return new_summary


    def _extract_report_headers(self,
                                report_df: pd.DataFrame,
                                num_tries: int,
                                recovery_time: int
                                ) -> pd.DataFrame:
        self.INFO(f"{self.header} REPORT HEADER EXTRACTION proceed ....")

        total = report_df.groupby(["file"], sort=False).size().shape[0]

        header_df = report_df.groupby(["file"], sort=False)[["document"]].apply(
            self._extract_report_header_foreach,
            count=itertools.count(1),
            total=total,
            num_tries=num_tries,
            recovery_time=recovery_time,
        ).reset_index()

        self.INFO(f"{self.header} REPORT HEADER EXTRACTION proceed DONE")

        return header_df

    def _extract_report_header_foreach(
            self,
            report_df: pd.DataFrame,
            count: Callable[..., Iterator[int]],
            total: int,
            num_tries: int,
            recovery_time: int
        ) -> pd.Series:

        file_ = report_df.name

        content = "\n".join(
            report_df.document.apply(lambda document: document.page_content)
        )

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.extract_header_chain.invoke(
                    input={
                        "content": content,
                    },
                )

            except KeyboardInterrupt:
                raise

            except BaseException:
                self.ERROR(f"LLM({self.model[1].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
                recovery_time *= 2
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            response = ResearchReportIdentity(
                title_kr = "",
                title_en = "",
                institution = "",
                name = "",
                position = "",
                department = "",
                major = "",
            )

        timer.lap()

        entry = pd.Series({
            "title_kr": response.title_kr,
            "title_en": response.title_en,
            "institution": response.institution,
            "name": response.name,
            "position": response.position,
            "department": response.department,
            "major": response.major,
        })

        self.INFO(
            f"{self.header} REPORT HEADER EXTRACTION "
            f"FILE {Q(file_.name)} TIME {timer.seconds:.1f}s "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return entry

    def _calculate_similarity(self,
                              report_df: pd.DataFrame,
                              num_tries: int,
                              recovery_time: int
                              ) -> pd.DataFrame:
        self.INFO(f"{self.header} REPORT SIMILARITY proceed ....")

        columns = [
            "abstract_score",
            "abstract_difference",
            "outcome_score",
            "outcome_difference",
            "expectation_score",
            "expectation_difference",
            "keyword_score",
            "keyword_difference",
        ]

        report_df[columns] = report_df.apply(
            self._calculate_similarity_foreach,
            count=itertools.count(1),
            total=report_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            axis=1
        )

        self.INFO(f"{self.header} REPORT SIMILARITY proceed DONE")

        return report_df

    def _calculate_similarity_foreach(self,
                                      report: pd.Series,
                                      count: Callable[..., Iterator[int]],
                                      total: int,
                                      num_tries: int,
                                      recovery_time: int
                                      ) -> pd.Series:
        file_ = report.file

        summary_pv = json.dumps(report.summary_pv, ensure_ascii=False)
        summary_gt = json.dumps(report.summary_gt, ensure_ascii=False)
        
        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.similarity_chain.invoke(
                    input={
                        "summary_pv": summary_pv,
                        "summary_gt": summary_gt,
                    },
                )

            except KeyboardInterrupt:
                raise

            except BaseException:
                self.ERROR(f"LLM({self.model[1].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
                recovery_time *= 2
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            response = ResearchReportSimilarity(
                abstract_score=0,
                abstract_difference="",
                outcome_score=0,
                outcome_difference="",
                expectation_score=0,
                expectation_difference="",
                keyword_score=0,
                keyword_difference="",
            )

        timer = Timer()

        entry = pd.Series({
            "abstract_score": response.abstract_score,
            "abstract_difference": response.abstract_difference,
            "outcome_score": response.outcome_score,
            "outcome_difference": response.outcome_difference,
            "expectation_score": response.expectation_score,
            "expectation_difference": response.expectation_difference,
            "keyword_score": response.keyword_score,
            "keyword_difference": response.keyword_difference,
        })

        self.INFO(
            f"{self.header} REPORT SIMILARITY "
            f"FILE {Q(file_.name)} TIME {timer.seconds:.1f}s "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return entry

    def _prepare_htmls(self, report_df: pd.DataFrame) -> pd.DataFrame:

        self.INFO(f"{self.header} REPORT HTML PREPARATION proceed ....")

        columns = [
            "file",
            "css_file",
            "html_file",
            "html_args",
        ]

        report_df[columns] = report_df.apply(
            self._prepare_html_foreach,
            count=itertools.count(1),
            total=report_df.shape[0],
            axis=1
        )

        columns = [
            "suite_id",
            "case_id",
        ]
        report_df[columns] = [self.suite_id, self.case_id]

        self.INFO(f"{self.header} REPORT HTML PREPARATION proceed DONE")
        return report_df

    def _prepare_html_foreach(self,
                              report: pd.Series,
                              count: Callable[..., Iterator[int]],
                              total: int,
                              ) -> pd.Series:
        file_ = report.file

        html_args = {
            "file": str(file_.relative_to(self.corpus_dir)),
        } | {
            key: report[key] for key in [
                "title_kr",
                "title_en",
                "institution",
                "name",
                "position",
                "department",
                "major",
                "abstract_score",
                "abstract_difference",
                "outcome_score",
                "outcome_difference",
                "expectation_score",
                "expectation_difference",
                "keyword_score",
                "keyword_difference",
            ]
        }

        summary_duo = {
            "gt": report.summary_gt,
            "pv": report.summary_pv,
        }

        for key, summary in summary_duo.items():
            for item, record in summary.items():
                title, content = [
                    record[key] for key in ["title", "content"]
                ]

                args = {f"{item}_title": title}

                if item == "keyword":
                    keyword_text = self._decorate_keywords(content)

                    args |= {
                        f"keyword_kr_{key}": keyword_text[0],
                        f"keyword_en_{key}": keyword_text[1],
                    }
                else:
                    args |= {f"{item}_content_{key}": content,}

                html_args |= args

        entry = pd.Series({
            "file": str(file_),
            "css_file": str(self._css_file),
            "html_file": str(self._html_file),
            "html_args": html_args,
        })

        self.INFO(
            f"{self.header} REPORT HTML {Q(file_.name)} "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return entry

    def _decorate_keywords(self,
                           all_keywords: list[list[str]]
                           ) -> list[str]:
        all_keywords = [
            pad_list(keywords, self._num_keywords)
            for keywords in all_keywords
        ]

        return [
            f"<td style={QQ('text-align: justify;')}>{', '.join(keywords)}</td>"
            for keywords in all_keywords
        ]

    @property
    def layout(self) -> Struct:
        return self._layout

    @property
    def extract_header_chain(self) -> Runnable:
        return self._extract_header_chain

    @property
    def extract_summary_chain(self) -> Runnable:
        return self._extract_summary_chain

    @property
    def similarity_chain(self) -> Runnable:
        return self._similarity_chain
