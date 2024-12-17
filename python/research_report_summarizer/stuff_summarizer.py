from typing import Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import load_prompt

from research_report_template import (
    ResearchReportItemFail,
    ResearchReportItemSummary, ResearchReportItemKeyword
)

from research_report_summarizer import ResearchReportSummarizer


class StuffSummarizer(ResearchReportSummarizer):
    def __init__(self,
                 models: Sequence[str],
                 num_keywords: int,
                 max_tokens: int,
                 extract_header_prompt: str,
                 extract_summary_prompt: str,
                 summarize_prompt: str,
                 keyword_prompt: str,
                 similarity_prompt: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            summarize_prompt.split(".")[0],
            keyword_prompt.split(".")[0],
        ])

        super().__init__(
            models=models,
            num_keywords=num_keywords,
            max_tokens=max_tokens,
            extract_header_prompt=extract_header_prompt,
            extract_summary_prompt=extract_summary_prompt,
            similarity_prompt=similarity_prompt,
            case_suffix=case_suffix,
            metadata=metadata,
        )

        self.INFO(f"{self.header} prepared ....")

        self.metadata_df.loc[
            0, 
            [
                "summarize_prompt",
                "keyword_prompt",
            ]] = [
                summarize_prompt,
                keyword_prompt,
            ]

        self.main_chain = self._build_main_chain(
            summarize_prompt=summarize_prompt,
            keyword_prompt=keyword_prompt,
        )

        self.INFO(f"{self.header} prepared DONE")

    def _build_main_chain(self,
                          summarize_prompt: str,
                          keyword_prompt: str
                          ) -> Runnable:
        self.INFO(f"{self.header} MAIN CHAIN prepared ....")

        branches = [
            Struct(
                {
                    "pydantic_object": ResearchReportItemSummary,
                    "prompt": summarize_prompt,
                    "condition": lambda x: x["item"] not in ["keyword",],
                }
            ),
            Struct(
                {
                    "pydantic_object": ResearchReportItemKeyword,
                    "prompt": keyword_prompt,
                    "condition": lambda x: x["item"] in ["keyword",],
                }
            ),
        ]

        statements = []

        for branch in branches:
            parser = PydanticOutputParser(
                pydantic_object=branch.pydantic_object,
            )

            prompt_args, partials = load_prompt(self.content_dir / branch.prompt)

            full_args = prompt_args | {
                "partial_variables": {
                    partials[0]: parser.get_format_instructions(),
                },
            }

            prompt = PromptTemplate(**full_args)

            statements.append((branch.condition, prompt | self.model[0] | parser))

        statements.append(
            RunnableLambda(lambda _: ResearchReportItemFail(
                message="Invalid item",
            ))
        )

        chain = RunnableBranch(*statements)

        self.INFO(f"{self.header} MAIN CHAIN prepared DONE")
        return chain
