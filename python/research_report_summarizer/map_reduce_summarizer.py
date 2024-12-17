from typing import Annotated, Literal, Sequence, TypedDict

from operator import add, itemgetter

from langchain.chains.combine_documents import reduce
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnableMap
from langchain.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import load_prompt

from research_report_summarizer import (
    ResearchReportSummarizer, ResearchReportItemSummary,
    ResearchReportItemKeyword, ResearchReportItemFail
)


class OverallState(TypedDict):
    contents: list[str]
    summaries: Annotated[list, add]
    collapsed_summaries: list[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str


class MapReduceSummarizer(ResearchReportSummarizer):
    def __init__(self,
                 models: Sequence[str],
                 num_keywords: int,
                 max_tokens: int,
                 max_map_reduce_tokens: int,
                 extract_header_prompt: str,
                 extract_summary_prompt: str,
                 summarize_prompt: str,
                 keyword_prompt: str,
                 map_reduce_prompt: str,
                 similarity_prompt: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            summarize_prompt.split(".")[0],
            keyword_prompt.split(".")[0],
            map_reduce_prompt.split(".")[0],
            f"mr{max_map_reduce_tokens}",
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
                "map_reduce_prompt",
                "max_map_reduce_tokens",
            ]] = [
                summarize_prompt,
                keyword_prompt,
                map_reduce_prompt,
                max_map_reduce_tokens,
            ]

        self._max_map_reduce_tokens = max_map_reduce_tokens

        map_reduce_graph_chain = self._build_map_reduce_graph_chain(
            map_reduce_prompt = map_reduce_prompt,
        )

        self.main_chain = self._build_main_chain(
            map_reduce_graph_chain=map_reduce_graph_chain,
            summarize_prompt=summarize_prompt,
            keyword_prompt=keyword_prompt,
        )

        self.INFO(f"{self.header} prepared DONE")

    def _build_map_reduce_graph_chain(self, map_reduce_prompt: str) -> Runnable:
        self.INFO(f"{self.header} MAP-REDUCE GRAPH CHAIN prepared ....")

        prompt_args, _ = load_prompt(self.content_dir / map_reduce_prompt)
        prompt = PromptTemplate(**prompt_args)
        self._map_reduce_chain = prompt | self.model[0] | StrOutputParser()
        self._map_reduce_graph = self._build_map_reduce_graph()
        chain = RunnableLambda(self._run_map_reduce_graph)

        self.INFO(f"{self.header} MAP-REDUCE GRAPH CHAIN prepared DONE")
        return chain

    def _build_map_reduce_graph(self) -> CompiledStateGraph:
        graph = StateGraph(OverallState)

        graph.add_node("generate_summary", self._generate_summary)
        graph.add_node("collect_summaries", self._collect_summaries)
        graph.add_node("collapse_summaries", self._collapse_summaries)
        graph.add_node("generate_final_summary", self._generate_final_summary)

        graph.add_conditional_edges(START, self._map_summaries, ["generate_summary"],)

        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self._should_collapse)
        graph.add_conditional_edges("collapse_summaries", self._should_collapse)
        graph.add_edge("generate_final_summary", END)

        compiled_graph = graph.compile()
        return compiled_graph

    def _run_map_reduce_graph(self, content: str) -> str:
        self.INFO(f"{self.header} MAP-REDUCE GRAPH PROCEED ....")

        documents = [Document(page_content=content)]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self._max_map_reduce_tokens / 2),
            chunk_overlap=0,
        )

        split_documents = splitter.split_documents(documents)
        contents = [document.page_content for document in split_documents]

        response = self._map_reduce_graph.invoke(
            {"contents": contents},
            {"recursion_limit": 10},
        )

        self.INFO(f"{self.header} MAP-REDUCE GRAPH PROCEED DONE")
        return response["final_summary"]

    def _generate_summary(self, state: SummaryState):
        response = self._map_reduce_chain.invoke(state["content"])
        return {"summaries": [response]}

    def _map_summaries(self, state: OverallState):
        return [
            Send("generate_summary", {"content": content})
            for content in state["contents"]
        ]

    def _collect_summaries(self, state: OverallState):
        return {
            "collapsed_summaries": [
                Document(summary) for summary in state["summaries"]
            ]
        }

    def _collapse_summaries(self, state: OverallState):
        all_documents = reduce.split_list_of_docs(
            state["collapsed_summaries"],
            self._get_num_tokens,
            self._max_map_reduce_tokens
        )

        results = [
            reduce.collapse_docs(documents, self._map_reduce_chain.invoke)
            for documents in all_documents
        ]

        return {"collapsed_summaries": results}

    def _should_collapse(self, state: OverallState
                         ) -> Literal["collapse_summaries",
                                      "generate_final_summary"]:

        num_tokens = self._get_num_tokens(state["collapsed_summaries"])

        return  (
            "collapse_summaries"
            if num_tokens > self._max_map_reduce_tokens else
            "generate_final_summary"
        )

    def _generate_final_summary(self, state: OverallState):
        response = self._map_reduce_chain.invoke(state["collapsed_summaries"])
        return {"final_summary": response}

    def _get_num_tokens(self, documents: list[Document]) -> int:
        return sum(
            self.model[0].get_num_tokens(document.page_content)
            for document in documents
        )

    def _build_main_chain(
        self,
        map_reduce_graph_chain: Runnable,
        summarize_prompt: str,
        keyword_prompt: str
        ) -> Runnable:
        self.INFO(f"{self.header} MAIN CHAIN prepared ....")

        branches = [
            Struct(
                {
                    "pydantic_object": ResearchReportItemSummary,
                    "prompt": summarize_prompt,
                    "condition": lambda x: x["item"] != "keyword",
                }
            ),
            Struct(
                {
                    "pydantic_object": ResearchReportItemKeyword,
                    "prompt": keyword_prompt,
                    "condition": lambda x: x["item"] == "keyword",
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

            statements.append(
                (
                    branch.condition, 
                    RunnableMap(
                        {
                            key: itemgetter(key) for key in [
                                "title",
                                "max_tokens",
                                "num_keywords",
                            ]
                        } | {
                            "content":
                                itemgetter("content") |
                                map_reduce_graph_chain,
                        }
                    ) |
                    prompt |
                    self.model[0] |
                    parser
                )
            )

        statements.append(
            RunnableLambda(lambda _: ResearchReportItemFail(
                message="Invalid item",
            ))
        )

        chain = RunnableBranch(*statements)

        self.INFO(f"{self.header} MAIN CHAIN prepared DONE")
        return chain
