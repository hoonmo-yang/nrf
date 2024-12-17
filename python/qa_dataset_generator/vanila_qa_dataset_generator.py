from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import load_chatprompt

from qa_dataset_generator import QADatasetGenerator


class VanilaQADatasetGenerator(QADatasetGenerator):
    def __init__(self,
                 models: Sequence[str],
                 chunk_size: int,
                 chunk_overlap: int,
                 num_datasets: int,
                 max_tokens: int,
                 generate_prompt: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            generate_prompt.split(".")[0],
        ])

        super().__init__(
            models=models,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_datasets=num_datasets,
            max_tokens=max_tokens,
            case_suffix=case_suffix,
            metadata=metadata,
        )

        self.INFO(f"{self.header} prepared ....")

        self.metadata_df.loc[0, "generate_prompt"] = generate_prompt

        self.main_chain = self._build_generate_chain(
            generate_prompt=generate_prompt,
        )

        self.INFO(f"{self.header} prepared DONE")

    def _build_generate_chain(self,
                              generate_prompt: str
                              ) -> Runnable:
        self.INFO(f"{self.header} MAIN CHAIN prepared ....")

        prompt_args = load_chatprompt(self.content_dir / generate_prompt)
        prompt = ChatPromptTemplate(**prompt_args)

        chain = (
            prompt |
            self.model[0] |
            StrOutputParser()
        )

        self.INFO(f"{self.header} MAIN CHAIN prepared DONE")

        return chain
