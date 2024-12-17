from typing import Any, Union

from io import BytesIO
from datetime import datetime
import pandas as pd
from pathlib import Path
from shutil import rmtree
import streamlit as st
import streamlit.components.v1 as components

from cenai_core import cenai_path
from cenai_core import Logger
from cenai_core.dataman import generate_zip_buffer, get_empty_html, load_json_yaml, Q
from cenai_core.grid import GridRunner


class ResearchResportSummarizationStreamlit(Logger):
    logger_name = "cenai.system"
    profile_dir = cenai_path("python/research_report_summarizer/profile")
    profile_file = profile_dir / "nrf-poc-otf.yaml"
    corpus_dir = cenai_path("data/research-report-summarizer/corpus")

    runner = GridRunner()

    def __init__(self):
        st.set_page_config(
            layout="wide",
        )

        self._profile = load_json_yaml(self.profile_file)

        if "result" not in st.session_state:
            st.session_state.result = {
                "select_file": [],
                "html": [],
            }

        self._choice = None
        self._run_button = None

    def _get_dirs(self, name_only: bool) -> list[Union[Path, str]]:
        dirs = [
            dir_ for dir_ in self.corpus_dir.glob("*")
            if dir_.is_dir()
        ]

        if name_only:
            dirs = [dir_.name for dir_ in dirs]

        return dirs

    def _upload_files(self):
        with st.sidebar:
            st.subheader("파일 관리")

            prefix = st.text_input(
                "Enter folder for file upload:", "sample"
            )

            uploaded_files = st.file_uploader(
                "File upload",
                ["hwpx", "hwp", "docx", "pdf"],
                label_visibility="collapsed",
                accept_multiple_files=True,
            )

            if uploaded_files:
                upload_dir = self.corpus_dir / prefix
                upload_dir.mkdir(parents=True, exist_ok=True)

                for uploaded_file in uploaded_files:
                    target = upload_dir / uploaded_file.name

                    with target.open("wb") as fout:
                        fout.write(uploaded_file.getbuffer())

            dirs = st.multiselect(
                " Select folders to delete:",
                self._get_dirs(True),
            )

            if st.button("Delete folders", use_container_width=True):
                for dir_ in dirs:
                    rmtree(self.corpus_dir / dir_)
                    st.success(f"{Q(dir_)} deleted")

    def _change_parameter_values(self):
        with st.sidebar:
            st.subheader("파라미터 세팅")

            model = st.selectbox(
                "Select LLM model",
                ["gpt-4o", "hcx-003",]
            )

            module = st.selectbox(
                "Select module",
                ["stuff_summarizer", "map_reduce_summarizer",]
            )

            prefix = st.selectbox(
                "Select input folder:", self._get_dirs(True)
            )

            self._run_button = st.button("Run", use_container_width=True)

            self._document_button = st.button(
                "Generate documents", use_container_width=True
            )

        label = f"{datetime.now().strftime("%Y-%m-%d")}_{prefix}"

        self._profile["metadata"]["label"] = label
        self._profile["models"] = [[model, "gpt-4o"]]
        self._profile["corpora"][0]["prefix"] = [prefix]
        self._profile["cases"][0]["module"] = [module]

        if module == "map_reduce_summarizer":
            self._profile["cases"][0]["parameter"].append("num_map_reduce_tokens")

    @staticmethod
    @st.cache_data
    def _get_result(profile: dict[str, Any]) -> dict[str, Any]:
        runner = ResearchResportSummarizationStreamlit.runner
        runner.update(profile)
        result_df = runner.yield_result()

        result_df["select_file"] = result_df.file.apply(
            lambda field: Path(field).name
        )

        result = {
            key: result_df[key].tolist()
            for key in [
                "select_file",
                "html",
            ]
        }

        return result

    @staticmethod
    @st.cache_data
    def _generate_documents(profile: dict[str, Any]) -> tuple[BytesIO, str]:
        runner = ResearchResportSummarizationStreamlit.runner
        runner.update(profile)

        export_dir, extensions = runner.export_documents()

        files = [
            file_ for extension in extensions
            for file_ in export_dir.glob(f"*{extension}")
            if file_.is_file()
        ]

        zip_buffer = generate_zip_buffer(files)

        for file_ in files:
            file_.unlink()

        return [zip_buffer, f"{runner.suite_id}.zip"]

    def invoke(self):
        self._upload_files()
        self._change_parameter_values()

        if self._run_button:
            st.session_state.result = self._get_result(self._profile)
            self._profile["directive"]["force"] = False
            self._profile["directive"]["truncate"] = False

        result = st.session_state.result

        with st.sidebar:
            if self._document_button and result["select_file"]:
                data, file_name = self._generate_documents(self._profile)
                st.download_button(
                    label="Download ZIP file",
                    data=data,
                    file_name=file_name,
                    mime="application/zip",
                    use_container_width=True,
                )

            if st.button("Clear Cache", use_container_width=True):
                st.cache_data.clear()

                self._profile["directive"]["force"] = True
                self._profile["directive"]["truncate"] = True

                st.success("Cache ias been cleared")

            st.subheader("파일 선택")

            choice = st.selectbox(
                "Choose a file:",
                range(len(result["select_file"])),
                format_func=lambda i: result["select_file"][i]
            )

        if result["select_file"]:
            html = result["html"][choice]

        else:
            html = get_empty_html()

        st.subheader("요약 비교")
        components.html(html, height=4800)


def main():
    research_report_summarizer = ResearchResportSummarizationStreamlit()
    research_report_summarizer.invoke()


if __name__ == "__main__":
    main()
