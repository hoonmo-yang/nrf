from typing import Optional
from datetime import timedelta
import dotenv
import os
from pathlib import Path
import time
from timeit import default_timer


def cenai_path(*args) -> Path:
    return Path(os.environ["CENAI_DIR"]).joinpath(*args).resolve()


def load_dotenv(langsmith: Optional[bool] = None) -> None:
    dotenv_path = cenai_path("cf/.env")
    dotenv.load_dotenv(dotenv_path)

    if langsmith is not None:
        value = "true" if langsmith else "false"
        os.environ["LANGCHAIN_TRACING_V2"] = value


class Timer:
    def __init__(self):
        self.start()

    def start(self) -> None:
        self.current = default_timer()
        self.previous = self.current

    def lap(self) -> None:
        self.current = default_timer()

    @property
    def seconds(self) -> float:
        return self.current - self.previous

    @property
    def delta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    @property
    def delta_string(self) -> str:
        minutes, seconds = divmod(self.seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 60)

        return "".join([
            f"{value:.2f}{unit}"
            for value, unit in zip(
                [days, hours, minutes, seconds], "dhms"
            ) if value > 0
        ])

    @staticmethod
    def delay(sec: int) -> None:
        time.sleep(sec)
