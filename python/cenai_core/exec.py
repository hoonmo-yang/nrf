from typing import (Any, IO, Optional, Union)

import re
import subprocess


def extern_exec(
    cmd: Union[list[str], str], shell: bool = False, **kwargs
) -> int:
    if isinstance(cmd, str) and not shell:
        cmd = re.split(r"\s+", cmd)

    complete = subprocess.run(cmd, shell=shell, **kwargs)
    return complete.returncode


def pipe_exec(cmd: Union[list[str], str],
              shell: bool = False,
              **kwargs
              ) -> tuple[str, str, int]:

    if isinstance(cmd, str) and not shell:
        cmd = re.split(r"\s+", cmd)

    with subprocess.Popen(
        cmd, shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs
    ) as proc:
        stdout: Optional[IO[Any]] = proc.stdout
        stderr: Optional[IO[Any]] = proc.stderr

        stream_out = (
            "" if stdout is None else
            stdout.read().decode("utf-8").strip()
        )

        stream_err = (
            "" if stderr is None else
            stderr.read().decode("utf-8").strip()
        )

    return stream_out, stream_err, proc.returncode
