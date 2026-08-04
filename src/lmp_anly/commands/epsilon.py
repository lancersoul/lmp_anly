from pathlib import Path
from typing import Annotated

import typer

from lmp_anly.cal_ep_via_dipole_dump import cal_ep_via_dipole


def epsilon(
    file: Annotated[str, typer.Argument(help="dipole file path")]
):
    work_dir = Path(file).parent
    log_file = [item for item in work_dir.iterdir() if str(
        item).startswith("log.") or str(item).endswith(".log")]
    epsilon_dump_file = work_dir / "epsilon.txt"
    cal_ep_via_dipole(
        dipole_file=file,
        log_file=log_file,
        epsilon_dump_file=epsilon_dump_file,
    )
