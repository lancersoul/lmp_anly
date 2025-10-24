import typer
import pandas as pd
from typing_extensions import Annotated
from pathlib import Path
from lmp_anly.utils import read_log
from lmp_anly.utils import thermo_statis
from lmp_anly.utils import cal_ep


def epsilon(
    file: Annotated[str, typer.Argument(help="dipole file path")]
):
    work_dir = Path(file).parent
    log_file = [item for item in work_dir.iterdir() if str(
        item).startswith("log.")]
    df_log = read_log(log_file)
    df_dipole = pd.read_csv(
        file, sep=" ", skiprows=2,
        names=["Step", "Time", "dipole_x", "dipole_y", "dipole_z"],
    )
    vol, temp = thermo_statis(df_log)
    ep0 = cal_ep(df_dipole.loc[40000:, "dipole_x"],
                 df_dipole.loc[40000:, "dipole_y"],
                 df_dipole.loc[40000:, "dipole_z"],
                 vol, temp,
                 )
    with open(work_dir / "epsilon.txt", "w") as f:
        f.write("epsilon by dipole.txt\n" + str(ep0))
