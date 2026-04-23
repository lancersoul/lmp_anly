import typer
from typing import Literal
from typing_extensions import Annotated
from pathlib import Path
from rich import print
from lmp_anly.config_utils import create_default_config, read_config, load_figstyle
from lmp_anly.utils import read_log
from lmp_anly.log_ploter import plot_log
from lmp_anly.cal_ep_via_log import cal_ep_via_log, MissingValueError


def log(
    file: Annotated[str, typer.Argument(
        help="relative path of LAMMPS log file")],
    figformat: Annotated[
        Literal["png", "svg"],
        typer.Option("--format", "-f", help="figure format, can be png or svg")
    ] = "svg",
    epsilon: Annotated[bool, typer.Option(
        "--epsilon", "-e", help="caculate epsilon via dipole moment in log")] = False,
):
    create_default_config()
    config = read_config()
    load_figstyle(config)
    line_element = config["line_element"]
    df_log = read_log(file)
    work_dir = Path(file).parent
    fig_path = work_dir / "figure"
    Path.mkdir(fig_path, parents=True, exist_ok=True)

    plot_log(
        df_log=df_log,
        line_element=line_element,
        fig_format=figformat,
        fig_path=fig_path
    )

    if epsilon:
        try:
            epsilon_file = work_dir / "epsilon.txt"
            cal_ep_via_log(
                df_log=df_log,
                line_element=line_element,
                epsilon_dump_file=epsilon_file
            )
        except MissingValueError as e:
            print(f"[green bold]{
                  e.missing_keys
                  }[/green bold] not in LAMMPS log \nLAMMPS log head is \n{
                  e.context['log_head']
                  }")
