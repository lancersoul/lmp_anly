import typer
import matplotlib.pyplot as plt
from typing import Literal
from typing_extensions import Annotated
from rich import print
from pathlib import Path
from collections import defaultdict
from lmp_anly.config_creater import ensure_default_config
from lmp_anly.utils import measure_log
from lmp_anly.utils import read_log
from lmp_anly.utils import load_config
from lmp_anly.utils import thermo_statis
from lmp_anly.utils import cal_ep


def log(
    file: Annotated[str, typer.Argument(
        help="relative path of LAMMPS log file")],
    figformat: Annotated[
        Literal["png", "svg"],
        typer.Option("--format", "-f", help="figure format, can be png or svg")
    ] = "png",
    epsilon: Annotated[bool, typer.Option(
        "--epsilon", "-e", help="caculate epsilon via dipole moment in log")] = False,
):
    config_file = ensure_default_config()
    config = load_config(config_file)
    for key, value in config["mpl_style"].items():
        plt.rcParams[key] = value
    df_log = read_log(file, measure_log(file))
    work_dir = Path(file).parent
    fig_path = work_dir / "figure"
    Path.mkdir(fig_path, parents=True, exist_ok=True)

    def find_multiplot_fig(fig_element):
        all_fig = defaultdict(list)
        for key, fig in fig_element.items():
            value = fig["fig_name"]
            all_fig[value].append(key)
        return dict(all_fig)

    line_element = config["line_element"]
    all_fig = find_multiplot_fig(line_element)
    for fig_name, line_list in all_fig.items():
        if set([line_element[line]["column_name"] for line in line_list]) & set(df_log.columns):
            plt.figure()
            for line in line_list:
                if line_element[line]["column_name"] in df_log.columns:
                    plt.plot(
                        df_log["Time"],
                        df_log[line_element[line]["column_name"]],
                        label=line_element[line]["label"]
                    )
            plt.xlabel("Time (ps)")
            plt.ylabel(line_element[line_list[0]]["ylabel"])
            plt.legend()
            plt.savefig(fig_path / (fig_name + "." + figformat))

    if epsilon and line_element["dipole_x"]["column_name"] in df_log.columns:
        vol, temp = thermo_statis(df_log)
        ep0 = cal_ep(
            df_log.loc[40000:, line_element["dipole_x"]["column_name"]],
            df_log.loc[40000:, line_element["dipole_y"]["column_name"]],
            df_log.loc[40000:, line_element["dipole_z"]["column_name"]],
            vol, temp
        )
        print("ep0=" + str(ep0))
        with open(work_dir / "epsilon.txt", "w") as f:
            f.write("epsilon by log\n" + str(ep0))
