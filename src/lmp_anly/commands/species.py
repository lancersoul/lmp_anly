import typer
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
from typing_extensions import Annotated
from lmp_anly.config_creater import ensure_default_config


def species(
    dir: Annotated[str, typer.Argument(help="output dir of reax_tools")],
    threshold: Annotated[int, typer.Option(
        "--threshold", "-th", help="threshold of species count")] = 0,
    timestep: Annotated[int, typer.Option(
        "--timestep", "-t", help="timestep of dump file")] = ...,
    figformat: Annotated[Literal["png", "svg"], typer.Option(
        "--figformat", "-f", help="figure format, can be png or svg")] = "png",
):
    ensure_default_config()
    species_file = Path(dir) / "species_count.csv"
    bond_file = Path(dir) / "bond_count.csv"
    work_dir = Path(dir).parent
    fig_path = work_dir / "figure"
    if Path(species_file).exists():
        df_species = pd.read_csv(species_file, sep=",", header=0)
        plt.figure()
        for species, maxium_index in df_species.idxmax().items():
            if df_species.loc[maxium_index, species] >= threshold:
                plt.plot(
                    df_species.index * timestep / 1e+3,
                    df_species[species],
                    label=species,
                )
        plt.xlabel("Time (ps)")
        plt.ylabel("Product Number")
        plt.legend(fontsize="x-small")
        plt.savefig(fig_path / ("species." + figformat))
    df_bond = pd.read_csv(bond_file, sep=",", header=0)
    plt.figure()
    for i in df_bond.columns:
        plt.plot(
            df_bond.index * timestep / 1e+3,
            df_bond[i],
            label=i,
        )
        plt.xlabel("Time (ps)")
        plt.ylabel("Bond Number")
        plt.legend(fontsize="small")
        plt.savefig(fig_path / ("bond." + figformat))
