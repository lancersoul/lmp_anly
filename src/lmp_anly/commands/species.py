import typer
from pathlib import Path
from typing import Literal
from typing_extensions import Annotated
from rich import print
from lmp_anly.config_creater import ensure_default_config
from lmp_anly.utils import read_config
from lmp_anly.utils import load_figstyle
from lmp_anly.plot_species import plot_species
from lmp_anly.plot_bond import plot_bond


def species(
    dir: Annotated[str, typer.Argument(help="output dir of reax_tools")],
    threshold: Annotated[int, typer.Option(
        "--threshold", "-th", help="threshold of species count")] = 0,
    timestep: Annotated[int, typer.Option(
        "--timestep", "-t", help="timestep of dump file")] = ...,
    figformat: Annotated[Literal["png", "svg"], typer.Option(
        "--figformat", "-f", help="figure format, can be png or svg")] = "png",
):
    config_file = ensure_default_config()
    config = read_config(config_file)
    load_figstyle(config)
    species_file = Path(dir) / "species_count.csv"
    bond_file = Path(dir) / "bond_count.csv"
    work_dir = Path(dir).parent
    fig_path = work_dir / "figure"
    try:
        plot_species(
            species_file,
            threshold=threshold,
            timestep=timestep,
            figformat=figformat,
            fig_path=fig_path
        )
    except FileNotFoundError:
        print(f"[bold green]{species_file}[/bold green] [red]doesn\'t exist[/red]")
    try:
        plot_bond(
            bond_file,
            timestep=timestep,
            figformat=figformat,
            fig_path=fig_path,
        )
    except FileNotFoundError:
        print(f"[bold green]{bond_file}[/bold green] [red]doesn\'t exist[/red]")
