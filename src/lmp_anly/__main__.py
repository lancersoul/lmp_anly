from pathlib import Path
from typing import Annotated

import typer

from lmp_anly.commands.epsilon import epsilon
from lmp_anly.commands.log import log
from lmp_anly.commands.species import species
from lmp_anly.config_utils import create_default_config, fd_config_path

app = typer.Typer(
    no_args_is_help=True, help="structure and plot figures for your LAMMPS output file"
)


app.command(help="plot figure via LAMMPS log")(log)
app.command(help="caculate epsilon via dipole file")(epsilon)
app.command(help="clean species in species file")(species)


def reset_config(value=bool):
    if value:
        config_file = fd_config_path()
        Path.mkdir(config_file.parent, parents=True, exist_ok=True)
        create_default_config()
        print(f"Your profile has been reset in {config_file}")


@app.callback(
    invoke_without_command=True,
    help="structure and plot figures for your LAMMPS output file",
)
def call_back(
    reset: Annotated[
        bool,
        typer.Option(
            "--reset",
            "-r",
            help="reset configuration of lmp-anly",
            callback=reset_config,
            is_eager=True,
        ),
    ] = False,
):
    config_file = fd_config_path()
    if not Path.exists(config_file):
        Path.mkdir(config_file.parent, parents=True, exist_ok=True)
        create_default_config()
        print(f"Your profile has been created in {config_file}")


def main():
    app()


if __name__ == "__main__":
    main()
