import typer
from lmp_anly.commands.log import log
from lmp_anly.commands.epsilon import epsilon
from lmp_anly.commands.species import species

app = typer.Typer(
    no_args_is_help=True,
    help="structure and plot figures for your LAMMPS output file"
)


app.command(help="plot figure via LAMMPS log")(log)
app.command(help="caculate epsilon via dipole file")(epsilon)
app.command(help="clean species in species file")(species)


def main():
    app()


if __name__ == "__main__":
    main()
