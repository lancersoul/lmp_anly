import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer
from rich import print
from typing import Literal
from typing_extensions import Annotated
from scipy import signal
from scipy import fft
from scipy import constants

app = typer.Typer()


@app.command()
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
    df_log = read_log(file, log_size(file))
    work_dir = fd_work_dir(file)
    fig_path = work_dir + "figure/"
    os.makedirs(fig_path, exist_ok=True)
    figstyle()
    fig_dict = [
        {"name": "Density", "ylabel": r"$\mathrm{g/cm^3}$",
         "label": "Density", "file": "density"},
        {"name": "Temp", "ylabel": "Temperature (°C)",
         "label": "Temperature", "file": "temp"},
        {"name": "v_rmsd", "ylabel": "RMSD",
         "label": "RMSD", "file": "rmsd"},
    ]
    energy_dict = [
        {"name": "TotEng", "label": "Total Energy"},
        {"name": "KinEng", "label": "Kinetic Energy"},
        {"name": "PotEng", "label": "Potential Energy"},
        {"name": "E_mol", "label": "Molecular Energy"},
        {"name": "E_pair", "label": "Pairwise Energy"},
    ]
    dipole_dict = [
        {"name": "v_dipole_x", "label": "Dipole Moment in X"},
        {"name": "v_dipole_y", "label": "Dipole Moment in Y"},
        {"name": "v_dipole_z", "label": "Dipole Moment in Z"},
    ]

    for item in fig_dict:
        if item["name"] in df_log.columns:
            plt.figure()
            plt.plot(df_log["Time"], df_log[item["name"]],
                     label=item["label"])
            plt.xlabel("Time (ps)")
            plt.ylabel(item["ylabel"])
            plt.legend()
            plt.savefig(fig_path + item["file"] + "." + figformat)

    if any(element for element in energy_dict if item["name"] in df_log.columns):
        plt.figure()
        for item in energy_dict:
            if item["name"] in df_log.columns:
                plt.plot(df_log["Time"], df_log[item["name"]],
                         label=item["label"])
                plt.xlabel("Time (ps)")
                plt.ylabel("Energy (kcal/mol)")
                plt.legend()
                plt.savefig(fig_path + "energy." + figformat)

    if any(element for element in dipole_dict if item["name"] in df_log.columns):
        plt.figure()
        for item in dipole_dict:
            plt.plot(df_log["Time"], df_log[item["name"]], label=item["label"])
            plt.xlabel("Time (ps)")
            plt.ylabel("Dipole Moment")
            plt.legend()
            plt.savefig(fig_path + "dipole." + figformat)

    if epsilon and any(
            element for element in dipole_dict if item["name"] in df_log.columns):
        vol = np.mean(df_log.loc[40000:, "Volume"]
                      ) * constants.angstrom ** 3
        temp = np.mean(df_log.loc[40000:, "Temp"])
        ep0 = cal_ep(df_log.loc[40000:, "v_dipole_x"],
                     df_log.loc[40000:, "v_dipole_y"],
                     df_log.loc[40000:, "v_dipole_z"],
                     vol, temp,
                     )
        print("ep0=" + str(ep0))
        with open(work_dir + "epsilon.txt", "w") as f:
            f.write("epsilon by log\n" + str(ep0))


@app.command(help="caculate epsilon via dipole file")
def epsilon(
    file: Annotated[str, typer.Argument(help="dipole file path")]
):
    work_dir = fd_work_dir(file)
    log_file = [item for item in os.listdir(
        work_dir) if item.startswith("log.")]
    df_log = read_log(log_file)
    df_dipole = pd.read_csv(
        file, sep=" ", skiprows=2,
        names=["Step", "Time", "dipole_x", "dipole_y", "dipole_z"],
    )
    vol = np.mean(df_log.loc[40000:, "Volume"]
                  ) * constants.angstrom ** 3
    temp = np.mean(df_log.loc[40000:, "Temp"])
    ep0 = cal_ep(df_dipole[40000:, "dipole_x"],
                 df_dipole[40000:, "dipole_y"],
                 df_dipole[40000:, "dipole_z"],
                 vol, temp,
                 )
    with open(work_dir + "epsilon.txt", "w") as f:
        f.write("epsilon by dipole.txt\n" + str(ep0))


@app.command(help="clean species in species file")
def species(
    dir: Annotated[str, typer.Argument(help="output dir of reax_tools")],
    threshold: Annotated[int, typer.Option(
        "--threshold", "-th", help="threshold of species count")] = 0,
    timestep: Annotated[int, typer.Option(
        "--timestep", "-t", help="timestep of dump file")] = ...,
    figformat: Annotated[Literal["png", "svg"], typer.Option(
        "--figformat", "-f", help="figure format, can be png or svg")] = "png",
):
    species_file = dir + "species_count.csv"
    bond_file = dir + "bond_count.csv"
    work_dir = os.path.dirname(dir)
    fig_path = work_dir + "figure/"
    figstyle()
    if os.path.exists(species_file):
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
        plt.savefig(fig_path + "species." + figformat)
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
        plt.savefig(fig_path + "bond." + figformat)


def read_log(filename, item_num):
    log_width = [15 for i in range(item_num)]
    log_width[0] = 12
    table = pd.read_fwf(filename, widths=log_width)
    table = table.dropna()
    header = table.loc[table[table.columns[0]] == "Step"].values.tolist()
    table.columns = header
    table = table.map(lambda x: pd.to_numeric(x, errors='coerce'))
    table = table.dropna()
    table = table.reset_index(drop=True)
    table["Time"] = table["Time"] / 1e+3
    return table


def log_size(filename):
    with open(filename, "br") as file:
        file.seek(-2, 2)
        for i in range(60):
            file.seek(-2, 1)
            while file.read(1) != b"\n":
                file.seek(-2, 1)
        line_content = file.readline()
        line_size = len(line_content)
        log_nitem = int((line_size - 12) / 15 + 1)
    return log_nitem


def fd_work_dir(logfile):
    if logfile.rfind("/") != -1:
        work_dir = logfile[0:logfile.rfind("/") + 1]
    else:
        work_dir = "./"
    return work_dir


def cal_ep(dipole_x, dipole_y, dipole_z, vol, temp):
    m_2 = np.sum(
        np.array([np.var(dipole_x, axis=0), np.var(dipole_y, axis=0), np.var(dipole_z, axis=0)])
    ) * (constants.e * constants.angstrom) ** 2
    ep0 = 1 + 4 * constants.pi * m_2 / (
        3 * constants.k * vol * temp * constants.epsilon_0)
    return ep0


def figstyle():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.figsize'] = [3.54, 2.36]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['lines.linewidth'] = 0.35


def main():
    app()


if __name__ == "__main__":
    main()
