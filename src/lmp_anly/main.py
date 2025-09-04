import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft
from scipy import constants


def read_log(filename, log_nitem):
    log_width = [15 for i in range(log_nitem)]
    log_width[0] = 12
    table = pd.read_fwf(filename, widths=log_width)
    table = table.dropna()
    header = table.loc[table[table.columns[0]] == "Step"].values.tolist()
    table.columns = header
    table = table.map(lambda x: pd.to_numeric(x, errors='coerce'))
    table = table.dropna()
    table = table.reset_index(drop=True)
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
        np.array([np.var(dipole_x), np.var(dipole_y), np.var(dipole_z)])
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logfile", type=str,
        help="relative path of LAMMPS log file"
    )
    parser.add_argument(
        "-f", "--figformat", type=str, default="png", choices={"png", "svg"},
        help="figure format, default png"
    )
    parser.add_argument(
        "--dipole", "-d", action="store_true",
        help="caculate dielectric constant by log file"
    )
    parser.add_argument(
        "--DIPOLE", "-D", action="store_true",
        help="caculate dielectric constant by dipole.txt in the parent dir"
    )
    parser.add_argument(
        "--nothermo", "-no", action="store_true",
        help="don't plot thermodynamics figures"
    )
    parser.add_argument(
        "--species", "-s", nargs=2, type=int,
        help="time per frames of lammpstrj and threshold of species"
    )
    args = parser.parse_args()

    logfile = args.logfile
    work_dir = fd_work_dir(logfile)
    dipolefile = work_dir + "dipole.txt"
    fig_path = work_dir + "figure/"
    os.makedirs(fig_path, exist_ok=True)
    figstyle()

    df_log = read_log(args.logfile, log_size(args.logfile))
    df_log["Time"] = df_log["Time"] / 1e+3
    if not args.nothermo:
        plt.figure()
        for i in ["TotEng", "KinEng", "PotEng", "E_pair", "E_mol"]:
            if i in df_log.columns:
                plt.plot(df_log["Time"], df_log[i], label=i)
        plt.xlabel("Time (ps)")
        plt.ylabel("Energy (kcal/mol)")
        plt.legend()
        plt.savefig(fig_path + "energy." + args.figformat)

        plt.figure()
        plt.plot(df_log["Time"], df_log["Density"], label="Density")
        plt.xlabel("Time (ps)")
        plt.ylabel(r"Density ($\mathrm{g/cm^3}$)")
        plt.legend()
        plt.savefig(fig_path + "density." + args.figformat)

        plt.figure()
        plt.plot(df_log["Time"], df_log["Temp"], label="Temperature")
        plt.xlabel("Time (ps)")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.savefig(fig_path + "temp." + args.figformat)

        if "v_rmsd" in df_log.columns:
            plt.figure()
            plt.plot(df_log["Time"], df_log["v_rmsd"], label="RMSD")
            plt.xlabel("Time (ps)")
            plt.ylabel("RMSD")
            plt.legend()
            plt.savefig(fig_path + "rmsd." + args.figformat)

        flag = 0
    if args.dipole:
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
        flag = 1
    if args.DIPOLE:
        vol = np.mean(df_log.loc[40000:, "Volume"]
                      ) * constants.angstrom ** 3
        temp = np.mean(df_log.loc[40000:, "Temp"])
        df_dipole = pd.read_csv(
            dipolefile, sep=" ", skiprows=2,
            names=["Step", "Time", "dipole_x", "dipole_y", "dipole_z"],
        )
        ep0 = cal_ep(df_dipole[40000:, "dipole_x"],
                     df_dipole[40000:, "dipole_y"],
                     df_dipole[40000:, "dipole_z"],
                     vol, temp,
                     )
        if flag == 0:
            with open(work_dir + "epsilon.txt", "w") as f:
                f.write("epsilon by dipole.txt\n" + str(ep0))
        else:
            with open(work_dir + "epsilon.txt", "a") as f:
                f.write("\nepsilon by dipole.txt\n" + str(ep0))
    if args.species:
        species_file = work_dir + "output/species_count.csv"
        bond_file = work_dir + "output/bond_count.csv"
        if os.path.exists(species_file):
            df_species = pd.read_csv(species_file, sep=",", header=0)
            plt.figure()
            for species, maxium_index in df_species.idxmax().items():
                if df_species.loc[maxium_index, species] >= args.thersold:
                    plt.plot(
                        df_species.index * args.species / 1e+3,
                        df_species[species],
                        label=species,
                    )
            plt.xlabel("Time (ps)")
            plt.ylabel("Product Number")
            plt.legend(fontsize="x-small")
            plt.savefig(fig_path + "species." + args.figformat)
        df_bond = pd.read_csv(bond_file, sep=",", header=0)
        plt.figure()
        for i in df_bond.columns:
            plt.plot(
                df_bond.index * args.species / 1e+3,
                df_bond[i],
                label=i,
            )
            plt.xlabel("Time (ps)")
            plt.ylabel("Bond Number")
            plt.legend(fontsize="small")
            plt.savefig(fig_path + "bond." + args.figformat)


if __name__ == "__main__":
    main()
