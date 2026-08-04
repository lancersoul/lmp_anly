import matplotlib.pyplot as plt
import pandas as pd


def plot_bond(bond_file, timestep, figformat, fig_path):
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
