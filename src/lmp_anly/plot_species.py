import pandas as pd
import matplotlib.pyplot as plt


def plot_species(species_file, threshold, timestep, figformat, fig_path):
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
