from collections import defaultdict

import matplotlib.pyplot as plt
from rich import print
from rich.progress import track


def plot_log(df_log, line_element, fig_path, fig_format):
    def find_multiplot_fig(fig_element):
        all_fig = defaultdict(list)
        for key, fig in fig_element.items():
            value = fig["fig_name"]
            all_fig[value].append(key)
        return dict(all_fig)

    all_fig = find_multiplot_fig(line_element)
    fig_plot_count = 0
    for fig_name, line_list in track(all_fig.items(), description="Plotting..."):
        if {line_element[line]["column_name"] for line in line_list} & set(df_log.columns):
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
            plt.savefig(fig_path / (fig_name + "." + fig_format))
            fig_plot_count += 1
        else:
            print(f"{fig_name} is not in LAMMPS log!")
    print(f"Plotted {fig_plot_count} figures.")
