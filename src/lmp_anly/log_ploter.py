import matplotlib.pyplot as plt
from collections import defaultdict


def plot_log(df_log, line_element, fig_path, fig_format):
    def find_multiplot_fig(fig_element):
        all_fig = defaultdict(list)
        for key, fig in fig_element.items():
            value = fig["fig_name"]
            all_fig[value].append(key)
        return dict(all_fig)

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
            plt.savefig(fig_path / (fig_name + "." + fig_format))
