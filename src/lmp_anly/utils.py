import pandas as pd
import numpy as np
import scipy.constants as CONSTANTS
import json
import matplotlib.pyplot as plt
from pathlib import Path
from rich import print


def read_log(filename):
    def measure_log(filename):
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
    item_num = measure_log(filename)
    log_width = [15 for i in range(item_num)]
    log_width[0] = 12
    table = pd.read_fwf(filename, widths=log_width)
    table = table.dropna()
    header = table.loc[table[table.columns[0]]
                       == "Step"].values.flatten().tolist()
    table.columns = header
    table = table.map(lambda x: pd.to_numeric(x, errors='coerce'))
    table = table.dropna()
    table = table.reset_index(drop=True)
    table["Time"] = table["Time"] / 1e+3
    return table


def thermo_statis(dataframe_log):
    if dataframe_log.shape[0] <= 40000:
        raise ValueError("log must record at least 40000 lines")
    volume = np.mean(dataframe_log.loc[40000:, "Volume"]
                     ) * CONSTANTS.angstrom ** 3
    tempertaure = np.mean(dataframe_log.loc[40000:, "Temp"])
    return volume, tempertaure


def cal_ep(dipole_x_array, dipole_y_array, dipole_z_array, vol, temp):
    m_2 = np.sum(
        np.array([np.var(dipole_x_array, axis=0), np.var(
            dipole_y_array, axis=0), np.var(dipole_z_array, axis=0)])
    ) * (CONSTANTS.e * CONSTANTS.angstrom) ** 2
    ep0 = 1 + 4 * CONSTANTS.pi * m_2 / (
        3 * CONSTANTS.k * vol * temp * CONSTANTS.epsilon_0)
    return ep0
