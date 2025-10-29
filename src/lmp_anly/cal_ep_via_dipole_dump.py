import pandas as pd
from lmp_anly.utils import read_log, thermo_statis, cal_ep


def cal_ep_via_dipole(dipole_file, log_file, epsilon_dump_file):
    df_log = read_log(log_file)
    df_dipole = pd.read_csv(
        dipole_file, sep=" ", skiprows=2,
        names=["Step", "Time", "dipole_x", "dipole_y", "dipole_z"],
    )
    vol, temp = thermo_statis(df_log)
    ep0 = cal_ep(df_dipole.loc[40000:, "dipole_x"],
                 df_dipole.loc[40000:, "dipole_y"],
                 df_dipole.loc[40000:, "dipole_z"],
                 vol, temp,
                 )
    with open(epsilon_dump_file.txt, "w") as f:
        f.write("epsilon by dipole.txt\n" + str(ep0))
