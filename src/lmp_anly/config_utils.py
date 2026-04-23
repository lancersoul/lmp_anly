import matplotlib.pyplot as plt
import rtoml
import platform
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class FigDict:
    column_name: str
    ylabel: str
    label: str
    fig_name: str


def create_default_config():
    config_dir = fd_config_path() / "lmp-anly"
    config_file = config_dir / "config.toml"
    if not Path.exists(config_file):
        Path.mkdir(config_dir, parents=True, exist_ok=True)
        DEFAULT_CONFIG = {
            "mpl_style": {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "TImes", "Nimbus Roman", "Nimbus Roman No9 L","Liberation Serif", "SimSun", "Songti SC", "STSong", "Noto Serif SC", "WenQuanYi Micro Hei", "DejaVu Serif"],
                "mathtext.fontset": "stix",
                "figure.constrained_layout.use": True,
                "figure.figsize": [3.54, 2.36],
                "figure.dpi": 300,
                "lines.linewidth": 0.35,
            },
            "line_element": {
                "density": asdict(FigDict("Density", r"Density ($\mathrm{g/cm^3}$)", "Density", "density")),
                "press": asdict(FigDict("Press", "Pressure (atm.)", "Press", "press")),
                "volume": asdict(FigDict("Volume", r"Volume ($\mathrm{Å^{3}}$)", "Volume", "volume")),
                "temp": asdict(FigDict("Temp", "Temperature(°C)", "Temperature", "temp")),
                "rmsd": asdict(FigDict("v_rmsd", "RMSD (Å)", "RMSD", "rmsd")),
                "total_energy": asdict(FigDict("TotEng", "Energy (kcal/mol)", "Total Energy", "energy")),
                "kin_energy": asdict(FigDict("KinEng", "Energy (kcal/mol)", "Kinetic Energy", "energy")),
                "pot_energy": asdict(FigDict("PotEng", "Energy (kcal/mol)", "Potential Energy", "energy")),
                "pair_energy": asdict(FigDict("E_pair", "Energy (kcal/mol)", "Pairwise Energy", "energy")),
                "coul_energy": asdict(FigDict("E_coul", "Energy (kcal/mol)", "Coulombic Energy", "energy")),
                "vdwl_energy": asdict(FigDict("E_vdwl", "Energy (kcal/mol)", "van der Waals Pairwise Energy", "energy")),
                "long_energy": asdict(FigDict("E_long", "Energy (kcal/mol)", "Long-range Energy", "energy")),
                "mol_energy": asdict(FigDict("E_mol", "Energy (kcal/mol)", "Molecular Energy", "energy")),
                "bond_energy": asdict(FigDict("E_bond", "Energy (kcal/mol)", "Bond Energy", "energy")),
                "angle_energy": asdict(FigDict("E_angle", "Energy (kcal/mol)", "Angle Energy", "energy")),
                "dihed_energy": asdict(FigDict("E_dihed", "Energy (kcal/mol)", "Dihedral Energy", "energy")),
                "imp_energy": asdict(FigDict("E_impro", "Energy (kcal/mol)", "Improper Energy", "energy")),
                "enthalpy": asdict(FigDict("Enthalpy", "Energy (kcal/mol)", "Enthalpy", "energy")),
                "dipole_x": asdict(FigDict("v_dipole_x", r"Dipole Moment ($\mathrm{eV \cdot Å}$)", "Dipole Moment in X", "dipole")),
                "dipole_y": asdict(FigDict("v_dipole_y", r"Dipole Moment ($\mathrm{eV \cdot Å}$)", "Dipole Moment in Y", "dipole")),
                "dipole_z": asdict(FigDict("v_dipole_z", r"Dipole Moment ($\mathrm{eV \cdot Å}$)", "Dipole Moment in Z", "dipole")),
            }
        }
        rtoml.dump(DEFAULT_CONFIG, config_file)
    pass


def read_config():
    config_file = fd_config_path() / "lmp-anly" / "config.toml"
    config_dict = rtoml.load(config_file)
    return config_dict


def load_figstyle(config):
    for key, value in config["mpl_style"].items():
        plt.rcParams[key] = value


def fd_config_path():
    system_name = platform.system()
    if system_name == "Windows":
        config_dir = Path.home() / "AppData" / "Local"
    elif system_name == "Darwin":  # macOS
        config_dir = Path.home() / "Library" / "Application Support"
    elif system_name == "Linux":
        config_dir = Path.home() / ".config"
    else:
        config_dir = Path.home() / ".config"  # 默认使用主目录
    return config_dir
