from rich import print

from lmp_anly.utils import cal_ep, thermo_statis


class MissingValueError(Exception):
    def __init__(self, misssing_keys, message=None, context=None):
        self.missing_keys = tuple(misssing_keys)
        self.context = context
        if message is None:
            message = f"Message required values: {', '.join(self.missing_keys)}"
        super().__init__(message)


def cal_ep_via_log(df_log, line_element, epsilon_dump_file):
    missing = [miss_key for miss_key in ["dipole_x", "dipole_y", "dipole_z"]
               if not line_element[miss_key]["column_name"] in df_log.columns]
    if missing:
        raise MissingValueError(misssing_keys=missing, context={
                                "log_head": list(df_log.columns)})
    vol, temp = thermo_statis(df_log)
    ep0 = cal_ep(
        df_log.loc[40000:, line_element["dipole_x"]["column_name"]],
        df_log.loc[40000:, line_element["dipole_y"]["column_name"]],
        df_log.loc[40000:, line_element["dipole_z"]["column_name"]],
        vol, temp
    )
    print("ep0=" + str(ep0))
    with open(epsilon_dump_file, "w") as f:
        f.write("epsilon by log\n" + str(ep0))
