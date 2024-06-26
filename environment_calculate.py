import pandas as pd


def get_environment_df():
    env_dataset_df = pd.read_csv("Labels/食材のTMR係数.csv")
    env_dataset_df = env_dataset_df.rename(
        columns={
            "TMR係数\n(kg-TMR/kg)": "TMR係数",
            "GWP\n(kg-CO2-eq/kg)": "GWP",
            "反応性窒素\n(kg-NOx/kg)": "反応性窒素",
        }
    )[
        [
            "食品番号",
            "TMR係数",
            "GWP",
            "反応性窒素",
        ]
    ]
    env_dataset_df = env_dataset_df.set_index("食品番号")
    env_dataset_df = env_dataset_df.fillna(0).astype(float)
    return env_dataset_df
