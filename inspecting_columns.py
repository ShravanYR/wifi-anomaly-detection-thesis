import pandas as pd

df = pd.read_csv("ml_ready_data.csv")

df_active = df[df['is_active'] == 1]

print(df_active.describe())