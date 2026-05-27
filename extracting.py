import pandas as pd

df = pd.read_csv("results/if_scores_all.csv")

print(df['client_mac'].nunique())

print(df['client_mac'].value_counts())