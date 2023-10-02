import numpy as np
import pandas as pd

df= pd.read_csv('/home/user/Загрузки/titanic_with_labels.csv', sep=' ', index_col=0)

df['sex'].replace('-', float('nan'))
df['sex'].replace('Не указан', float('nan'))
df = df.dropna(subset=['sex'])

df['sex'] = ((df['sex'] == 'м') | (df['sex'] == 'M')).astype(int)

df['row_number'].fillna(df['row_number'].max())

filtered_values = df[(df['liters_drunk'] >= 0) & (df['liters_drunk'] < 10)]

mean_liters_drunk = filtered_values['liters_drunk'].mean()

df.loc[df['liters_drunk'] < 0, 'liters_drunk'] = mean_liters_drunk
df.loc[df['liters_drunk'] > 10, 'liters_drunk'] = mean_liters_drunk

print(df)
