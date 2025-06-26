import pandas as pd

df = pd.read_csv('final_merged_data/외래 진료과별 상위20 주요상병.csv')

pivot = df.pivot_table(
    index='상명코드',
    columns='진료과',
    values='건수',
    aggfunc='sum',
    fill_value=0
)

pivot.to_csv('new_merged_data/상병코드별_진료과_건수_더미.csv')
