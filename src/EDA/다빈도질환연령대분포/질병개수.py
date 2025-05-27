import pandas as pd


df = pd.read_csv('final_merged_data/다빈도 질환 환자 연령별 분포.csv')

arr = df['상병코드'].unique()

print(arr)
print(len(arr))