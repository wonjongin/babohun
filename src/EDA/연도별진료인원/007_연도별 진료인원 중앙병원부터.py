import pandas as pd


df = pd.read_csv('data/연도별 진료인원 UTF8/한국보훈복지의료공단_년도별 국가유공자 진료인원_중앙보훈병원_20231231.csv')


result = df.groupby(['년도', '구분']).sum().reset_index()

result.to_csv('data/연도별 진료인원 UTF8/한국보훈복지의료공단_년도별 국가유공자 진료인원_중앙보훈병원_20231231.csv', index=False, encoding='utf-8')