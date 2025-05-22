import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_list = os.listdir('merged_data/')
print(file_list)

for file in file_list:
    if file.endswith('.csv'):
        print("" + "="*50)
        print(file)
        df = pd.read_csv('merged_data/' + file)
        print(df.shape)                         # 데이터 크기(행, 열)
        print(df.columns)                       # 컬럼명
        print(df.dtypes)                        # 데이터 타입
        print(df.head())                        # 상위 5개 행 미리보기
        print(df.info())                        # 전체 요약 정보(결측치, 타입 등)
        print(df.describe())                    # 수치형 변수의 기초 통계량
        print(df.describe(include='object'))    # 범주형 변수의 기초 통계량