import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc


font_path = "fonts/Pretendard-Regular.ttf"
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('merged_data/외래 진료과별 상위20 주요상병.csv')
print(df.shape)                         # 데이터 크기(행, 열)
print(df.columns)                       # 컬럼명
print(df.dtypes)                        # 데이터 타입
print(df.head())                        # 상위 5개 행 미리보기
print(df.info())                        # 전체 요약 정보(결측치, 타입 등)
print(df.describe())                    # 수치형 변수의 기초 통계량
print(df.describe(include='object'))    # 범주형 변수의 기초 통계량

regions = df['지역'].unique()
years = df['년도'].unique()
department = df['진료과'].unique()
## 진료과 별로도 하고, 진료과 합쳐서 순위도 하기

## 진료과, 지역, 년도 별로 상위 20개 주요상병을 시각화
for region in regions:
    for year in years:
        for dep in department:
            filtered_df = df[(df['지역'] == region) & (df['년도'] == year) & (df['진료과'] == dep)]
            if filtered_df.empty:
                print(f"지역: {region}, 년도: {year}, 진료과: {dep}에 대한 데이터가 없습니다.")
                continue
            sorted_df = filtered_df.sort_values(by='순위', ascending=True)
            print("="*50)
            # print(sorted_df.head(20))  # 상위 20개 행 출력
            print(f"지역: {region}, 년도: {year}, 진료과: {dep}")

            sns.barplot(x="상병명", y="건수", data=sorted_df)
            plt.xticks(rotation=90)
            plt.title(f"{region} {year} {dep} 외래 진료과별 상위20 주요상병")
            plt.xlabel("질병명")
            plt.ylabel("진료건수")
            plt.grid(True)
            plt.savefig(f"imgs/EDA_상위20 주요상병/plot {region} {year} {dep}.png", bbox_inches='tight')
            plt.close()

# 진료과 구분 없이 병원 별로 상위 20개 주요상병을 시각화
for region in regions:
    for year in years:
        filtered_df = df[(df['지역'] == region) & (df['년도'] == year)]
        if filtered_df.empty:
            print(f"지역: {region}, 년도: {year}에 대한 데이터가 없습니다.")
            continue
        sorted_df = filtered_df.sort_values(by='순위', ascending=True)
        print("="*50)
        # print(sorted_df.head(20))  # 상위 20개 행 출력
        print(f"지역: {region}, 년도: {year}")

        sns.barplot(x="상병명", y="건수", data=sorted_df)
        plt.xticks(rotation=90)
        plt.title(f"{region} {year} 외래 진료과별 상위20 주요상병")
        plt.xlabel("질병명")
        plt.ylabel("진료건수")
        plt.grid(True)
        plt.savefig(f"imgs/EDA_상위20 주요상병/plot top20 {region} {year}.png", bbox_inches='tight')
        plt.close()




# print(df['구분'].unique())
# df['구분'] = df['구분'].replace({
#     '입원 실인원': '입원(실인원)',
#     '입원 연인원': '입원(연인원)'
# })

# df = df.replace('NA', np.nan)


# # df['merged'] = df['col1'].fillna(df['col2'])


# df['상병명'] = df['상병명'].fillna(df['명칭'])
# df.drop(columns=['명칭'], inplace=True)

# age_columns = ['X59이하', 'X60.64', 'X65.69', 'X70.79', 'X80.89', 'X90이상']
# df[age_columns] = df[age_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
# df.loc[df['연령별.합계'].isna(), '연령별.합계'] = df.loc[df['연령별.합계'].isna(), age_columns].sum(axis=1)

# df = df[['년도', '지역', '구분', '코드', '상병명', '연령별.합계', 'X59이하', 'X60.64', 'X65.69', 'X70.79', 'X80.89', 'X90이상']]

# print(df.columns)
# df.replace(np.nan, 'NaN', inplace=True)
# df.to_csv('result_utf8.csv', index=False, encoding='utf-8')


'''
 RangeIndex: 14296 entries, 0 to 14295
Data columns (total 7 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   진료과     14296 non-null  object
 1   순위      14296 non-null  int64 
 2   상명코드    13645 non-null  object
 3   상병명     14267 non-null  object
 4   건수      14296 non-null  int64 
 5   지역      14296 non-null  object
 6   년도      14296 non-null  int64 
dtypes: int64(3), object(4)
'''