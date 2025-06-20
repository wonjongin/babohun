import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import plotly.express as px

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('merged_data/외래 진료과별 상위20 주요상병.csv')
'''
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

## 기술 통계
# 진료과별 건수 통계
dep_stats = df.groupby('진료과')['건수'].agg(['mean', 'sum', 'std']).reset_index()
print("진료과별 건수 통계")
print(dep_stats)
dep_stats.to_csv("src/EDA/외래진료과별20대주요상병/진료과별_건수_통계.csv", index=False, encoding='utf-8-sig')

# 상병별 건수 통계
disease_stats = df.groupby('상병명')['건수'].agg(['mean', 'sum', 'std']).reset_index()
print("\n상병별 건수 통계")
print(disease_stats)
disease_stats.to_csv("src/EDA/외래진료과별20대주요상병/상병명별_건수_통계.csv", index=False, encoding='utf-8-sig')

# 지역별 건수 통계
region_stats = df.groupby('지역')['건수'].agg(['mean', 'sum', 'std']).reset_index()
print("\n지역별 건수 통계")
print(region_stats)

# 연도별 건수 통계
year_stats = df.groupby('년도')['건수'].agg(['mean', 'sum', 'std']).reset_index()
print("\n연도별 건수 통계")
print(year_stats)

# Sunburst chart : 상위 1000개
sunburst_df = df.sort_values('건수', ascending=False).head(1000)

fig = px.sunburst(
    sunburst_df,
    path=['진료과', '상병명', '지역'],
    values='건수',
    title='진료과 - 주요 상병 - 지역별 건수 Sunburst Chart'
)
fig.show()

# 진료과별 상병 건수 Treemap
treemap_df = df.groupby(['진료과', '상병명'], as_index=False)['건수'].sum()

fig = px.treemap(
    treemap_df,
    path=['진료과', '상병명'],
    values='건수',
    title='진료과별 주요 상병 건수 Treemap'
)
fig.show()

# 전국 단위 Top N 상병 막대 그래프
top_n = 20
top_diseases = df.groupby('상병명')['건수'].sum().sort_values(ascending=False).head(top_n).index.tolist()

trend_df = df[df['상병명'].isin(top_diseases)].groupby(['년도', '상병명'], as_index=False)['건수'].sum()

plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_df, x='년도', y='건수', hue='상병명', marker='o')
plt.title(f'전국 상위 {top_n} 주요 상병 연도별 건수 추이')
plt.ylabel('건수')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()

# 주요 상병 누적 영역형 차트
pivot_df = trend_df.pivot(index='년도', columns='상병명', values='건수').fillna(0)

pivot_df.plot(kind='area', stacked=True, figsize=(12,6), cmap='tab20')
plt.title(f'전국 상위 {top_n} 주요 상병 누적 영역형 차트')
plt.ylabel('건수')
plt.grid(True)
plt.show()

# 진료과별 주요 상병 Top N 막대 그래프
for dep in df['진료과'].unique():
    dep_df = df[df['진료과'] == dep].groupby('상병명')['건수'].sum().sort_values(ascending=False).head(top_n).reset_index()
    plt.figure(figsize=(10,5))
    sns.barplot(x='건수', y='상병명', data=dep_df, hue='상병명', palette='viridis', legend=False)
    plt.title(f'{dep} 진료과별 상위 {top_n} 주요 상병')
    plt.xlabel('건수')
    plt.ylabel('상병명')
    plt.tight_layout()
    plt.show()

# 진료과별 상병 건수 boxplot
plt.figure(figsize=(14,7))
sns.boxplot(x='진료과', y='건수', data=df)
plt.title('진료과별 상병 건수 분포 (지역별 혼합)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 상병별 총 건수 pareto chart
pareto_df = df.groupby('상병명')['건수'].sum().sort_values(ascending=False).reset_index()
pareto_df['누적합'] = pareto_df['건수'].cumsum()
pareto_df['누적백분율'] = 100 * pareto_df['누적합'] / pareto_df['건수'].sum()

fig, ax1 = plt.subplots(figsize=(12,6))

# bar plot
ax1.bar(pareto_df['상병명'], pareto_df['건수'], color='C0')
ax1.set_ylabel('건수', color='C0')

# xticks 설정
ax1.set_xticks(range(len(pareto_df['상병명'])))  # tick 위치 명시
ax1.set_xticklabels(pareto_df['상병명'], rotation=90)

# 보조 y축
ax2 = ax1.twinx()
ax2.plot(pareto_df['상병명'], pareto_df['누적백분율'], color='C1', marker='D', ms=5)
ax2.axhline(80, color='r', linestyle='dashed')  # 80% 기준선
ax2.set_ylabel('누적 백분율 (%)', color='C1')
ax2.set_ylim(0, 110)  # y축 범위 약간 여유 있게

plt.title('상병별 총 건수 Pareto Chart')
plt.subplots_adjust(bottom=0.25, top=0.90) 
plt.show()

# 특정 주요 상병의 지역별 발생 건수 버블 차트
top_disease = top_diseases[0]

region_df = df[df['상병명'] == top_disease].groupby('지역')['건수'].sum().reset_index()

plt.figure(figsize=(10,7))
sns.scatterplot(x='지역', y='건수', size='건수', sizes=(100, 1000), data=region_df, legend=False)
plt.xticks(rotation=45)
plt.title(f'{top_disease} 지역별 건수 버블 차트')
plt.xlabel('지역')
plt.ylabel('건수')
plt.grid(True)
plt.show()