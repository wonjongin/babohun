import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("merged_data/연령대별 20대상병정보.csv")

print(df.info())
print(df.head())
print(df.isna().sum())
print(df.duplicated().sum())

chronic_codes = df['상병코드'].unique()
print(chronic_codes)

chronic_codes = [
     'I10', 'E11', 'F20', 'F06', 'I20', 'I25', 'I50', 'I65', 'I67', 'I63', 'I69', 'G81', 'G82', 
    'C16', 'C18', 'C22', 'C25', 'C34', 'C61', 'C67', 'D09', 'D41', 'N18' 
]
df['만성질환여부'] = df['상병코드'].apply(lambda x: '만성질환' if x in chronic_codes else '비만성질환')

## 기술 통계
# 연령대별, 연도별, 지역별, 상병별 집계
age_sum = df.groupby('연령대')['건수'].sum().reset_index()
age_sum.to_csv("src/EDA/연령대별20대상병정보/연령대별_건수_합계.csv", index=False, encoding='utf-8-sig')

year_sum = df.groupby('년도')['건수'].sum().reset_index()
year_sum.to_csv("src/EDA/연령대별20대상병정보/연도별_건수_합계.csv", index=False, encoding='utf-8-sig')

region_sum = df.groupby('지역')['건수'].sum().reset_index()
region_sum.to_csv("src/EDA/연령대별20대상병정보/지역별_건수_합계.csv", index=False, encoding='utf-8-sig')

disease_sum = df.groupby('상병명')['건수'].sum().reset_index()
disease_sum.to_csv("src/EDA/연령대별20대상병정보/상병명별_건수_합계.csv", index=False, encoding='utf-8-sig')

stats = df.groupby('연령대')['건수'].agg(['sum', 'mean', 'std']).reset_index()
stats.to_csv("src/EDA/연령대별20대상병정보/연령대별_건수_통계.csv", index=False, encoding='utf-8-sig')
print(stats)

## 시각화
# 연령대별 주요 상병 Top N 막대 그래프
df_2023 = df[df['년도'] == 2023]
top_n = 5
for age in df_2023['연령대'].unique():
    temp = df_2023[df_2023['연령대'] == age]
    top_diseases = temp.groupby('상병명')['건수'].sum().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8,5))
    sns.barplot(x=top_diseases.values, y=top_diseases.index, palette='viridis')
    plt.title(f'{age} 연령대 주요 상병 Top {top_n}')
    plt.xlabel('건수')
    plt.ylabel('상병명')
    plt.show()

# 만성질환 비율 시각화(연령대별, 연도별)
chronic_age = df_2023.groupby(['연령대', '만성질환여부'])['건수'].sum().unstack().fillna(0)
chronic_age['만성질환비율'] = chronic_age['만성질환'] / (chronic_age['만성질환'] + chronic_age['비만성질환'])
chronic_age.reset_index(inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(x='연령대', y='만성질환비율', data=chronic_age, palette='coolwarm')
plt.title('연령대별 만성질환 비율')
plt.ylabel('만성질환 비율')
plt.ylim(0,1)
plt.show()

# 지역별, 연령대별 주요 상병 분포 히트맵
pivot = df_2023.groupby(['지역', '연령대'])['건수'].sum().unstack().fillna(0)

plt.figure(figsize=(12,6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('지역별 연령대별 총 건수 히트맵')
plt.ylabel('지역')
plt.xlabel('연령대')
plt.show()

# 전국적 주요 상병 발생 건수 연도별 추이 선 그래프
top_disease_names = df.groupby('상병명')['건수'].sum().sort_values(ascending=False).head(5).index.tolist()

plt.figure(figsize=(10,6))
for disease in top_disease_names:
    temp = df[df['상병명'] == disease].groupby('년도')['건수'].sum()
    plt.plot(temp.index, temp.values, marker='o', label=disease)

plt.title('주요 상병 연도별 발생 건수 추이')
plt.xlabel('년도')
plt.ylabel('건수')
plt.legend()
plt.show()

