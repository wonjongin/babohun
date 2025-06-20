import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("new_merged_data/국가보훈부_보훈대상자 성별연령별 실인원현황.csv", encoding='euc-kr')

# 통계량 계산
cols = ['합계', '남', '여']
for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

print(data.dtypes)

statistic_monthly = data.groupby(['기준일', '본인유족구분'])[['합계']].agg(['mean', 'min', 'median', 'max', 'std', 'var'])
print("월별 통계: \n", statistic_monthly)
statistic_monthly.to_csv("src/EDA/보훈대상자성별연령별실인원현황/보훈대상자현황_월별통계.csv", encoding='utf-8-sig')

statistic_age = data.groupby(['연령구분', '본인유족구분'])[['합계']].agg(['mean', 'min', 'median', 'max', 'std', 'var'])
print("연령대별 통계: \n", statistic_age)
statistic_age.to_csv("src/EDA/보훈대상자성별연령별실인원현황/보훈대상자현황_연령대별통계.csv", encoding='utf-8-sig')

data_long = data.melt(
    id_vars=['기준일', '본인유족구분', '연령구분'], 
    value_vars=['남', '여'], 
    var_name='성별', 
    value_name='인원수'
)
print(data_long.head(5))

statistic_gender = data_long.groupby(['성별', '본인유족구분'])['인원수'].agg(['mean', 'min', 'median', 'max', 'std', 'var'])
print("성별 통계 : \n", statistic_gender)
statistic_gender.to_csv("src/EDA/보훈대상자성별연령별실인원현황/보훈대상자현황_성별통계.csv", encoding='utf-8-sig')

statistic_monthly_gender = data_long.groupby(['기준일', '본인유족구분', '성별'])['인원수'].agg(['mean', 'min', 'median', 'max', 'std', 'var'])
print("월별 성별 통계: \n", statistic_monthly_gender)
statistic_monthly_gender.to_csv("src/EDA/보훈대상자성별연령별실인원현황/보훈대상자현황_월별성별통계.csv", encoding='utf-8-sig')

statistic_age_gender = data_long.groupby(['연령구분', '본인유족구분', '성별'])['인원수'].agg(['mean', 'min', 'median', 'max', 'std', 'var'])
print("연령대별 성별 통계: \n", statistic_age_gender)
statistic_age_gender.to_csv("src/EDA/보훈대상자성별연령별실인원현황/보훈대상자현황_연령별성별통계.csv", encoding='utf-8-sig')

## 시각화
# 연령대별 인원변화 누적 막대 그래프
data['기준일'] = pd.to_datetime(data['기준일'])
data['연도'] = data['기준일'].dt.year
age_order=['0~4세', '5~9세', '10~14세', '15~19세', '20~24세', '25~29세', '30~34세', '35~39세', '40~44세', '45~49세',
           '50~54세', '55~59세', '60~64세', '65~69세', '70~74세', '75~79세', '80~84세', '85~89세', '90~94세', '95~99세', '100세 이상', '해당없음']
data['연령구분'] = pd.Categorical(data['연령구분'], categories=age_order, ordered=True)
age_yearly = data.groupby(['연도', '연령구분'])['합계'].sum().reset_index()
pivot_age = age_yearly.pivot(index='연도', columns='연령구분', values='합계')
print(pivot_age.head())
palette = sns.color_palette("Set3", n_colors=len(pivot_age.columns))
ax=pivot_age.plot(kind='bar', stacked=True, figsize=(12, 100), color=palette)
ax.ticklabel_format(style='plain', axis='y')
plt.title('연령대별 인원 변화 (그룹형 막대 그래프)')
plt.xlabel('연도')
plt.ylabel('인원수')
plt.legend(title='연령구분')
plt.show()

age_group_map = {
    '0~4세': '0대',
    '5~9세': '0대',
    '10~14세': '10대',
    '15~19세': '10대',
    '20~24세': '20대',
    '25~29세': '20대',
    '30~34세': '30대',
    '35~39세': '30대',
    '40~44세': '40대',
    '45~49세': '40대',
    '50~54세': '50대',
    '55~59세': '50대',
    '60~64세': '60대',
    '65~69세': '60대',
    '70~74세': '70대',
    '75~79세': '70대',
    '80~84세': '80대',
    '85~89세': '80대',
    '90~94세': '90대',
    '95~99세': '90대',
    '100세 이상': '100대 이상',
    '해당없음': '기타'
}
data['연령대구분'] = data['연령구분'].map(age_group_map)
big_age_order = ['0대', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대', '100대 이상', '기타']
data['연령대구분'] = pd.Categorical(data['연령대구분'], categories=big_age_order, ordered=True)
age_yearly_big = data.groupby(['연도', '연령대구분'], observed=False)['합계'].sum().reset_index()
pivot_age_big = age_yearly_big.pivot(index='연도', columns='연령대구분', values='합계')
colors = ['#205781', '#4F959D', '#98D2C0', '#FFE99A', '#ADB2D4', '#F7CFD8', '#86A788', '#347433', 
          '#FFC107', '#F5ECD5', '#F3A26D', '#FFD6BA']
pivot_age_big.plot(kind='bar', stacked=True, figsize=(12, 6), color=colors)
plt.xlabel('연도')
plt.ylabel('인원수')
plt.title('연령대별 인원 변화 (그룹형 막대 그래프)')
plt.legend(title='연령대구분', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 연령대별 인원 변화 선 그래프
age_summary = data.groupby(['연도', '연령대구분'], observed=False)['합계'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=age_summary, x='연도', y='합계', hue='연령대구분', palette='tab10')
plt.title('연도별 연령대 인원 변화')
plt.xlabel('연도')
plt.ylabel('인원수')
plt.legend(title='연령대', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 연령대별 인원수 히스토그램(KDE)
age_sum = data.groupby('연령대구분', observed=False)['합계'].sum().reset_index()
age_sum = age_sum.sort_values('연령대구분')
plt.figure(figsize=(10,6))
sns.histplot(data=age_sum, x='연령대구분', weights='합계', bins=len(big_age_order), kde=True, discrete=True, color='#8E7DBE')

plt.title('연령대별 인원수 히스토그램(밀도 곡선)')
plt.xlabel('연령대')
plt.ylabel('인원수')
plt.xticks(rotation=45)
plt.show()

# 연령대별 본인유족구분 비교 누적 막대 그래프
df_grouped = data.groupby(['연령대구분', '본인유족구분'], observed=False)['합계'].sum().reset_index()
pivot_df = df_grouped.pivot(index='연령대구분', columns='본인유족구분', values='합계').fillna(0)
pivot_df = pivot_df.reindex(big_age_order)
colors = ['#B8CFCE', '#F7CFD8'] 
pivot_df.plot(kind='bar', stacked=False, figsize=(12,6), color=colors)

plt.title('연령대별 본인유족구분 비교 누적 막대 그래프')
plt.xlabel('연령대')
plt.ylabel('인원수')
plt.xticks(rotation=45)
plt.legend(title='Status')
plt.tight_layout()
plt.show()

# 본인유족 구분 비율 파이 차트
for year in data['연도'].unique():
    df_year = data[data['연도'] == year]
    total_by_group = df_year.groupby('본인유족구분', observed=False)['합계'].sum()
    
    plt.figure(figsize=(6,6))
    plt.pie(total_by_group, labels=total_by_group.index, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f'{year} Year - 본인/유족 비율')
    plt.axis('equal')
    plt.show()

# 65세 이상 노년층 인구 비율 변화 선 그래프
young_ages = ['0~4세', '5~9세', '10~14세', '15~19세', '20~24세', '25~29세', '30~34세', '35~39세']
middle_ages = ['40~44세', '45~49세', '50~54세', '55~59세', '60~64세']
senior_ages = ['65~69세', '70~74세', '75~79세', '80~84세', '85~89세', '90~94세', '95~99세', '100세 이상']
grouped = data.groupby(['연도', '연령구분'], observed=False)['합계'].sum().reset_index()

def get_age_group(row):
    if row['연령구분'] in young_ages:
        return '청년층'
    elif row['연령구분'] in middle_ages:
        return '중장년층'
    elif row['연령구분'] in senior_ages:
        return '노년층'
    else:
        return '기타'

grouped['연령구분'] = grouped.apply(get_age_group, axis=1)
group_summary = grouped.groupby(['연도', '연령구분'], observed=True)['합계'].sum().reset_index()

colors=['#F7CFD8','#A6D6D6','#8E7DBE']
plt.figure(figsize=(12, 6))
sns.lineplot(data=group_summary, x='연도', y='합계', hue='연령구분', marker='o', color=colors)
plt.title('연도별 연령층 비율 변화')
plt.ylabel('인원수')
plt.show()

# 평균 연령 변화 추이 선 그래프
# 대표연령 매핑
age_mapping = {
    '0~4세': 2, '5~9세': 7, '10~14세': 12, '15~19세': 17,
    '20~24세': 22, '25~29세': 27, '30~34세': 32, '35~39세': 37,
    '40~44세': 42, '45~49세': 47, '50~54세': 52, '55~59세': 57,
    '60~64세': 62, '65~69세': 67, '70~74세': 72, '75~79세': 77,
    '80~84세': 82, '85~89세': 87, '90~94세': 92, '95~99세': 97,
    '100세이상': 102, '해당없음': None
}
data['대표연령'] = data['연령구분'].map(age_mapping)

# 가중평균 함수 정의
def weighted_mean(df, value_col, weight_col):
    valid = df.dropna(subset=[value_col, weight_col])
    return (valid[value_col] * valid[weight_col]).sum() / valid[weight_col].sum()

# 그룹별 가중평균 계산
result = data.groupby(['연도', '본인유족구분']).apply(
    lambda x: weighted_mean(x, '대표연령', '합계')
).reset_index(name='가중평균연령')

# 전체(본인+유족) 가중평균 계산
mean_all = data.groupby(['연도']).apply(
    lambda g: weighted_mean(g, '대표연령', '합계')
).reset_index(name='가중평균연령')
mean_all['본인유족구분'] = '전체'

# 합치기
mean_combined = pd.concat([result, mean_all], ignore_index=True)

print(mean_combined.pivot(index='연도', columns='본인유족구분', values='가중평균연령'))

plt.figure(figsize=(12,6))
sns.lineplot(data=mean_combined, x='연도', y='가중평균연령', hue='본인유족구분', 
             marker='o', style='본인유족구분', dashes=False)
plt.title('보훈대상자 가중평균 연령 추이')
plt.xlabel('연도')
plt.ylabel('가중평균 연령')
plt.legend(title='구분')
plt.tight_layout()
plt.show()
