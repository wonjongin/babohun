import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv")
df['진료비(천원)'] = pd.to_numeric(df['진료비(천원)'], errors='coerce').fillna(0)

df['실인원'] = pd.to_numeric(df['실인원'], errors='coerce')
df['연인원'] = pd.to_numeric(df['연인원'], errors='coerce')

## 기술통계:
# 3개년 전체 및 각 연도별 상병코드별 실인원/연인원/진료비 평균, 표준편차, 합계
# 단, 진료비는 NA 많아서 부정확할 가능성 높아보임
stats = df.groupby(['년도', '상병코드']).agg(
    실인원_합계 = ('실인원', 'sum'), 실인원_평균=('실인원', 'mean'), 실인원_표준편차=('실인원', 'std'),
    연인원_합계=('연인원', 'sum'), 연인원_평균=('연인원', 'mean'), 연인원_표준편차=('연인원', 'std'),
    진료비_합계 = ('진료비(천원)', 'sum'), 진료비_평균=('진료비(천원)', 'mean'), 진료비_표준편차=('진료비(천원)', 'std')
).reset_index()

total_stats = df.groupby('상병코드').agg(
    실인원_합계=('실인원', 'sum'), 실인원_평균=('실인원', 'mean'), 실인원_표준편차=('실인원', 'std'),
    연인원_합계=('연인원', 'sum'), 연인원_평균=('연인원', 'mean'), 연인원_표준편차=('연인원', 'std'),
    진료비_합계=('진료비(천원)', 'sum'), 진료비_평균=('진료비(천원)', 'mean'), 진료비_표준편차=('진료비(천원)', 'std')
).reset_index()
total_stats.insert(0, '년도', '전체')

combined_stats = pd.concat([stats, total_stats], ignore_index=True)
print(combined_stats)

# 연령대별 환자수 합계
age_cols = ['59이하', '60-64', '65-69', '70-79', '80-89', '90이상']
for col in age_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 연도별 + 연령대별
year_age = df.groupby('년도')[age_cols].sum().reset_index()
print("연도별 + 연령대별 환자 수 합계: \n", year_age)

# 병원별 + 연령대별
region_age = df.groupby('지역')[age_cols].sum().reset_index()
print("병원별 + 연령대별 환자 수 합계: \n", region_age)

# 연도별 + 병원별 + 연령대별
year_region_age = df.groupby(['년도', '지역'])[age_cols].sum().reset_index()
print("연도별 + 병원별 + 연령대별 환자 수 합계: \n", year_region_age)

## 시각화
# 상병코드별 실인원/연인원 추이 선 그래프 (연도별)
num_cols = ['실인원', '연인원', '59이하', '60-64', '65-69', '70-79', '80-89', '90이상']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

top_codes = df.groupby('상병코드')['연인원'].sum().nlargest(5).index
df_line = df[df['상병코드'].isin(top_codes)]

fig = px.line(df_line, x='년도', y='연인원', color='상병코드',
              title='상병코드별 연도별 연인원 추이',
              markers=True)
fig.show()

# 상병코드별 연령대별 환자 분포 히트맵
age_cols = ['59이하', '60-64', '65-69', '70-79', '80-89', '90이상']
heat_data = df[df['상병코드'].isin(top_codes)].groupby('상병코드')[age_cols].sum()

plt.figure(figsize=(10, 6))
sns.heatmap(heat_data, annot=True, fmt='.0f', cmap='Oranges')
plt.title('상병코드별 연령대별 환자 분포 히트맵')
plt.xlabel('연령대')
plt.ylabel('상병코드')
plt.tight_layout()
plt.show()

# 지역별 주요 상병코드별 환자수 분포 Choropleth Map 또는 버블 차트
df_bubble = df[df['상병코드'].isin(top_codes)].groupby(['지역', '상병코드'])['실인원'].sum().reset_index()

fig2 = px.scatter(df_bubble, x='지역', y='상병코드', size='실인원', color='상병코드',
                  title='지역별 주요 상병코드별 환자수 분포 (버블 차트)',
                  size_max=60)
fig2.show()

# 상병코드별 '연인원/실인원' 비율 계산 및 시각화 (막대 그래프 또는 산점도)
df['방문비율'] = df['연인원'] / df['실인원']
top20 = df.sort_values('방문비율', ascending=False).head(20)

chronic_codes = df['상병코드'].unique()
print(chronic_codes)
df['상병코드'] = df['상병코드'].str.strip()

chronic_codes = [ # 수정 필요..?
    'E11', 'E14', 'I10', 'I11', 'I20', 'I25', 'I48', 'I50', 'I63', 'I67',
    'I69', 'I70', 'N18', 'J44', 'J45', 'M17', 'M19', 'M43', 'M48', 'M50',
    'M51', 'M54', 'M75', 'M96', 'F00', 'F03', 'C16', 'C18', 'C22', 'C25',
    'C34', 'C61', 'C67', 'G20', 'G81', 'G82', 'G95', 'G62', 'K05', 'K29',
    'K63', 'K80', 'K81', 'L89', 'L30', 'L29', 'H25', 'H35', 'H90', 'F20',
    'F06', 'Z49', 'Z50', 'Z51', 'Z29'
]
df['만성질환여부'] = df['상병코드'].apply(
    lambda x: '만성' if any(x.strip().startswith(code) for code in chronic_codes) else '기타'
)
print(df[['상병코드', '만성질환여부']].head())


plt.figure(figsize=(14, 6))
sns.barplot(data=df, x='상병코드', y='방문비율', hue='만성질환여부')  # 또는 y='진료비'
plt.xticks(rotation=90)
plt.title('상병코드별 방문비율')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='만성질환여부', y='방문비율', palette='Set2')
plt.title('만성질환 vs 기타질환 방문비율 비교')
plt.tight_layout()
plt.show()

# 진료비 vs. (연인원/실인원 비율) 산점도: 방문 빈도와 진료비 간의 관계 파악.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='방문비율', y='진료비(천원)', hue='만성질환여부', alpha=0.7, palette='Set1')
plt.title('방문비율 vs 진료비 (상병코드별)')
plt.xlabel('연인원 / 실인원')
plt.ylabel('진료비')
plt.tight_layout()
plt.show()

# 입원/외래 분석
grouped = df.groupby(['상병코드', '구분'])['실인원'].sum().unstack(fill_value=0)
ratio = grouped.div(grouped.sum(axis=1), axis=0)
top10_codes = grouped.sum(axis=1).sort_values(ascending=False).head(10).index
ratio_top10 = ratio.loc[top10_codes]
ratio_top10.plot(kind='bar', stacked=True, figsize=(12,6), colormap='Set2')
plt.title('상병코드별 입원/외래 환자 비율 (상위 10개)')
plt.ylabel('비율')
plt.xticks(rotation=45)
plt.legend(title='구분')
plt.tight_layout()
plt.show()

# 상병코드별 평균 진료비 막대 그래프
df['진료비(천원)'] = df['진료비(천원)'].fillna(df['진료비(천원)'].mean())
stats = df.groupby('상병코드')['진료비(천원)'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(20)
plt.figure(figsize=(12,6))
plt.bar(stats.index, stats['mean'], yerr=stats['std'], capsize=5, color='skyblue')
plt.xticks(rotation=45)
plt.title('상병코드별 평균 진료비 (상위 20개)')
plt.ylabel('평균 진료비')
plt.tight_layout()
plt.show()

# 연령대별 평균 진료비 Boxplot
df['진료비'] = df['진료비(천원)'] * 1000
age_cols = ['59이하', '60-64', '65-69', '70-79', '80-89', '90이상']
df_melted = df.melt(
    id_vars=['상병코드', '상병명', '진료비'],
    value_vars=age_cols,
    var_name='연령대',
    value_name='인원수'
)

df_melted = df_melted.dropna(subset=['인원수'])
df_melted['1인당진료비'] = df_melted['진료비'] / df_melted['인원수']

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_melted, x='연령대', y='1인당진료비', palette='pastel')
plt.title('연령대별 1인당 진료비 분포')
plt.ylabel('1인당 진료비 (원)')
plt.xlabel('연령대')
plt.tight_layout()
plt.show()

# 65세 이상 노년층 다빈도 상위 질환
old_cols = ['65-69', '70-79', '80-89', '90이상']
df['노년층환자수'] = df[old_cols].sum(axis=1)
top_n = 10
top_disease = df.groupby(['상병코드', '상병명'])['노년층환자수'].sum().nlargest(top_n).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=top_disease, x='노년층환자수', y='상병명', palette='viridis')
plt.title('노년층(65세 이상) 다빈도 질환 Top 10')
plt.xlabel('환자 수')
plt.ylabel('질환명')
plt.tight_layout()
plt.show()

# 만성/비만성 연령대별 막대그래프
df_melted = df.melt(
    id_vars=['상병코드', '상병명', '만성질환여부'],
    value_vars=age_cols,
    var_name='연령대',
    value_name='환자수'
)
grouped = df_melted.groupby(['연령대', '만성질환여부'])['환자수'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped, x='연령대', y='환자수', hue='만성질환여부')
plt.title('연령대별 만성/비만성 질환자 분포')
plt.ylabel('환자 수')
plt.tight_layout()
plt.show()

# 만성질환의 연인원/실인원 비율 확인
chronic_df = df[df['상병코드'].isin(chronic_codes)].copy()
chronic_df['방문비율'] = chronic_df['연인원'] / chronic_df['실인원']
chronic_df_sorted = chronic_df.sort_values(by='방문비율', ascending=False)[['상병코드', '상병명', '방문비율']]
print(chronic_df_sorted.head(10))
chronic_df_sorted.to_csv("src/EDA/다빈도질환연령대분포/만성질환_방문비율_전체.csv", index=False, encoding='utf-8-sig')

# 연도별 Top N 질환 순위 변화
pivot = df.groupby(['년도', '상병코드'])['실인원'].sum().reset_index()
pivot['순위'] = pivot.groupby('년도')['실인원'].rank(method='first', ascending=False)
top_n = 10
top_codes = pivot[pivot['순위'] <= top_n]
pivot_chart = top_codes.pivot(index='상병코드', columns='년도', values='순위')

plt.figure(figsize=(10, 6))
for code in pivot_chart.index:
    plt.plot(pivot_chart.columns, pivot_chart.loc[code], marker='o', label=code)

plt.gca().invert_yaxis()
plt.title('연도별 질환 순위 변화 (Top 10)')
plt.xlabel('년도')
plt.ylabel('순위 (낮을수록 상위)')
plt.legend(title='상병코드', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()