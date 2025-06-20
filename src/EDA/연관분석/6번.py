import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df1 = pd.read_csv("data/연령대별 진료과 예약건수/한국보훈복지의료공단_보훈병원 연령대별 진료과 예약건수_20231231.csv", encoding='euc-kr')
df2 = pd.read_csv("new_merged_data/외래 진료과별 상위20 주요상병_진료과통일.csv")
df3 = pd.read_csv("new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv")

print(df1.head(3))
print(df2.head(3))
print(df3.head(3))

# df1 연령대별 예약건수 long format 변환
age_cols_1 = ['20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대']
df1_long = df1.melt(id_vars=['병원명', '진료과'], value_vars=age_cols_1,
                    var_name='연령대', value_name='예약건수')
df1_long = df1_long[df1_long['예약건수'] > 0].reset_index(drop=True)

# df2 진료과별 주요 상병별 건수(외래 건수) 집계
df2_sum = df2.groupby('진료과')['건수'].sum().reset_index().rename(columns={'건수':'외래건수'})

# df3 진료비 long format 변환
age_map_3 = {'65-69': '60대', '70-79': '70대', '80-89': '80대', '90이상': '90대'}
age_cols_3 = ['65-69', '70-79', '80-89', '90이상']

df3_long = df3.melt(id_vars=['년도', '지역', '상병코드', '상병명'], value_vars=age_cols_3,
                   var_name='연령대_raw', value_name='진료비')

df3_long['연령대'] = df3_long['연령대_raw'].map(age_map_3)
df3_long = df3_long.dropna(subset=['진료비'])
df3_long = df3_long[df3_long['진료비'] > 0]

# 진료과 - 상병명 매핑 (df2에서 추출)
disease_dept_map = df2[['상병명', '진료과']].drop_duplicates()

# df3_long에 진료과 추가
df3_long = pd.merge(df3_long, disease_dept_map, on='상병명', how='left')

## 예약건수와 진료비 기반 외래진료 건수 비교
# 예약건수(df1_long)와 진료비(df3_long)를 진료과, 연령대 기준으로 합치기
analysis_df = pd.merge(df1_long, df3_long, on=['진료과', '연령대'], how='inner')

# 외래건수가 df2_sum에 있으므로 연령대 정보 없지만, 진료과 기준 외래건수 매칭(참고용)
analysis_df = pd.merge(analysis_df, df2_sum, on='진료과', how='left')

# 진료비 대비 예약건수 비교를 위한 비율 계산
analysis_df['예약대비진료비비율'] = analysis_df['진료비'] / analysis_df['예약건수']

## 예약과 외래 건수 차이 분석
# df1_long과 df2_sum 진료과별 총 예약 vs 외래건수 합치기 (연령대별 아님)
reserv_total = df1_long.groupby('진료과')['예약건수'].sum().reset_index()
outpatient_total = df2_sum.copy()

diff_df = pd.merge(reserv_total, outpatient_total, on='진료과', how='outer').fillna(0)
diff_df['예약-외래_차이'] = diff_df['예약건수'] - diff_df['외래건수']

## 시각화
# 연령대별 진료과 예약건수 vs 진료비 산점도
plt.figure(figsize=(10,6))
sns.scatterplot(data=analysis_df, x='예약건수', y='진료비', hue='연령대', alpha=0.7)
plt.title('연령대별 진료과 예약건수와 진료비 산점도')
plt.xlabel('예약건수')
plt.ylabel('진료비 (천원)')
plt.legend(title='연령대')
plt.show()

# 연령대-진료과별 예약건수와 진료비 히트맵
top_depts = analysis_df.groupby('진료과')['예약건수'].sum().sort_values(ascending=False).head(10).index
heatmap_data = analysis_df[analysis_df['진료과'].isin(top_depts)].pivot_table(index='진료과', columns='연령대', values='예약대비진료비비율', aggfunc='mean')

plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='RdYlGn', center=1)
plt.title('상위 10 진료과 연령대별 예약 대비 진료비 비율 히트맵')
plt.xlabel('연령대')
plt.ylabel('진료과')
plt.show()

## 예약 많은데 진료비 적은 진료과
low_ratio_df = analysis_df[analysis_df['예약대비진료비비율'] < 0.5]
low_ratio_summary = low_ratio_df.groupby('진료과')['예약대비진료비비율'].mean().sort_values()

plt.figure(figsize=(10,6))
sns.barplot(x=low_ratio_summary.values, y=low_ratio_summary.index, hue=low_ratio_summary.index, palette='Reds_r', legend=False)
plt.title('예약 대비 진료비 비율 낮은 진료과 (잠재적 예약 후 미방문)')
plt.xlabel('평균 예약 대비 진료비 비율')
plt.ylabel('진료과')
plt.show()

# 연령대별 예약건수 상위 진료과 5개와 다빈도 질환 분포 비교
top5_depts_by_age = df1_long.groupby(['연령대', '진료과'])['예약건수'].sum().reset_index()
top5_depts_by_age = top5_depts_by_age.groupby('연령대').apply(
    lambda x: x.nlargest(5, '예약건수'),
    include_groups=False
).reset_index()

# 2. 컬럼명 공백 제거 (예방 차원)
top5_depts_by_age.columns = top5_depts_by_age.columns.str.strip()

# 3. '연령대' 컬럼 존재 여부 확인 출력 (디버깅용)
print(top5_depts_by_age.columns)
print(top5_depts_by_age.head())

# 4. 각 진료과별로 plot 그리기 (빈 데이터는 건너뜀)
plt.figure(figsize=(12, 8))
for dept in top5_depts_by_age['진료과'].unique():
    temp = top5_depts_by_age[top5_depts_by_age['진료과'] == dept]
    
    # 빈 데이터 확인
    if temp.empty:
        print(f"[경고] '{dept}'에 해당하는 데이터가 없습니다. 건너뜁니다.")
        continue
    
    plt.plot(temp['연령대'], temp['예약건수'], marker='o', label=f'{dept} 예약건수')

plt.xlabel('연령대')
plt.ylabel('예약건수')
plt.title('연령대별 진료과 상위 5 예약건수')
plt.legend()
plt.grid(True)
plt.show()