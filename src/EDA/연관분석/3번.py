import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df1 = pd.read_csv("merged_data/과별 퇴원환자 20대 주진단_열병합.csv")
df2 = pd.read_csv("merged_data/월별 진료과별 퇴원_열병합.csv")
df3 = pd.read_csv("new_merged_data/병상정보.csv")
df4 = pd.read_csv("new_merged_data/병원_통합_데이터.csv")

print(df1.head(3))
print(df2.head(3))
print(df3.head(3))
print(df4.head(3))

## 진료과별 연간/월간 퇴원 환자 수 집계
annual_demand = df1.groupby(['년도', '진료과'])['실인원'].sum().reset_index()
annual_demand.to_csv("src/EDA/연관분석/연도별_진료과별_실인원합계.csv", index=False, encoding='utf-8-sig')

monthly_demand = df2.groupby(['월', '퇴원과'])['건수'].sum().reset_index()
monthly_demand.to_csv("src/EDA/연관분석/월별_퇴원과별_건수합계.csv", index=False, encoding='utf-8-sig')
print(annual_demand, monthly_demand)

## 지역, 병상 구분 별 병상 합계 집계
regional_beds = df3.groupby(['지역', '병상 구분'])['병상 합계'].sum().reset_index()
regional_beds.to_csv("src/EDA/연관분석/지역_병상구분별_병상합계.csv", index=False, encoding='utf-8-sig')
print(regional_beds)

## 의료진 수 집계
doc_cols = [col for col in df4.columns if col.endswith('전문의수')]
dept_names = [col.replace('_전문의수', '') for col in doc_cols]
doc_sum = pd.DataFrame({
    '진료과': dept_names,
    '전문의수': df4[doc_cols].sum(axis=0).values
})
doc_sum.to_csv("src/EDA/연관분석/진료과별_전문의수_합계.csv", index=False, encoding='utf-8-sig')
print(doc_sum)

## 진료과별 병상/의료진 대비 퇴원환자 수 비교 시각화
# 연도별 진료과별 퇴원환자 수
annual_demand = df1.groupby(['년도', '진료과'])['실인원'].sum().reset_index()

# 진료과별 전문의 수 합계
doc_cols = [col for col in df4.columns if col.endswith('전문의수')]
dept_names = [col.replace('_전문의수', '') for col in doc_cols]
doc_sum = pd.DataFrame({
    '진료과': dept_names,
    '전문의수': df4[doc_cols].sum(axis=0).values
})

# 연도별 진료과별 환자 수와 전문의 수 결합
merged = pd.merge(annual_demand, doc_sum, on='진료과', how='left').fillna(0)

plt.figure(figsize=(14,8))
sns.lineplot(data=merged, x='년도', y='실인원', hue='진료과', style='진료과', markers=True, dashes=False, legend=False)
plt.title('2019~2023년 진료과별 연간 퇴원환자 수 추이')
plt.ylabel('퇴원 환자 수')
plt.xlabel('년도')
plt.show()

merged = pd.merge(annual_demand, doc_sum, on='진료과', how='left').fillna(0)

plt.figure(figsize=(14,8))
sns.lineplot(data=merged, x='년도', y='실인원', hue='진료과', marker='o')
plt.title('2019~2023년 진료과별 연간 퇴원환자 수 추이')
plt.ylabel('퇴원 환자 수')
plt.xlabel('년도')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 연도별 전문의 수 대비 퇴원환자 수 비교(19~23)
plt.figure(figsize=(12,6))
for year in sorted(merged['년도'].unique()):
    data_year = merged[merged['년도'] == year].copy()
    # 전문의수에 약간의 jitter 추가 (x축 겹침 방지)
    data_year['전문의수_jitter'] = data_year['전문의수'] + np.random.uniform(-0.2, 0.2, size=len(data_year))
    plt.scatter(data_year['전문의수_jitter'], data_year['실인원'], label=year, alpha=0.7)

plt.xlabel('전문의 수 (Jitter 적용)')
plt.ylabel('퇴원 환자 수')
plt.title('2019~2023년 진료과별 전문의 수 대비 퇴원환자 수 변화')
plt.legend(title='년도')
plt.tight_layout()
plt.show()

## 월별 진료과별 퇴원환자 수 변화(22~23)
df2['월'] = pd.to_datetime(df2['월'].str[:7], errors='coerce')
monthly_demand = df2[(df2['월'].dt.year >= 2022) & (df2['월'].dt.year <= 2023)]

# 예시: '순환기내과' 월별 퇴원환자 수
dept = '순환기내과'
monthly_dept = monthly_demand[monthly_demand['퇴원과'] == dept]
monthly_agg = monthly_dept.groupby('월')['건수'].sum().reset_index()

plt.figure(figsize=(14,5))
sns.lineplot(data=monthly_agg, x='월', y='건수')
plt.title(f'{dept} 2022~2023 월별 퇴원 환자 수 변화')
plt.xlabel('월')
plt.ylabel('퇴원 환자 수')
plt.show()

## 병상 정보 연도별 지역별 추세(19~23)
# 지역별로 서브플롯 분할
df3['년도'] = df3['년도'].astype(int)
beds_yearly = df3.groupby(['년도', '지역'])['병상 합계'].sum().reset_index()

regions = beds_yearly['지역'].unique()
n_regions = len(regions)

ncols = 3
nrows = (n_regions + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4), sharey=True)

for i, region in enumerate(regions):
    ax = axes[i // ncols, i % ncols]
    data = beds_yearly[beds_yearly['지역'] == region]
    sns.barplot(data=data, x='년도', y='병상 합계', ax=ax)
    ax.set_title(f'{region} 지역 병상 합계 추이')
    ax.set_xlabel('년도')
    ax.set_ylabel('병상 합계')

for j in range(i+1, nrows*ncols):
    fig.delaxes(axes[j // ncols, j % ncols])

plt.tight_layout()
plt.show()

# 한 그래프에 지역별 선 그래프
plt.figure(figsize=(14,7))
sns.lineplot(data=beds_yearly, x='년도', y='병상 합계', hue='지역', marker='o')
plt.title('2019~2023년 지역별 병상 합계 추이')
plt.xlabel('년도')
plt.ylabel('병상 합계')
plt.legend(title='지역', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

## 연도별 특정 진료과 퇴원환자수, 총 병상수, 의료진수 변화
# 예: 특정 진료과 선택
target_dept = '순환기내과'

df1_agg = df1.groupby(['년도', '진료과'])['실인원'].sum().reset_index()
dept_patient = df1_agg[df1_agg['진료과'] == target_dept]

df3['년도'] = df3['년도'].astype(int)
beds_yearly = df3.groupby('년도')['병상 합계'].sum().reset_index()

doc_cols = [col for col in df4.columns if col.endswith('전문의수')]
dept_names = [col.replace('_전문의수', '') for col in doc_cols]
doc_sum = pd.DataFrame({
    '진료과': dept_names,
    '전문의수': df4[doc_cols].sum(axis=0).values
})
dept_doc = doc_sum[doc_sum['진료과'] == target_dept]

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(dept_patient['년도'], dept_patient['실인원'], 'o-', color='tab:blue', label='퇴원 환자 수')
ax1.set_xlabel('년도')
ax1.set_ylabel('퇴원 환자 수', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(beds_yearly['년도'], beds_yearly['병상 합계'], 's-', color='tab:green', label='총 병상 수')
ax2.set_ylabel('총 병상 수', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

ept_doc = doc_sum[doc_sum['진료과'] == target_dept]

if len(dept_doc) == 0:
    doc_num = 0
else:
    doc_num = dept_doc['전문의수'].values[0]

# 이후 scatter에 doc_num 사용
ax2.scatter(dept_patient['년도'], [doc_num]*len(dept_patient), 
            color='tab:red', marker='D', label='의료진 수')

fig.suptitle(f'{target_dept} 연도별 퇴원환자 수, 총 병상 수, 의료진 수 변화')
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
fig.tight_layout(rect=[0,0,1,0.95])
plt.show()

## 연도별 진료과별 병상 점유율, 의료진당 퇴원 환자 수 막대 그래프
df2['월'] = df2['월'].astype(str)
df2['월'] = pd.to_datetime(df2['월'].str[:7], errors='coerce')
df2 = df2.dropna(subset=['월'])

df2['년도'] = df2['월'].dt.year

doc_cols = [col for col in df4.columns if col.endswith('전문의수')]
dept_names = [col.replace('_전문의수', '') for col in doc_cols]
doc_sum = pd.DataFrame({
    '진료과': dept_names,
    '전문의수': df4[doc_cols].sum(axis=0).values
})
years = sorted(df2['년도'].unique())  # 2022, 2023

for year in years:
    df2_year = df2[df2['년도'] == year]
    # 진료과별 월별 '건수' 합계 → 연간 집계
    dept_patients = df2_year.groupby('퇴원과')['건수'].sum().reset_index()
    dept_patients = dept_patients.rename(columns={'퇴원과':'진료과', '건수':'실인원'})
    
    # 진료과별 의료진 수와 병합
    merged = pd.merge(dept_patients, doc_sum, on='진료과', how='left').fillna(0)
    merged['의료진당_퇴원환자수'] = merged.apply(
        lambda x: x['실인원'] / x['전문의수'] if x['전문의수'] > 0 else None, axis=1)
    
    # 그래프 그리기
    plt.figure(figsize=(14,7))
    sns.barplot(data=merged.sort_values('의료진당_퇴원환자수', ascending=False),
                x='진료과', y='의료진당_퇴원환자수')
    plt.xticks(rotation=45)
    plt.title(f'{year}년 진료과별 의료진당 퇴원 환자 수 (df2 기준)')
    plt.ylabel('의료진당 퇴원 환자 수')
    plt.xlabel('진료과')
    plt.tight_layout()
    plt.show()