import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rc('font', family='AppleGothic')
os.makedirs('imgs/EDA_연관D/001', exist_ok=True)

# 데이터 불러오기
df_resv = pd.read_csv('final_merged_data/연령대별 진료과 예약건수_2023.csv')
df_diag = pd.read_csv('final_merged_data/외래 진료과별 상위20 주요상병.csv')

# 연령대별 예약건수 합산 → 진료과별 전체 예약건수
age_cols = ["20대","30대","40대","50대","60대","70대","80대","90대"]
df_resv['예약합계'] = df_resv[age_cols].replace(',', '', regex=True).astype(float).sum(axis=1)
dept_resv = df_resv.groupby('진료과')['예약합계'].sum().sort_values(ascending=False)

dept_diag = df_diag.groupby('진료과')['건수'].sum().sort_values(ascending=False)

# 과거 진료 데이터에서 연도 컬럼이 있다면
if '년도' in df_diag.columns:
    # 2023년 데이터만 추출
    diag_2023 = df_diag[df_diag['년도'] == 2023].groupby('진료과')['건수'].sum()
    # 과거 연평균(2023년 제외, 연도별 합계의 평균)
    dept_year_sum = df_diag[df_diag['년도'] < 2023].groupby(['진료과', '년도'])['건수'].sum()
    diag_past = dept_year_sum.groupby('진료과').mean()
    # 비교 데이터프레임
    dept_compare = pd.DataFrame({
        '과거_연평균진료건수': diag_past,
        '2023_예약건수': dept_resv
    }).fillna(0)
else:
    # 연도 컬럼이 없으면 전체 누적 진료건수 vs 2023 예약건수로 비교
    dept_compare = pd.DataFrame({
        '과거_진료건수': dept_diag,
        '2023_예약건수': dept_resv
    }).fillna(0)

dept_compare = dept_compare.sort_values('2023_예약건수', ascending=False).head(15)

dept_compare.plot(kind='bar', figsize=(12,6))
plt.title('진료과별 과거 진료건수 vs 최근 예약건수')
plt.ylabel('건수')
plt.xlabel('진료과')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('imgs/EDA_연관D/001/진료과별_과거진료_vs_최근예약_막대.png')
plt.close()

target_dept = '가정의학과'

# 과거 주요 상병별 진료건수
top_diag = df_diag[df_diag['진료과']==target_dept].sort_values('건수', ascending=False).head(5)

# 최근 예약건수(진료과별, 연령대별)
resv_dept = df_resv[df_resv['진료과']==target_dept]
resv_sum = resv_dept[age_cols].replace(',', '', regex=True).astype(float).sum().sort_values(ascending=False)

# 그룹형 막대그래프
fig, ax1 = plt.subplots(figsize=(10,5))
sns.barplot(x=top_diag['상병명'], y=top_diag['건수'], color='skyblue', label='과거 진료건수', ax=ax1)
ax2 = ax1.twinx()
sns.lineplot(x=resv_sum.index, y=resv_sum.values, color='red', marker='o', label='최근 예약건수', ax=ax2)
ax1.set_ylabel('과거 진료건수')
ax2.set_ylabel('최근 예약건수(연령대별)')
plt.title(f'{target_dept} 주요 상병별 과거 진료 vs 최근 예약(연령대)')
plt.tight_layout()
plt.savefig(f'imgs/EDA_연관D/001/{target_dept}_상병별_과거진료_최근예약.png')
plt.close()

chronic_kw = ['당뇨','고혈압','만성','신부전','지질','골다공증','협심증','심부전','암','neoplasm','carcinoma','cancer']
acute_kw = ['급성','감염','염','폐렴','감기','출혈','통증','외상','골절']

def is_chronic(x):
    return any(kw in str(x) for kw in chronic_kw)
def is_acute(x):
    return any(kw in str(x) for kw in acute_kw)

df_diag['만성'] = df_diag['상병명'].apply(is_chronic)
df_diag['급성'] = df_diag['상병명'].apply(is_acute)

# 진료과별 만성/급성 비율
chronic_ratio = df_diag.groupby('진료과')['만성'].mean()
acute_ratio = df_diag.groupby('진료과')['급성'].mean()

# 진료과별 총 진료건수, 예약건수
dept_total = df_diag.groupby('진료과')['건수'].sum()
dept_resv = df_resv.groupby('진료과')['예약합계'].sum()

# 버블차트용 데이터프레임
bubble = pd.DataFrame({
    '예약건수': dept_resv,
    '만성비율': chronic_ratio,
    '급성비율': acute_ratio,
    '총진료건수': dept_total
}).fillna(0)

plt.figure(figsize=(10,7))
sns.scatterplot(data=bubble, x='예약건수', y='만성비율', size='총진료건수', hue='급성비율', sizes=(100,1000), palette='coolwarm', legend='brief')
plt.title('진료과별 예약건수 vs 만성질환비율 (버블=총진료건수, 색=급성비율)')
plt.xlabel('최근 예약건수')
plt.ylabel('만성질환 비율')
plt.tight_layout()
plt.savefig('imgs/EDA_연관D/001/진료과별_예약_만성비율_버블.png')
plt.close()

plt.figure(figsize=(10,7))
sns.scatterplot(data=bubble, x='예약건수', y='급성비율', size='총진료건수', hue='만성비율', sizes=(100,1000), palette='viridis', legend='brief')
plt.title('진료과별 예약건수 vs 급성질환비율 (버블=총진료건수, 색=만성비율)')
plt.xlabel('최근 예약건수')
plt.ylabel('급성질환 비율')
plt.tight_layout()
plt.savefig('imgs/EDA_연관D/001/진료과별_예약_급성비율_버블.png')
plt.close()