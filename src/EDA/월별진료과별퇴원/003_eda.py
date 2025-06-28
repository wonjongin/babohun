import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mtick

plt.rc('font', family='Pretendard')
os.makedirs('imgs/EDA_월별진료과별퇴원/003', exist_ok=True)

# 데이터 불러오기
df = pd.read_csv('final_merged_data/월별 진료과별 퇴원.csv')

# 진료과명 통일
respiratory_names = ['호흡기·알레르기내과', '호흡기알레르기내과', '호흡기내과']
df['퇴원과'] = df['퇴원과'].replace(respiratory_names, '호흡기내과')

# 월 컬럼이 문자열이면 정수로 변환
df['월'] = df['월'].astype(str).str.zfill(2)
df['월'] = df['월'].astype(int)

# 연도-월을 하나의 x축으로
df['연월'] = df['년도'].astype(str) + '-' + df['월'].astype(str).str.zfill(2)

# 1. 월별 전체 퇴원 환자 수 추이 (연도별)
plt.figure(figsize=(10,5))
sns.lineplot(data=df.groupby(['년도','월'])['건수'].sum().reset_index(), x='월', y='건수', hue='년도', marker='o')
plt.title('월별 전체 퇴원 환자 수 추이 (연도별)')
plt.xlabel('월')
plt.ylabel('퇴원 환자 수')
plt.xticks(range(1,13))
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/월별_전체_퇴원환자수_추이.png')
# plt.show()

# 2. 주요 진료과별 월별 퇴원 환자 수 추이 (상위 10개 진료과)
top_dept = df.groupby('퇴원과')['건수'].sum().sort_values(ascending=False).head(10).index
plt.figure(figsize=(12,6))
sns.lineplot(data=df[df['퇴원과'].isin(top_dept)].groupby(['퇴원과','월'])['건수'].sum().reset_index(),
             x='월', y='건수', hue='퇴원과', marker='o')
plt.title('주요 진료과별 월별 퇴원 환자 수 추이')
plt.xlabel('월')
plt.ylabel('퇴원 환자 수')
plt.xticks(range(1,13))
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/주요진료과별_월별_퇴원환자수_추이.png')
# plt.show()

# 3. 진료과별, 월별 퇴원 환자 수 히트맵
pivot = df.pivot_table(index='퇴원과', columns='월', values='건수', aggfunc='sum', fill_value=0)
plt.figure(figsize=(14,8))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('진료과별 월별 퇴원 환자 수 히트맵')
plt.xlabel('월')
plt.ylabel('진료과')
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/진료과별_월별_퇴원환자수_히트맵.png')
# plt.show()

# 4. 월별, 주상병명별 퇴원 환자 수 (상위 10개 주상병명)
top_disease = df.groupby('주상병명')['건수'].sum().sort_values(ascending=False).head(10).index
plt.figure(figsize=(12,6))
sns.lineplot(data=df[df['주상병명'].isin(top_disease)].groupby(['주상병명','월'])['건수'].sum().reset_index(),
             x='월', y='건수', hue='주상병명', marker='o')
plt.title('월별 상위 주상병명별 퇴원 환자 수')
plt.xlabel('월')
plt.ylabel('퇴원 환자 수')
plt.xticks(range(1,13))
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/월별_주상병명별_퇴원환자수_추이.png')
# plt.show()

# 5. 퇴원결과별 월별 분포 (상위 5개 퇴원결과)
top_result = df.groupby('퇴원결과')['건수'].sum().sort_values(ascending=False).head(5).index
plt.figure(figsize=(12,6))
sns.lineplot(data=df[df['퇴원결과'].isin(top_result)].groupby(['퇴원결과','월'])['건수'].sum().reset_index(),
             x='월', y='건수', hue='퇴원결과', marker='o')
plt.title('퇴원결과별 월별 퇴원 환자 수')
plt.xlabel('월')
plt.ylabel('퇴원 환자 수')
plt.xticks(range(1,13))
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/퇴원결과별_월별_퇴원환자수_추이.png')
# plt.show()

# 6. 연도별 월별 패턴 비교 (전체)
if '년도' in df.columns:
    plt.figure(figsize=(10,5))
    sns.lineplot(data=df.groupby(['년도','월'])['건수'].sum().reset_index(), x='월', y='건수', hue='년도', marker='o')
    plt.title('연도별 월별 퇴원 환자 수 패턴 비교')
    plt.xlabel('월')
    plt.ylabel('퇴원 환자 수')
    plt.xticks(range(1,13))
    plt.tight_layout()
    plt.savefig('imgs/EDA_월별진료과별퇴원/003/연도별_월별_퇴원환자수_비교.png')
    # plt.show()

# 7. 지역별 전체 퇴원 환자 수 (막대그래프)
plt.figure(figsize=(10,5))
region_sum = df.groupby('지역')['건수'].sum().sort_values(ascending=False)
sns.barplot(x=region_sum.index, y=region_sum.values, palette='viridis')
plt.title('지역별 전체 퇴원 환자 수')
plt.xlabel('지역')
plt.ylabel('퇴원 환자 수')
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/지역별_전체_퇴원환자수.png')
# plt.show()

# 8. 퇴원결과별 전체 퇴원 환자 수 (파이차트)
result_sum = df.groupby('퇴원결과')['건수'].sum().sort_values(ascending=False)
plt.figure(figsize=(8,8))
plt.pie(result_sum, labels=result_sum.index, autopct='%1.1f%%', startangle=140)
plt.title('퇴원결과별 전체 퇴원 환자 비율')
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/퇴원결과별_전체_비율.png')
# plt.show()

# 9. 진료과별 퇴원결과 분포 (누적 막대그래프)
pivot = df.pivot_table(index='퇴원과', columns='퇴원결과', values='건수', aggfunc='sum', fill_value=0)
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]  # 퇴원환자 많은 순 정렬
pivot.plot(kind='bar', stacked=True, figsize=(14,7), colormap='tab20')
plt.title('진료과별 퇴원결과 분포')
plt.xlabel('진료과')
plt.ylabel('퇴원 환자 수')
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/진료과별_퇴원결과_분포.png')
# plt.show()

# 10. 지역별 주요 진료과 Top10 (히트맵)
top_dept = df.groupby('퇴원과')['건수'].sum().sort_values(ascending=False).head(10).index
pivot = df[df['퇴원과'].isin(top_dept)].pivot_table(index='지역', columns='퇴원과', values='건수', aggfunc='sum', fill_value=0)
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('지역별 주요 진료과 퇴원 환자 수 히트맵')
plt.xlabel('진료과')
plt.ylabel('지역')
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/지역별_주요진료과_히트맵.png')
# plt.show()

# 11. 주상병명별 전체 퇴원 환자 수 Top10 (막대그래프)
top_disease = df.groupby('주상병명')['건수'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,5))
sns.barplot(x=top_disease.values, y=top_disease.index, palette='magma')
plt.title('주상병명별 전체 퇴원 환자 수 Top10')
plt.xlabel('퇴원 환자 수')
plt.ylabel('주상병명')
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/주상병명별_전체_퇴원환자수_Top10.png')
# plt.show()

# 12. 지역별 연도별 월별 퇴원환자수 추이 (라인플롯)
plt.figure(figsize=(16,6))
sns.lineplot(
    data=df.groupby(['지역', '연월'])['건수'].sum().reset_index(),
    x='연월', y='건수', hue='지역', marker='o'
)
plt.title('지역별 연도별 월별 퇴원환자수 추이')
plt.xlabel('연월')
plt.ylabel('퇴원 환자 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/지역별_연도별_월별_퇴원환자수_추이.png')
plt.close()

# 13. 월별 연도별 퇴원환자수 비교 (막대그래프)
plt.figure(figsize=(14,6))
sns.barplot(
    data=df.groupby(['년도', '월'])['건수'].sum().reset_index(),
    x='월', y='건수', hue='년도', palette=['#FF6B6B', '#4ECDC4']
)
plt.title('월별 연도별 퇴원환자수 비교')
plt.xlabel('월')
plt.ylabel('퇴원 환자 수')
plt.xticks(range(12))
plt.legend(title='년도')
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/월별_연도별_퇴원환자수_비교_막대.png')
plt.close()

# 14. 지역별 일반입원실(상급+일반) 대비 퇴원환자수 비율 시각화

# 병상 데이터 불러오기
beds = pd.read_csv('new_merged_data/병원_통합_데이터.csv')

# 병원통합데이터의 지역명을 '서울' → '중앙'으로 변경
beds['병원명'] = beds['병원명'].replace({'서울': '중앙', '서울_호스피스': '중앙_호스피스'})
beds['지역'] = beds['병원명'].str.replace('_호스피스', '', regex=False)

# 지역별 일반입원실 병상수 합계
beds_grouped = beds.groupby('지역')[['일반입원실_상급', '일반입원실_일반']].sum()
beds_grouped['일반입원실_합계'] = beds_grouped['일반입원실_상급'] + beds_grouped['일반입원실_일반']

# 퇴원환자수 데이터 (지역별)
discharge = df.groupby('지역')['건수'].sum().to_frame('퇴원환자수')

# 병상수와 퇴원환자수 병합
merged = beds_grouped[['일반입원실_합계']].join(discharge, how='inner')
merged['퇴원/병상 비율'] = merged['퇴원환자수'] / merged['일반입원실_합계']

# 시각화
plt.figure(figsize=(14,6))
ax = merged[['일반입원실_합계', '퇴원환자수']].plot(kind='bar', secondary_y='퇴원환자수', figsize=(14,6))
ax.set_xticklabels(merged.index, rotation=0)
ax.set_ylabel('일반입원실 병상수')
ax.right_ax.set_ylabel('퇴원환자수')
plt.title('지역별 일반입원실(상급+일반) 병상수와 퇴원환자수')

# 비율 라벨 추가
for i, (idx, row) in enumerate(merged.iterrows()):
    ax.text(i, row['일반입원실_합계']+max(merged['일반입원실_합계'])*0.02, f"{row['퇴원/병상 비율']:.1f}", 
            ha='center', va='bottom', fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/지역별_입원병상_퇴원환자수_비율.png')
plt.close()

# 비율만 따로 시각화
plt.figure(figsize=(10,5))
sns.barplot(x=merged.index, y=merged['퇴원/병상 비율'], palette='coolwarm')
plt.title('지역별 퇴원환자수/일반입원실(상급+일반) 병상수 비율')
plt.xlabel('지역')
plt.ylabel('퇴원/병상 비율')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
for i, v in enumerate(merged['퇴원/병상 비율']):
    plt.text(i, v, f'{v:.2%}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/지역별_퇴원_병상_비율.png')
plt.close()

# 15. 병원별 연도별 퇴원/병상 비율 비교 (막대그래프)
# 연도별 퇴원환자수 계산
discharge_by_year = df.groupby(['지역', '년도'])['건수'].sum().reset_index()

# 병상수와 연도별 퇴원환자수 병합
merged_by_year = beds_grouped[['일반입원실_합계']].reset_index()
merged_by_year = merged_by_year.merge(discharge_by_year, on='지역', how='inner')
merged_by_year['퇴원/병상 비율'] = merged_by_year['건수'] / merged_by_year['일반입원실_합계']

# 시각화
plt.figure(figsize=(14,6))
sns.barplot(
    data=merged_by_year,
    x='지역', y='퇴원/병상 비율', hue='년도', palette=['#FF6B6B', '#4ECDC4']
)
plt.title('병원별 연도별 퇴원환자수/일반입원실(상급+일반) 병상수 비율')
plt.xlabel('병원')
plt.ylabel('퇴원/병상 비율')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend(title='년도')
plt.xticks(rotation=45)

# 비율 값 표시
for i, row in merged_by_year.iterrows():
    plt.text(
        merged_by_year[merged_by_year['년도'] == row['년도']].index.get_loc(i), 
        row['퇴원/병상 비율'], 
        f'{row["퇴원/병상 비율"]:.1%}', 
        ha='center', va='bottom', fontsize=9
    )

plt.tight_layout()
plt.savefig('imgs/EDA_월별진료과별퇴원/003/병원별_연도별_퇴원_병상_비율_비교.png')
plt.close()