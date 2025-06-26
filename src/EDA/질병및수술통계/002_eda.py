import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import folium
import json
import geopandas as gpd

plt.rc('font', family='Pretendard')
os.makedirs('imgs/EDA_질병및수술통계/002', exist_ok=True)

# 데이터 불러오기
df = pd.read_csv('final_merged_data/질병 및 수술 통계.csv')
print(df.head())
print(df.info())
print(df['구분'].unique())
print(df['년도'].unique())


'''
# 1. 전체 국비/사비 비율 파이차트
total_gukbi = df['국비'].sum()
total_sabi = df['사비'].sum()
plt.figure(figsize=(6,6))
plt.pie([total_gukbi, total_sabi], labels=['국비', '사비'], autopct='%1.1f%%', startangle=90)
plt.title('전체 국비/사비 인원 비율')
plt.savefig('imgs/EDA_질병및수술통계/002/전체_국비_사비_비율.png')
# plt.show()

# 2. 주요 상병별 국비/사비 비율 (상위 N개)
topN = 10
top_disease = df.groupby('상병명')[['국비','사비']].sum()
top_disease['합계'] = top_disease['국비'] + top_disease['사비']
top_disease = top_disease.sort_values('합계', ascending=False).head(topN)

top_disease[['국비','사비']].plot(kind='bar', stacked=True, figsize=(12,6))
plt.title(f'상위 {topN} 상병별 국비/사비 인원수(누적 막대)')
plt.ylabel('인원수')
plt.xlabel('상병명')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/상위상병별_국비_사비_누적막대.png')
# plt.show()

# 3. 국비 지원 인원수 많은 상위 N개 상병
top_gukbi = df.groupby('상병명')['국비'].sum().sort_values(ascending=False).head(topN)
plt.figure(figsize=(10,5))
sns.barplot(x=top_gukbi.values, y=top_gukbi.index, palette='Blues_r')
plt.title(f'국비 지원 인원수 많은 상위 {topN} 상병')
plt.xlabel('국비 인원수')
plt.ylabel('상병명')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/국비상위상병.png')
# plt.show()

# 4. 사비 부담 인원수 많은 상위 N개 상병
top_sabi = df.groupby('상병명')['사비'].sum().sort_values(ascending=False).head(topN)
plt.figure(figsize=(10,5))
sns.barplot(x=top_sabi.values, y=top_sabi.index, palette='Reds_r')
plt.title(f'사비 부담 인원수 많은 상위 {topN} 상병')
plt.xlabel('사비 인원수')
plt.ylabel('상병명')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/사비상위상병.png')
# plt.show()

# 5. 지역별 총 진료 인원수 중 국비 비율 (막대그래프)
region = df.groupby('지역')[['국비','사비']].sum()
region['국비비율'] = region['국비'] / (region['국비'] + region['사비'])
region = region.sort_values('국비비율', ascending=False)
plt.figure(figsize=(12,5))
sns.barplot(x=region.index, y=region['국비비율'])
plt.title('지역별 국비 인원 비율')
plt.ylabel('국비 인원 비율')
plt.xlabel('지역')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/지역별_국비비율.png')
# plt.show()

# 6. 특정 상병의 지역별 국비/사비 비율 (예: 상병명 Top1)
disease_sel = top_disease.index[0]
df_disease = df[df['상병명']==disease_sel].groupby('지역')[['국비','사비']].sum()
df_disease['국비비율'] = df_disease['국비'] / (df_disease['국비'] + df_disease['사비'])
df_disease = df_disease.sort_values('국비비율', ascending=False)
df_disease[['국비','사비']].plot(kind='bar', stacked=True, figsize=(10,5))
plt.title(f'{disease_sel} 지역별 국비/사비 인원수')
plt.ylabel('인원수')
plt.tight_layout()
plt.savefig(f'imgs/EDA_질병및수술통계/002/{disease_sel}_지역별_국비_사비_누적막대.png')
# plt.show()

# 7. '구분'별 국비/사비 비율 비교
grouped = df.groupby('구분')[['국비','사비']].sum()
grouped['국비비율'] = grouped['국비'] / (grouped['국비'] + grouped['사비'])
grouped[['국비','사비']].plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('구분별 국비/사비 총합')
plt.ylabel('인원수')
plt.tight_layout()
plt.savefig('imgs/EDA_진료정보/004/구분별_국비_사비_총합.png')
# plt.show()

grouped['국비비율'].plot(kind='bar', color='orange', figsize=(8,5))
plt.title('구분별 국비비율')
plt.ylabel('국비비율')
plt.tight_layout()
plt.savefig('imgs/EDA_진료정보/004/구분별_국비비율.png')
# plt.show()

# 8. 연도별 국비/사비 비율 변화 추이
year = df.groupby('년도')[['국비','사비']].sum()
year['국비비율'] = year['국비'] / (year['국비'] + year['사비'])
year[['국비','사비']].plot(kind='bar', stacked=True, figsize=(10,5))
plt.title('연도별 국비/사비 인원수(누적)')
plt.ylabel('인원수')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/연도별_국비_사비_누적막대.png')
# plt.show()

year['국비비율'].plot(marker='o')
plt.title('연도별 국비 인원 비율 추이')
plt.ylabel('국비 인원 비율')
plt.xlabel('년도')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/연도별_국비비율_추이.png')
# plt.show()

# 병원-시도 매핑
hospital_region_map = {
    '서울': ['Seoul', 'Gyeonggi'],
    '인천': ['Incheon'],
    '대구': ['Daegu', 'North Gyeongsang'],
    '대전': ['Daejeon', 'South Chungcheong', 'North Chungcheong', 'Sejong'],
    '광주': ['Gwangju', 'South Jeolla', 'North Jeolla'],
    '부산': ['Busan', 'Ulsan', 'South Gyeongsang']
}

# 병원별 국비비율 계산
hospital = df.groupby('지역')[['국비','사비']].sum()
hospital['국비비율'] = hospital['국비'] / (hospital['국비'] + hospital['사비'])

# 시도별로 값 할당
region_value = {}
region_count = {}
for hosp, regions in hospital_region_map.items():
    if hosp in hospital.index:
        value = hospital.loc[hosp, '국비비율']
        count = hospital.loc[hosp, '국비']
        for r in regions:
            region_value[r] = value
            region_count[r] = count

region_df = pd.DataFrame({
    '시도': list(region_value.keys()),
    '국비비율': list(region_value.values()),
    '국비인원수': [region_count[r] for r in region_value.keys()]
})

# region_df에 국비 인원수도 추가
region_df_count = region_df.copy()
region_df_count = region_df_count.merge(
    hospital['국비'].rename('국비인원수'), left_on='시도', right_index=True, how='left'
)

# geojson 파일을 geopandas로 읽기
gdf = gpd.read_file('kr.json', encoding='utf-8')

# region_df: 시도별 국비비율 DataFrame (folium에서 사용한 것과 동일)
# region_df['시도'], region_df['국비비율']

# geojson의 시도명과 region_df의 시도명이 일치해야 merge가 잘 됩니다.
# 예시: geojson의 'name' 컬럼이 시도명
gdf_merged = gdf.merge(region_df, left_on='name', right_on='시도', how='left')
gdf_merged_count = gdf.merge(region_df, left_on='name', right_on='시도', how='left')

print("geojson 시도명:", sorted(gdf['name'].unique()))
print("region_df 시도명:", sorted(region_df_count['시도'].unique()))

# 색칠 지도 그리기
ax = gdf_merged.plot(
    column='국비비율',
    cmap='YlGnBu',
    linewidth=1,
    edgecolor='black',
    figsize=(8,10),
    legend=True,
    missing_kwds={
        "color": "lightgrey",         # NaN 지역(강원, 제주 등)은 연한 회색
        "edgecolor": "black",
        "hatch": "///",
        "label": "데이터 없음"
    }
)
ax.set_axis_off()
plt.title('병원기준 지역별 국비 인원 비율', fontsize=16)
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/병원기준_지역별_국비비율_지도.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()

# geojson merge


# 국비 인원수로 색칠
ax = gdf_merged_count.plot(
    column='국비인원수',
    cmap='OrRd',
    linewidth=1,
    edgecolor='black',
    figsize=(8,10),
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "black",
        "hatch": "///",
        "label": "데이터 없음"
    }
)
ax.set_axis_off()
plt.title('병원기준 지역별 국비 인원수', fontsize=16)
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/병원기준_지역별_국비인원수_지도.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()

topN = 10
top_gukbi_region = hospital['국비'].sort_values(ascending=False).head(topN)
plt.figure(figsize=(10,5))
sns.barplot(x=top_gukbi_region.values, y=top_gukbi_region.index, palette='OrRd')
plt.title(f'국비 인원수 많은 상위 {topN} 지역')
plt.xlabel('국비 인원수')
plt.ylabel('지역')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/국비인원수_상위지역.png')
# plt.show()

plt.figure(figsize=(8,5))
plt.hist(hospital['국비'].dropna(), bins=10, color='skyblue', edgecolor='black')
plt.title('지역별 국비 인원수 분포')
plt.xlabel('국비 인원수')
plt.ylabel('지역 수')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/국비인원수_분포_히스토그램.png')
# plt.show()



# 9. 지역별 구분별 국비비율 히트맵
df['국비비율'] = df['국비'] / (df['국비'] + df['사비'])
pivot = df.pivot_table(index='지역', columns='구분', values='국비비율', aggfunc='mean')
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('지역별 구분별 국비비율 히트맵')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/지역별_구분별_국비비율_히트맵.png')
# plt.show()

# 10. 연도별 국비/사비 인원수 추이
year_sum = df.groupby('년도')[['국비','사비']].sum().reset_index()
plt.figure(figsize=(10,5))
plt.plot(year_sum['년도'], year_sum['국비'], marker='o', label='국비')
plt.plot(year_sum['년도'], year_sum['사비'], marker='o', label='사비')
plt.title('연도별 국비/사비 인원수 추이')
plt.xlabel('년도')
plt.ylabel('인원수')
plt.legend()
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/연도별_국비_사비_추이.png')
# plt.show()

# 11. 상병별 국비비율 분포
df['국비비율'] = df['국비'] / (df['국비'] + df['사비'])
plt.figure(figsize=(8,5))
plt.hist(df['국비비율'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('상병별 국비비율 분포')
plt.xlabel('국비비율')
plt.ylabel('상병 수')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/상병별_국비비율_분포.png')
# plt.show()

topN = 10
top_gukbi = df.groupby('상병명')['국비'].sum().sort_values(ascending=False).head(topN)
top_sabi = df.groupby('상병명')['사비'].sum().sort_values(ascending=False).head(topN)

plt.figure(figsize=(10,5))
sns.barplot(x=top_gukbi.values, y=top_gukbi.index, palette='Blues_r')
plt.title(f'국비 인원수 상위 {topN} 상병명')
plt.xlabel('국비 인원수')
plt.ylabel('상병명')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/국비_상위상병명.png')
# plt.show()

# 12. 사비 인원수 상위 N개 상병명
plt.figure(figsize=(10,5))
sns.barplot(x=top_sabi.values, y=top_sabi.index, palette='Reds_r')
plt.title(f'사비 인원수 상위 {topN} 상병명')
plt.xlabel('사비 인원수')
plt.ylabel('상병명')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/사비_상위상병명.png')
# plt.show()

'''

yearly_total = df.groupby('년도')['합계'].sum()
yearly_total.plot(marker='o')
plt.title('연도별 전체 진료 건수 추이')
plt.ylabel('진료 건수')
plt.xlabel('년도')
plt.savefig('imgs/EDA_질병및수술통계/002/연도별_전체_진료건수_추이.png')
# plt.show()

region_year = df.groupby(['년도', '지역'])['합계'].sum().unstack()
region_year.plot(figsize=(12,6), marker='o')
plt.title('연도별 지역별 진료 건수 추이')
plt.ylabel('진료 건수')
plt.xlabel('년도')
plt.legend(title='지역', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/연도별_지역별_진료건수_추이.png')
# plt.show()

yearly_fund = df.groupby('년도')[['국비', '사비']].sum()
yearly_fund_ratio = yearly_fund.div(yearly_fund.sum(axis=1), axis=0)
yearly_fund_ratio.plot(kind='bar', stacked=True)
plt.title('연도별 국비/사비 비율 추이')
plt.ylabel('비율')
plt.xlabel('년도')
plt.savefig('imgs/EDA_질병및수술통계/002/연도별_국비_사비_비율_추이.png')
# plt.show()

topN = 5
top_diseases = df.groupby('상병명')['합계'].sum().nlargest(topN).index
top_df = df[df['상병명'].isin(top_diseases)]
pivot = top_df.groupby(['년도', '상병명'])['합계'].sum().unstack()
pivot.plot(marker='o')
plt.title(f'연도별 상병명 Top {topN} 진료 건수 추이')
plt.ylabel('진료 건수')
plt.xlabel('년도')
plt.legend(title='상병명', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'imgs/EDA_질병및수술통계/002/연도별_상병명_Top{topN}_추이.png')
# plt.show()

df['국비비율'] = df['국비'] / (df['국비'] + df['사비'])
pivot = df.groupby(['년도', '지역'])['국비비율'].mean().unstack()
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Blues')
plt.title('연도별 지역별 국비 비율 히트맵')
plt.savefig('imgs/EDA_질병및수술통계/002/연도별_지역별_국비비율_히트맵.png')
# plt.show()

df['상병대분류'] = df['상병코드'].str[0]
pivot = df.groupby(['년도', '상병대분류'])['합계'].sum().unstack()
pivot.plot(marker='o')
plt.title('연도별 상병 대분류별 진료 건수 추이')
plt.ylabel('진료 건수')
plt.xlabel('년도')
plt.legend(title='상병대분류', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/연도별_상병대분류_진료건수_추이.png')
# plt.show()

# 상병코드 앞 세 글자(알파벳+숫자2개)로 그룹화하여 상위 20개 막대그래프
# (예: A00, B23 등)
df['상병코드3'] = df['상병코드'].astype(str).str[:3]
grouped_code3 = df.groupby('상병코드3')['합계'].sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(12,6))
sns.barplot(x=grouped_code3.index, y=grouped_code3.values, palette='viridis')
plt.title('상병코드 앞 세 글자 기준 상위 20개 진료 건수')
plt.xlabel('상병코드(앞3글자)')
plt.ylabel('진료 건수 합계')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/상병코드3_상위20_막대그래프.png')
# plt.show()

# 2023년 데이터만 필터링해서 상병코드 앞 세 글자 기준 상위 20개 막대그래프
# (예: A00, B23 등)
df_2023 = df[df['년도'] == 2023].copy()
df_2023['상병코드3'] = df_2023['상병코드'].astype(str).str[:3]
grouped_code3_2023 = df_2023.groupby('상병코드3')['합계'].sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(12,6))
sns.barplot(x=grouped_code3_2023.index, y=grouped_code3_2023.values, palette='viridis')
plt.title('2023년 상병코드 앞 세 글자 기준 상위 20개 진료 건수')
plt.xlabel('상병코드(앞3글자)')
plt.ylabel('진료 건수 합계')
plt.tight_layout()
plt.savefig('imgs/EDA_질병및수술통계/002/상병코드3_2023_상위20_막대그래프.png')
# plt.show()



