import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd

os.makedirs('imgs/EDA_연령대별 진료과 예약건수/001', exist_ok=True)

plt.rc('font', family='Pretendard')

df = pd.read_csv('final_merged_data/연령대별 진료과 예약건수_2023.csv')
print(df.head())
print(df.columns)

# 1. long format으로 변환 (melt)
age_cols = ["20대","30대","40대","50대","60대","70대","80대","90대"]
df_long = df.melt(id_vars=['병원명','진료과'], value_vars=age_cols, 
                  var_name='연령대', value_name='예약건수')
print(df_long.head())

# 2. 기술통계
print(df_long.groupby('진료과')['예약건수'].agg(['mean', 'sum', 'std']))
print(df_long.groupby('연령대')['예약건수'].agg(['mean', 'sum', 'std']))
print(df_long.groupby('병원명')['예약건수'].agg(['mean', 'sum', 'std']))

# 3. 전체 연령대 인기 진료과 Top N
topN = 10
top_dept = df_long.groupby('진료과')['예약건수'].sum().sort_values(ascending=False).head(topN)
plt.figure(figsize=(10,5))
sns.barplot(x=top_dept.values, y=top_dept.index, palette='viridis')
plt.title(f'전체 연령대 인기 진료과 Top {topN}')
plt.xlabel('예약건수')
plt.ylabel('진료과')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/전체연령_인기진료과_Top10.png')
# plt.show()

# 4. 연령대별 인기 진료과 Top N
for age in age_cols:
    top_dept_age = df_long[df_long['연령대']==age].groupby('진료과')['예약건수'].sum().sort_values(ascending=False).head(topN)
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_dept_age.values, y=top_dept_age.index, palette='magma')
    plt.title(f'{age} 인기 진료과 Top {topN}')
    plt.xlabel('예약건수')
    plt.ylabel('진료과')
    plt.tight_layout()
    plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/'+age+'_인기진료과_Top10.png')
# plt.show()

# 5. 진료과별 예약건수 비율 파이차트 (전체)
dept_ratio = df_long.groupby('진료과')['예약건수'].sum().sort_values(ascending=False)
plt.figure(figsize=(8,8))
plt.pie(dept_ratio, labels=dept_ratio.index, autopct='%1.1f%%', startangle=140)
plt.title('진료과별 예약건수 비율')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/진료과별_예약건수_비율.png')
# plt.show()

# 6. 특정 연령대 파이차트 (예: 30대)
age_sel = '30대'
dept_ratio_age = df_long[df_long['연령대']==age_sel].groupby('진료과')['예약건수'].sum().sort_values(ascending=False)
plt.figure(figsize=(8,8))
plt.pie(dept_ratio_age, labels=dept_ratio_age.index, autopct='%1.1f%%', startangle=140)
plt.title(f'{age_sel} 진료과별 예약건수 비율')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/'+age_sel+'_진료과별_예약건수_비율.png')
# plt.show()

# 7. 진료과별 연령대 분포 (누적 막대그래프)
pivot = df_long.pivot_table(index='진료과', columns='연령대', values='예약건수', aggfunc='sum', fill_value=0)
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]  # 예약건수 많은 순 정렬
pivot.plot(kind='bar', stacked=True, figsize=(14,6), colormap='tab20')
plt.title('진료과별 연령대별 예약건수(누적)')
plt.ylabel('예약건수')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/진료과별_연령대별_예약건수_누적.png')
# plt.show()

# 8. 100% 기준 누적 막대그래프
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)
pivot_pct.plot(kind='bar', stacked=True, figsize=(14,6), colormap='tab20')
plt.title('진료과별 연령대별 예약건수 비율(100% 기준)')
plt.ylabel('비율')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/진료과별_연령대별_예약건수_비율.png')
# plt.show()

# 9. 병원별, 연령대별 인기 진료과 히트맵
heatmap_data = df_long.pivot_table(index='병원명', columns='연령대', values='예약건수', aggfunc='sum', fill_value=0)
plt.figure(figsize=(14,8))
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('병원별 연령대별 예약건수 히트맵')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/병원별_연령대별_예약건수_히트맵.png')
# plt.show()

# 10. 병원별 인기 진료과 Top3
for hosp in df['병원명'].unique():
    top_dept_hosp = df_long[df_long['병원명']==hosp].groupby('진료과')['예약건수'].sum().sort_values(ascending=False).head(3)
    print(f"{hosp} 인기 진료과 Top3:\n", top_dept_hosp)

# 11. 전체 연령대 예약 비율 파이차트
age_ratio = df_long.groupby('연령대')['예약건수'].sum().sort_values(ascending=False)
plt.figure(figsize=(8,8))
plt.pie(age_ratio, labels=age_ratio.index, autopct='%1.1f%%', startangle=140)
plt.title('전체 연령대 예약 비율')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/전체연령대_예약비율_파이.png')
# plt.show()

# 11-1. 전체 연령대 예약 비율 막대그래프 (연령 순서대로)
age_order = ["20대","30대","40대","50대","60대","70대","80대","90대"]
age_ratio = df_long.groupby('연령대')['예약건수'].sum().reindex(age_order)
plt.figure(figsize=(8,5))
sns.barplot(x=age_ratio.index, y=age_ratio.values, palette='viridis')
plt.title('전체 연령대 예약 비율 (막대그래프, 연령 순)')
plt.xlabel('연령대')
plt.ylabel('예약건수')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/전체연령대_예약비율_막대_연령순.png')
# plt.show()

# 12. 병원별 연령대 예약 비율 (100% 누적 막대그래프)
pivot_hosp = df_long.pivot_table(index='병원명', columns='연령대', values='예약건수', aggfunc='sum', fill_value=0)
pivot_hosp_pct = pivot_hosp.div(pivot_hosp.sum(axis=1), axis=0)
pivot_hosp_pct.plot(kind='bar', stacked=True, figsize=(14,7), colormap='tab20')
plt.title('병원별 연령대 예약 비율 (100% 기준)')
plt.ylabel('비율')
plt.xlabel('병원명')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/병원별_연령대_예약비율_100.png')
# plt.show()

# 병원명 → 권역(시도 리스트) 매핑
hospital_region_map = {
    '중앙보훈': ['Seoul', 'Gyeonggi'],
    '인천보훈': ['Incheon'],
    '대구보훈': ['Daegu', 'North Gyeongsang'],
    '대전보훈': ['Daejeon', 'South Chungcheong', 'North Chungcheong', 'Sejong'],
    '광주보훈': ['Gwangju', 'South Jeolla', 'North Jeolla'],
    '부산보훈': ['Busan', 'Ulsan', 'South Gyeongsang']
}

# 시도별 예약건수 누적용 딕셔너리
region_resv = {}

for idx, row in df_long.iterrows():
    hosp = row['병원명']
    val = row['예약건수']
    if pd.isnull(val): continue
    val = float(val)
    for region in hospital_region_map.get(hosp, []):
        region_resv[region] = region_resv.get(region, 0) + val

# DataFrame 변환
region_sum = pd.DataFrame(list(region_resv.items()), columns=['시도영문', '예약건수'])

# geojson 파일 읽기
gdf = gpd.read_file('kr.json', encoding='utf-8')

# merge
gdf_merged = gdf.merge(region_sum, left_on='name', right_on='시도영문', how='left')

ax = gdf_merged.plot(
    column='예약건수',
    cmap='YlGnBu',
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
plt.title('권역(시도)별 전체 예약건수', fontsize=16)
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/권역별_전체예약건수_지도.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()

age_sel = '70대'
region_resv_age = {}
for idx, row in df_long[df_long['연령대']==age_sel].iterrows():
    hosp = row['병원명']
    val = row['예약건수']
    if pd.isnull(val): continue
    val = float(val)
    for region in hospital_region_map.get(hosp, []):
        region_resv_age[region] = region_resv_age.get(region, 0) + val
region_sum_age = pd.DataFrame(list(region_resv_age.items()), columns=['시도영문', '예약건수'])
gdf_merged_age = gdf.merge(region_sum_age, left_on='name', right_on='시도영문', how='left')

ax = gdf_merged_age.plot(
    column='예약건수',
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
plt.title(f'권역(시도)별 {age_sel} 예약건수', fontsize=16)
plt.tight_layout()
plt.savefig(f'imgs/EDA_연령대별 진료과 예약건수/001/권역별_{age_sel}_예약건수_지도.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()

topN = 6
region_sum = df_long.groupby('병원명')['예약건수'].sum().sort_values(ascending=False).head(topN)
plt.figure(figsize=(8,5))
sns.barplot(x=region_sum.values, y=region_sum.index, palette='YlGnBu')
plt.title(f'지역별 전체 예약건수 Top {topN}')
plt.xlabel('예약건수')
plt.ylabel('지역')
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/지역별_전체예약건수_막대.png')
plt.close()

