import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew
from scipy.stats import kurtosis

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

# 4-1. 연령대별 진료과 예약자 수 변화 (꺾은선 그래프)
# 전체에서 상위 10개 진료과 선택
top_depts_overall = df_long.groupby('진료과')['예약건수'].sum().sort_values(ascending=False).head(10).index

plt.figure(figsize=(14, 8))

# 연령대 순서
age_order = ["20대","30대","40대","50대","60대","70대","80대","90대"]

# 각 진료과별로 연령대 변화 그리기
for dept in top_depts_overall:
    dept_data = df_long[df_long['진료과'] == dept].groupby('연령대')['예약건수'].sum().reindex(age_order).fillna(0)
    plt.plot(range(len(age_order)), dept_data.values, marker='o', linewidth=2, markersize=6, label=dept)

plt.title('연령대별 진료과 예약자 수 변화 (상위 10개 진료과)', fontsize=14)
plt.xlabel('연령대')
plt.ylabel('예약건수')
plt.xticks(range(len(age_order)), age_order)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/연령대별_진료과_예약자수_변화_꺾은선.png', dpi=300, bbox_inches='tight')
plt.close()

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

# 11-1. 전체 연령대 예약 비율 히스토그램 (연령 순서대로)
age_order = ["20대","30대","40대","50대","60대","70대","80대","90대"]
age_ratio = df_long.groupby('연령대')['예약건수'].sum().reindex(age_order)

plt.figure(figsize=(10,6))

# 히스토그램 그리기
bars = plt.bar(range(len(age_order)), age_ratio.values, alpha=0.7, color='skyblue', edgecolor='navy')

# 정규분포 형태의 부드러운 곡선을 위한 가우시안 스무딩
x_smooth = np.linspace(0, len(age_order)-1, 100)
# 가우시안 필터로 부드럽게 만들기 (sigma 값으로 부드러움 조절)
y_smooth = gaussian_filter1d(age_ratio.values, sigma=1.0)
# 보간으로 더 부드럽게
f_smooth = interp1d(range(len(age_order)), y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
y_smooth_interp = f_smooth(x_smooth)

# 부드러운 곡선 그리기
plt.plot(x_smooth, y_smooth_interp, 'r-', linewidth=3, label='부드러운 곡선')
# 원본 데이터 포인트 표시
plt.plot(range(len(age_order)), age_ratio.values, 'ro', markersize=8, label='실제 데이터')

# 연령대에 대한 평균, 표준편차, 왜도 계산 (가중 통계)
age_nums = [25, 35, 45, 55, 65, 75, 85, 95]  # 각 연령대의 대표값
weights = age_ratio.values
mean_age = np.average(age_nums, weights=weights)
# 가중 분산 계산
weighted_variance = np.average((np.array(age_nums) - mean_age)**2, weights=weights)
std_age = np.sqrt(weighted_variance)

# 왜도 계산 방법 1: 직접 계산
weighted_skewness = np.average(((np.array(age_nums) - mean_age) / std_age)**3, weights=weights)

# 왜도 계산 방법 2: scipy 사용 (확인용)
# 각 연령대를 해당 가중치만큼 반복하여 배열 생성
age_array = []
for age, weight in zip(age_nums, weights):
    age_array.extend([age] * int(weight))
skew_scipy = skew(age_array)

# 첨도 계산 (분포의 뾰족함)
weighted_kurtosis = np.average(((np.array(age_nums) - mean_age) / std_age)**4, weights=weights)
kurt_scipy = kurtosis(age_array)

print(f"연령대별 예약건수: {dict(zip(age_order, weights))}")
print(f"연령 평균: {mean_age:.1f}세")
print(f"연령 표준편차: {std_age:.1f}세")
print(f"직접 계산한 왜도: {weighted_skewness:.3f}")
print(f"scipy 계산한 왜도: {skew_scipy:.3f}")
print(f"직접 계산한 첨도: {weighted_kurtosis:.3f}")
print(f"scipy 계산한 첨도: {kurt_scipy:.3f}")
print(f"초과첨도 (정규분포 대비): {kurt_scipy:.3f}")

# 분포 해석
if weighted_skewness > 0.5:
    skew_interpretation = "오른쪽으로 치우친 분포 (고연령층 우세)"
elif weighted_skewness < -0.5:
    skew_interpretation = "왼쪽으로 치우친 분포 (저연령층 우세)"
else:
    skew_interpretation = "대칭적인 분포"

# 첨도 해석
if kurt_scipy > 1:
    kurt_interpretation = "매우 뾰족한 분포 (특정 연령대 집중)"
elif kurt_scipy > 0:
    kurt_interpretation = "뾰족한 분포 (정규분포보다 집중)"
elif kurt_scipy > -1:
    kurt_interpretation = "평평한 분포 (정규분포보다 분산)"
else:
    kurt_interpretation = "매우 평평한 분포 (연령대별 고르게 분산)"

print(f"분포 해석: {skew_interpretation}")
print(f"첨도 해석: {kurt_interpretation}")

plt.axhline(y=age_ratio.mean(), color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'예약건수 평균: {age_ratio.mean():.1f}')
plt.fill_between(range(len(age_order)), age_ratio.mean()-age_ratio.std(), age_ratio.mean()+age_ratio.std(), alpha=0.2, color='red', label=f'예약건수 표준편차: ±{age_ratio.std():.1f}')

plt.title(f'전체 연령대 예약 비율 (히스토그램, 연령 순)\n(연령 평균: {mean_age:.1f}세, 연령 표준편차: {std_age:.1f}세, 연령 왜도: {weighted_skewness:.3f})', fontsize=12)
plt.xlabel('연령대')
plt.ylabel('예약건수')
plt.xticks(range(len(age_order)), age_order)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imgs/EDA_연령대별 진료과 예약건수/001/전체연령대_예약비율_히스토그램_연령순.png', dpi=300, bbox_inches='tight')
plt.close()

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

age_sel = '80대'
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

