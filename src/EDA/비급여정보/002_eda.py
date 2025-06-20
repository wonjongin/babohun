import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rc('font', family='Pretendard')
os.makedirs('imgs/EDA_비급여정보/002', exist_ok=True)

# 데이터 불러오기
df = pd.read_csv('final_merged_data/비급여정보.csv')

# 1. 데이터 클렌징 (비용 결측치, 이상치, 타입 변환 등)
df['비용'] = pd.to_numeric(df['비용'], errors='coerce')
df = df.dropna(subset=['비용'])  # 비용 없는 행 제거

# 2. 기술통계 (전체, 대분류/중분류/소분류별)
stat_cols = ['비용']
group_cols = ['대분류', '중분류', '소분류']

# 전체
print("전체 기술통계:\n", df['비용'].describe())

# 대분류별
print("대분류별 기술통계:\n", df.groupby('대분류')['비용'].describe())

# 중분류별
print("중분류별 기술통계:\n", df.groupby('중분류')['비용'].describe())

# 소분류별
print("소분류별 기술통계:\n", df.groupby('소분류')['비용'].describe())

# 3. 전체 비용 분포 (히스토그램, Boxplot)
plt.figure(figsize=(10,5))
sns.histplot(df['비용'], bins=50, kde=True)
plt.title('전체 비급여 항목 비용 분포 (히스토그램)')
plt.xlabel('비용(원)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/전체_비용_히스토그램.png')
# plt.show()

q99 = df['비용'].quantile(0.99)
filtered = df[df['비용'] <= q99]
plt.figure(figsize=(8,5))
sns.boxplot(x=filtered['비용'])
plt.title('전체 비급여 항목 비용 분포 (Boxplot, 상위 1% 제외)')
plt.xlabel('비용(원)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/전체_비용_Boxplot_99_filtered.png')
# plt.show()

# 4. 비용 상위 N개 항목 리스트업 및 막대그래프
topN = 10
top_items = df.sort_values('비용', ascending=False).head(topN)
plt.figure(figsize=(12,6))
sns.barplot(x=top_items['비용'], y=top_items['명칭'] + ' (' + top_items['대분류'].fillna('') + '/' + top_items['중분류'].fillna('') + ')', palette='Reds_r')
plt.title(f'비용 상위 {topN} 비급여 항목')
plt.xlabel('비용(원)')
plt.ylabel('항목명 (대분류/중분류)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/비용상위항목_Top10.png')
# plt.show()

# 5. 대분류/중분류별 비급여 항목 수 및 총 비용 (막대그래프)
# (1) 항목 수
cnt = df['대분류'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(x=cnt.index, y=cnt.values, palette='Blues')
plt.title('대분류별 비급여 항목 수')
plt.xlabel('대분류')
plt.ylabel('항목 수')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/대분류별_항목수.png')
# plt.show()

# (2) 총 비용
sum_cost = df.groupby('대분류')['비용'].sum().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=sum_cost.index, y=sum_cost.values, palette='Greens')
plt.title('대분류별 비급여 항목 총 비용')
plt.xlabel('대분류')
plt.ylabel('총 비용(원)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/대분류별_총비용.png')
# plt.show()

# 6. 치료재료대/약제비 포함여부별 비용 분포 (Boxplot)
plt.figure(figsize=(10,5))
sns.boxplot(x=filtered['치료재료대포함여부'].fillna('미기재'), y=filtered['비용'])
plt.title('치료재료대 포함여부별 비용 분포 (상위 1% 제외)')
plt.xlabel('치료재료대 포함여부')
plt.ylabel('비용(원)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/치료재료대포함여부별_Boxplot_99.png')
# plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x=df['약제비포함여부'].fillna('미기재'), y=df['비용'])
plt.title('약제비 포함여부별 비용 분포')
plt.xlabel('약제비 포함여부')
plt.ylabel('비용(원)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/약제비포함여부별_Boxplot.png')
# plt.show()

# 약제비포함여부별 Boxplot (상위 5% 이상치 제외)
q95 = df['비용'].quantile(0.95)
filtered_95 = df[df['비용'] <= q95]

plt.figure(figsize=(10,5))
sns.boxplot(x=filtered_95['약제비포함여부'].fillna('미기재'), y=filtered_95['비용'])
plt.title('약제비 포함여부별 비용 분포 (상위 5% 제외)')
plt.xlabel('약제비 포함여부')
plt.ylabel('비용(원)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/약제비포함여부별_Boxplot_95.png')
# plt.show()

# 7. 지역별 평균 비용 (막대그래프)
region_mean = df.groupby('지역')['비용'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=region_mean.index, y=region_mean.values, palette='coolwarm')
plt.title('지역별 비급여 항목 평균 비용')
plt.xlabel('지역')
plt.ylabel('평균 비용(원)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/지역별_평균비용.png')
# plt.show()

# 8. (선택) 연도별 특정 항목 비용 변화 추이 (예: 상위 1개 항목)
item_sel = top_items.iloc[0]['명칭']
item_trend = df[df['명칭'] == item_sel].groupby('년도')['비용'].mean()
plt.figure(figsize=(8,5))
item_trend.plot(marker='o')
plt.title(f'{item_sel} 연도별 평균 비용 추이')
plt.xlabel('년도')
plt.ylabel('평균 비용(원)')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/상위항목_연도별_비용추이.png')
# plt.show()

# 9. (선택) 대분류/중분류별 파이차트
pie = df['대분류'].value_counts()
plt.figure(figsize=(7,7))
plt.pie(pie, labels=pie.index, autopct='%1.1f%%', startangle=140)
plt.title('대분류별 비급여 항목 비율')
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/대분류별_항목비율_파이.png')
# plt.show()

# 10. (선택) "필수적" 항목(예: 비용 상위, 빈도 상위) 리스트업
# 실제로는 의학적 판단 필요, 여기선 비용 상위+빈도 상위 교집합 예시
freq_items = df['명칭'].value_counts().head(20).index
essential_items = df[df['명칭'].isin(freq_items)].sort_values('비용', ascending=False).head(10)
print("필수적(가정) + 비용상위 항목 예시:\n", essential_items[['명칭','비용','대분류','중분류','지역']])

# 1. 권역(시도 리스트) 매핑
region_zone_map = {
    '서울': ['Seoul', 'Gyeonggi'],
    '인천': ['Incheon'],
    '대구': ['Daegu', 'North Gyeongsang'],
    '대전': ['Daejeon', 'South Chungcheong', 'North Chungcheong', 'Sejong'],
    '광주': ['Gwangju', 'South Jeolla', 'North Jeolla'],
    '부산': ['Busan', 'Ulsan', 'South Gyeongsang']
}

# 2. 원본 지역별 평균 비용
region_mean = df.groupby('지역')['비용'].mean()

# 3. 권역별로 값 복사
region_cost = []
for region_kor, mean_cost in region_mean.items():
    for region_eng in region_zone_map.get(region_kor, []):
        region_cost.append({'시도영문': region_eng, '평균비용': mean_cost})

region_sum = pd.DataFrame(region_cost)

# 4. geojson merge
gdf = gpd.read_file('kr.json', encoding='utf-8')
gdf_merged = gdf.merge(region_sum, left_on='name', right_on='시도영문', how='left')

# 5. 지도 시각화
ax = gdf_merged.plot(
    column='평균비용',
    cmap='YlOrRd',
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
plt.title('권역(시도)별 비급여 항목 평균 비용', fontsize=16)
plt.tight_layout()
plt.savefig('imgs/EDA_비급여정보/002/권역별_평균비용_지도.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()

# 11. 지역별 연도별 비급여 총액(합계) 추이 꺾은선 그래프
if '년도' in df.columns and '지역' in df.columns:
    region_year_sum = df.groupby(['년도','지역'])['비용'].sum().reset_index()
    plt.figure(figsize=(12,6))
    for region in region_year_sum['지역'].unique():
        data = region_year_sum[region_year_sum['지역']==region]
        plt.plot(data['년도'], data['비용'], marker='o', label=region)
    plt.title('지역별 연도별 비급여 총액 추이')
    plt.xlabel('년도')
    plt.ylabel('총 비용(원)')
    plt.legend(title='지역')
    plt.tight_layout()
    plt.savefig('imgs/EDA_비급여정보/002/지역별_연도별_총액_추이.png')
    # plt.show()
