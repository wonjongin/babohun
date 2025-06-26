import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import norm, skew, kurtosis
import geopandas as gpd

os.makedirs('imgs/EDA_시군구별대상자/001', exist_ok=True)

plt.rc('font', family='Pretendard')

# 데이터 읽기
df = pd.read_csv('mpva_original_data/국가보훈부_국가보훈대상자 시군구별 대상별 성별 연령1세별 인원현황_20241231.csv')
print("데이터 형태:", df.shape)
print("컬럼명:", df.columns.tolist())
print("\n처음 5행:")
print(df.head())

# 연령 컬럼들 추출 (0세부터 100세 이상까지)
age_cols = []
for col in df.columns:
    if col.endswith('세') or col == '100세 이상':
        age_cols.append(col)

print(f"\n연령 컬럼 수: {len(age_cols)}")
print("연령 컬럼:", age_cols)

# 연령 컬럼을 숫자형으로 변환
for col in age_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# 연령 분포 데이터 생성 함수 정의
def create_age_distribution_data(df, age_cols, region_name):
    """지역별 연령 분포 데이터 생성"""
    region_df = df[df['시도'] == region_name]
    
    # 연령별 인원수를 개별 연령으로 확장
    age_distribution = []
    for col in age_cols:
        if col == '100세 이상':
            age = 100
        else:
            age = int(col.replace('세', ''))
        
        count = region_df[col].sum()
        # 각 연령에 해당하는 인원수만큼 반복
        age_distribution.extend([age] * count)
    
    return np.array(age_distribution)

# 1. 10세 단위로 그룹화하여 전체 인원 계산
def create_age_groups(df, age_cols):
    """연령을 10세 단위로 그룹화"""
    age_groups = {}
    
    for col in age_cols:
        if col == '100세 이상':
            age = 100
        else:
            age = int(col.replace('세', ''))
        
        # 10세 단위로 그룹화 (0-9세, 10-19세, 20-29세, ...)
        group = f"{age//10*10}대"
        if group not in age_groups:
            age_groups[group] = []
        age_groups[group].append(col)
    
    return age_groups

age_groups = create_age_groups(df, age_cols)
print("\n10세 단위 그룹:")
for group, cols in age_groups.items():
    print(f"{group}: {cols}")

# 10세 단위별 전체 인원 계산
age_group_totals = {}
for group, cols in age_groups.items():
    total = df[cols].sum().sum()
    age_group_totals[group] = total
    print(f"{group} 계산: {cols} -> {total:,}명")

print("\n10세 단위별 전체 인원:")
for group, total in age_group_totals.items():
    print(f"{group}: {total:,}명")

# 디버깅: 각 연령 컬럼별 합계 확인
print("\n각 연령별 전체 합계:")
for col in age_cols:
    total = df[col].sum()
    print(f"{col}: {total:,}명")

# 2. 시도별 전체 인원 계산
province_totals = df.groupby('시도')[age_cols].sum().sum(axis=1)
print("\n시도별 전체 인원:")
for province, total in province_totals.items():
    print(f"{province}: {total:,}명")

# 3. 시도별 10세 단위 인원 계산
province_age_data = {}
for province in df['시도'].unique():
    province_df = df[df['시도'] == province]
    province_age_data[province] = {}
    
    for group, cols in age_groups.items():
        total = province_df[cols].sum().sum()
        province_age_data[province][group] = total

# DataFrame으로 변환
province_age_df = pd.DataFrame(province_age_data).T
print("\n시도별 10세 단위 인원:")
print(province_age_df)

# 4. 시각화
# 4-1. 10세 단위 전체 인원 막대그래프
plt.figure(figsize=(12, 6))
# 내림차순 정렬
sorted_age_totals = sorted(age_group_totals.items(), key=lambda x: x[1], reverse=True)
age_order = [item[0] for item in sorted_age_totals]
age_values = [item[1] for item in sorted_age_totals]

plt.bar(age_order, age_values, color='skyblue', alpha=0.7)
plt.title('10세 단위별 전체 보훈대상자 수', fontsize=14)
plt.xlabel('연령대')
plt.ylabel('인원수')
plt.xticks(rotation=45)
for i, v in enumerate(age_values):
    plt.text(i, v + max(age_values)*0.01, f'{v:,}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/10세단위_전체인원.png', dpi=300, bbox_inches='tight')
plt.close()

# 4-2. 시도별 전체 인원 막대그래프
plt.figure(figsize=(12, 6))
# 내림차순 정렬
sorted_province_totals = sorted(province_totals.items(), key=lambda x: x[1], reverse=True)
provinces = [item[0] for item in sorted_province_totals]
values = [item[1] for item in sorted_province_totals]

plt.bar(provinces, values, color='lightcoral', alpha=0.7)
plt.title('시도별 전체 보훈대상자 수', fontsize=14)
plt.xlabel('시도')
plt.ylabel('인원수')
plt.xticks(rotation=45)
for i, v in enumerate(values):
    plt.text(i, v + max(values)*0.01, f'{v:,}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/시도별_전체인원.png', dpi=300, bbox_inches='tight')
plt.close()

# 4-3. 시도별 10세 단위 히트맵
plt.figure(figsize=(14, 8))
sns.heatmap(province_age_df, annot=True, fmt=',', cmap='YlOrRd', cbar_kws={'label': '인원수'})
plt.title('시도별 10세 단위 보훈대상자 수 히트맵', fontsize=14)
plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/시도별_10세단위_히트맵.png', dpi=300, bbox_inches='tight')
plt.close()

# 권역 매핑
domain_map = {
    '중앙권역': ['서울특별시', '경기도'],
    '부산권역': ['경상남도', '울산광역시', '부산광역시'],
    '광주권역': ['광주광역시', '전라남도', '전북특별자치도'],
    '대전권역': ['대전광역시', '세종특별자치시', '충청남도', '충청북도'],
    '대구권역': ['경상북도', '대구광역시'],
    '인천권역': ['인천광역시']
}

# 권역별 연령대별 인원 집계
domain_age_data = {}
for domain, provinces in domain_map.items():
    domain_age_data[domain] = province_age_df.loc[provinces].sum()

domain_age_df = pd.DataFrame(domain_age_data)
print("\n권역별 10세 단위 인원:")
print(domain_age_df)

# 권역별 10세 단위 인원 막대그래프
plt.figure(figsize=(12, 7))
# 내림차순 정렬 (총 인원수 기준)
domain_totals = domain_age_df.sum()
sorted_domains = domain_totals.sort_values(ascending=False).index
sorted_domain_age_df = domain_age_df[sorted_domains]

ax = sorted_domain_age_df.T[age_order].plot(kind='bar', stacked=True, figsize=(14,7), 
                                     colormap='Pastel1', alpha=1)
plt.title('권역별 10세 단위 보훈대상자 수 (누적 막대)', fontsize=15)
plt.xlabel('권역')
plt.ylabel('인원수')
plt.xticks(rotation=30)

# 총 인원수 레이블 추가
for i, domain in enumerate(sorted_domain_age_df.columns):
    total = sorted_domain_age_df[domain].sum()
    plt.text(i, total + max(sorted_domain_age_df.sum())*0.01, f'{total:,}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/권역별_10세단위_누적막대.png', dpi=300, bbox_inches='tight')
plt.close()

# 권역별 10세 단위 꺾은선그래프
plt.figure(figsize=(12, 7))
for domain in domain_age_df.columns:
    plt.plot(age_order, domain_age_df[domain].reindex(age_order).fillna(0), marker='o', label=domain)
plt.title('권역별 10세 단위 보훈대상자 수 (꺾은선)', fontsize=15)
plt.xlabel('연령대')
plt.ylabel('인원수')
plt.legend()
plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/권역별_10세단위_꺾은선.png', dpi=300, bbox_inches='tight')
plt.close()

# 권역별 10세 단위 히트맵
plt.figure(figsize=(14, 8))
sns.heatmap(domain_age_df, annot=True, fmt=',', cmap='YlOrRd', cbar_kws={'label': '인원수'})
plt.title('권역별 10세 단위 보훈대상자 수 히트맵', fontsize=15)
plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/권역별_10세단위_히트맵.png', dpi=300, bbox_inches='tight')
plt.close()

# 권역에 포함되지 않은 시도 확인
all_provinces = set(province_age_df.index)
used_provinces = set(sum(domain_map.values(), []))
not_used = all_provinces - used_provinces
print("\n권역에 포함되지 않은 시도:")
print(not_used)

# 5. 결과 저장
# 10세 단위 전체 인원
age_group_df = pd.DataFrame(list(age_group_totals.items()), columns=['연령대', '인원수'])
age_group_df.to_csv('new_merged_data/보훈대상자_10세단위_전체인원.csv', index=False, encoding='utf-8-sig')

# 시도별 전체 인원
province_df = pd.DataFrame(list(province_totals.items()), columns=['시도', '인원수'])
province_df.to_csv('new_merged_data/보훈대상자_시도별_전체인원.csv', index=False, encoding='utf-8-sig')

# 시도별 10세 단위 인원
province_age_df.to_csv('new_merged_data/보훈대상자_시도별_10세단위_인원.csv', encoding='utf-8-sig')
# 시도명을 컬럼으로 포함하여 다시 저장
province_age_df.reset_index().rename(columns={'index': '시도'}).to_csv('new_merged_data/보훈대상자_시도별_10세단위_인원_컬럼포함.csv', index=False, encoding='utf-8-sig')

# 권역별 평균연령 저장
domain_mean_age = {}
for domain, provinces in domain_map.items():
    domain_age_data = []
    for province in provinces:
        if province in df['시도'].unique():
            age_data = create_age_distribution_data(df, age_cols, province)
            domain_age_data.extend(age_data)
    
    if len(domain_age_data) > 0:
        domain_mean_age[domain] = np.mean(domain_age_data)

domain_mean_age_df = pd.DataFrame(list(domain_mean_age.items()), columns=['권역', '평균연령'])
domain_mean_age_df.to_csv('new_merged_data/보훈대상자_권역별_평균연령.csv', index=False, encoding='utf-8-sig')

print("\n=== 처리 완료 ===")
print("1. 10세 단위 전체 인원: new_merged_data/보훈대상자_10세단위_전체인원.csv")
print("2. 시도별 전체 인원: new_merged_data/보훈대상자_시도별_전체인원.csv")
print("3. 시도별 10세 단위 인원: new_merged_data/보훈대상자_시도별_10세단위_인원.csv")
print("4. 권역별 평균연령: new_merged_data/보훈대상자_권역별_평균연령.csv")
print("5. 시각화 파일: imgs/EDA_시군구별대상자/001/ 폴더")
print("6. 지역별 연령분포: imgs/EDA_시군구별대상자/001/지역별_연령분포/ 폴더")

# 4-6. 지역별 연령 분포 히스토그램과 정규분포 곡선
# 폴더 생성
os.makedirs('imgs/EDA_시군구별대상자/001/지역별_연령분포', exist_ok=True)

# 시도별 연령 분포 시각화
for province in df['시도'].unique():
    age_data = create_age_distribution_data(df, age_cols, province)
    
    if len(age_data) == 0:
        continue
    
    # 통계량 계산
    mean_age = np.mean(age_data)
    std_age = np.std(age_data)
    skewness = skew(age_data)
    kurt = kurtosis(age_data)
    
    # 그래프 생성
    plt.figure(figsize=(12, 8))
    
    # 히스토그램
    plt.hist(age_data, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 정규분포 곡선
    x = np.linspace(age_data.min(), age_data.max(), 100)
    y = norm.pdf(x, mean_age, std_age)
    plt.plot(x, y, 'r-', linewidth=2, label='정규분포')
    
    # 통계량 텍스트
    stats_text = f'평균: {mean_age:.1f}세\n표준편차: {std_age:.1f}세\n왜도: {skewness:.3f}\n첨도: {kurt:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=12)
    
    plt.title(f'{province} 보훈대상자 연령 분포', fontsize=16)
    plt.xlabel('연령 (세)')
    plt.ylabel('밀도')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 파일명에서 특수문자 제거
    safe_province = province.replace('특별자치도', '').replace('광역시', '').replace('특별시', '')
    plt.savefig(f'imgs/EDA_시군구별대상자/001/지역별_연령분포/{safe_province}_연령분포.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# 권역별 연령 분포 시각화
# 하나의 파일에 모든 권역을 서브플롯으로 표시
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, (domain, provinces) in enumerate(domain_map.items()):
    if idx >= 6:  # 최대 6개 권역
        break
        
    # 권역에 속한 시도들의 데이터 합치기
    domain_age_data = []
    for province in provinces:
        if province in df['시도'].unique():
            age_data = create_age_distribution_data(df, age_cols, province)
            domain_age_data.extend(age_data)
    
    domain_age_data = np.array(domain_age_data)
    
    if len(domain_age_data) == 0:
        continue
    
    # 통계량 계산
    mean_age = np.mean(domain_age_data)
    std_age = np.std(domain_age_data)
    skewness = skew(domain_age_data)
    kurt = kurtosis(domain_age_data)
    
    # 서브플롯에 히스토그램
    axes[idx].hist(domain_age_data, bins=20, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    
    # 정규분포 곡선
    x = np.linspace(domain_age_data.min(), domain_age_data.max(), 100)
    y = norm.pdf(x, mean_age, std_age)
    axes[idx].plot(x, y, 'r-', linewidth=2, label='정규분포')
    
    # 통계량 텍스트
    stats_text = f'평균: {mean_age:.1f}세\n표준편차: {std_age:.1f}세\n왜도: {skewness:.3f}\n첨도: {kurt:.3f}'
    axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=10)
    
    axes[idx].set_title(f'{domain} 보훈대상자 연령 분포', fontsize=14)
    axes[idx].set_xlabel('연령 (세)')
    axes[idx].set_ylabel('밀도')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

# 빈 서브플롯 숨기기
for idx in range(len(domain_map), 6):
    axes[idx].set_visible(False)

plt.suptitle('권역별 보훈대상자 연령 분포', fontsize=18)
plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/지역별_연령분포/권역별_연령분포_통합.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 4-7. 권역별 평균연령 지도 시각화
# 권역별 평균연령 계산
domain_mean_age = {}
for domain, provinces in domain_map.items():
    domain_age_data = []
    for province in provinces:
        if province in df['시도'].unique():
            age_data = create_age_distribution_data(df, age_cols, province)
            domain_age_data.extend(age_data)
    
    if len(domain_age_data) > 0:
        domain_mean_age[domain] = np.mean(domain_age_data)

print("\n권역별 평균연령:")
for domain, mean_age in domain_mean_age.items():
    print(f"{domain}: {mean_age:.1f}세")

# 권역별 시도 매핑 (영문) - 002 파일 참고
domain_region_map = {
    '중앙권역': ['Seoul', 'Gyeonggi'],
    '부산권역': ['South Gyeongsang', 'Ulsan', 'Busan'],
    '광주권역': ['Gwangju', 'South Jeolla', 'North Jeolla'],
    '대전권역': ['Daejeon', 'Sejong', 'South Chungcheong', 'North Chungcheong'],
    '대구권역': ['North Gyeongsang', 'Daegu'],
    '인천권역': ['Incheon']
}

# 권역별 평균연령을 시도별로 매핑
region_mean_age = {}
for domain, regions in domain_region_map.items():
    if domain in domain_mean_age:
        mean_age = domain_mean_age[domain]
        for region in regions:
            region_mean_age[region] = mean_age

# DataFrame 변환
region_mean_age_df = pd.DataFrame(list(region_mean_age.items()), columns=['시도영문', '평균연령'])

# geojson 파일 읽기
gdf = gpd.read_file('kr.json', encoding='utf-8')

# merge
gdf_merged = gdf.merge(region_mean_age_df, left_on='name', right_on='시도영문', how='left')

# 지도 시각화
ax = gdf_merged.plot(
    column='평균연령',
    cmap='RdYlBu_r',  # 빨강(높은 연령) - 파랑(낮은 연령)
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
plt.title('권역별 보훈대상자 평균연령', fontsize=16)
plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/권역별_평균연령_지도.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()

# 권역별 평균연령 막대그래프
plt.figure(figsize=(10, 6))
# 내림차순 정렬
sorted_domain_mean_age = sorted(domain_mean_age.items(), key=lambda x: x[1], reverse=True)
domains = [item[0] for item in sorted_domain_mean_age]
mean_ages = [item[1] for item in sorted_domain_mean_age]

plt.bar(domains, mean_ages, color='lightgreen', alpha=0.7)
plt.title('권역별 보훈대상자 평균연령', fontsize=14)
plt.xlabel('권역')
plt.ylabel('평균연령 (세)')
plt.xticks(rotation=30)
for i, v in enumerate(mean_ages):
    plt.text(i, v + max(mean_ages)*0.01, f'{v:.1f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('imgs/EDA_시군구별대상자/001/권역별_평균연령_막대그래프.png', dpi=300, bbox_inches='tight')
plt.close()
