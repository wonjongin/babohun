# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

os.makedirs('imgs/EDA_병원시설/002', exist_ok=True)

plt.rc('font', family='Pretendard')

# 1. 데이터 읽기
hospital_df = pd.read_csv('new_merged_data/병원_통합_데이터.csv')
veteran_df = pd.read_csv('new_merged_data/보훈대상자_시도별_10세단위_인원_컬럼포함.csv')

print("병원 데이터 형태:", hospital_df.shape)
print("보훈대상자 데이터 형태:", veteran_df.shape)

# 2. 권역 매핑
domain_map = {
    '중앙권역': ['서울특별시', '경기도'],
    '부산권역': ['경상남도', '울산광역시', '부산광역시'],
    '광주권역': ['광주광역시', '전라남도', '전북특별자치도'],
    '대전권역': ['대전광역시', '세종특별자치시', '충청남도', '충청북도'],
    '대구권역': ['경상북도', '대구광역시'],
    '인천권역': ['인천광역시']
}

# 3. 보훈대상자 데이터를 권역별로 합치기
veteran_df['권역'] = veteran_df['시도'].map(lambda x: next((domain for domain, provinces in domain_map.items() if x in provinces), '기타'))

# 권역별 보훈대상자 수 합계 (숫자 컬럼만 선택)
age_cols = [col for col in veteran_df.columns if col not in ['시도', '권역']]
veteran_domain = veteran_df.groupby('권역')[age_cols].sum().drop('기타', errors='ignore')
print("\n권역별 보훈대상자 수:")
print(veteran_domain.sum(axis=1))

# 4. 병원 데이터 전처리
# 호스피스 제외하고 일반 병원만 사용
hospital_main = hospital_df[~hospital_df['병원명'].str.contains('호스피스', na=False)].copy()

# 병원명을 권역으로 매핑
hospital_main['권역'] = hospital_main['병원명'].map({
    '서울': '중앙권역',
    '부산': '부산권역', 
    '대구': '대구권역',
    '인천': '인천권역',
    '광주': '광주권역',
    '대전': '대전권역'
})

# 권역별 병원 시설 합계
hospital_domain = hospital_main.groupby('권역').sum()

print("\n권역별 병원 시설:")
print(hospital_domain)

# 5. 보훈대상자 대비 의료자원 비율 계산
# 보훈대상자 수
veteran_totals = veteran_domain.sum(axis=1)

# _전문의수로 끝나는 모든 컬럼 찾기
specialist_cols = [col for col in hospital_domain.columns if col.endswith('_전문의수')]
print(f"\n전문의 관련 컬럼 수: {len(specialist_cols)}")
print("전문의 컬럼들:", specialist_cols)

# 주요 의료자원 컬럼들 (전문의 + 기타 의료자원, 일반 의사 제외)
medical_resources = {
    '일반입원실_일반': '일반병상',
    '일반입원실_상급': '상급병상', 
    '중환자실_성인': '성인중환자실',
    '중환자실_소아': '소아중환자실',
    '응급실': '응급실'
}

# 전문의 컬럼들 추가
for col in specialist_cols:
    resource_name = col.replace('_전문의수', '').replace('_', '')
    medical_resources[col] = resource_name

# 권역별 의료자원 대비 보훈대상자 비율 계산
resource_ratios = {}
for resource_col, resource_name in medical_resources.items():
    if resource_col in hospital_domain.columns:
        ratios = {}
        for domain in hospital_domain.index:
            if domain in veteran_totals.index:
                resource_count = hospital_domain.loc[domain, resource_col]
                veteran_count = veteran_totals[domain]
                if resource_count > 0 and veteran_count > 0:
                    ratio = veteran_count / resource_count
                    ratios[domain] = ratio
        resource_ratios[resource_name] = ratios

# 6. 시각화
# 6-1. 권역별 보훈대상자 수
plt.figure(figsize=(12, 6))
domains = list(veteran_totals.index)
veteran_counts = list(veteran_totals.values)

plt.bar(domains, veteran_counts, color='lightblue', alpha=0.7)
plt.title('권역별 보훈대상자 수', fontsize=14)
plt.xlabel('권역')
plt.ylabel('보훈대상자 수')
plt.xticks(rotation=30)
for i, v in enumerate(veteran_counts):
    plt.text(i, v + max(veteran_counts)*0.01, f'{v:,}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('imgs/EDA_병원시설/002/권역별_보훈대상자수.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-2. 권역별 주요 의료자원 현황
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

resource_cols = ['일반입원실_일반', '중환자실_성인', '내과_전문의수', '외과_전문의수']
resource_names = ['일반병상', '성인중환자실', '내과전문의', '외과전문의']

for idx, (col, name) in enumerate(zip(resource_cols, resource_names)):
    if col in hospital_domain.columns:
        values = hospital_domain[col].values
        domains = hospital_domain.index
        
        axes[idx].bar(domains, values, color='lightcoral', alpha=0.7)
        axes[idx].set_title(f'권역별 {name} 수', fontsize=12)
        axes[idx].set_ylabel(name)
        axes[idx].tick_params(axis='x', rotation=30)
        
        for i, v in enumerate(values):
            if v > 0:
                axes[idx].text(i, v + max(values)*0.01, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('imgs/EDA_병원시설/002/권역별_주요의료자원.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-3. 보훈대상자 대비 의료자원 비율 히트맵
ratio_df = pd.DataFrame(resource_ratios)
ratio_df = ratio_df.fillna(0)

plt.figure(figsize=(14, 10))
sns.heatmap(ratio_df.T, annot=True, fmt='.1f', cmap='RdYlBu_r', 
            cbar_kws={'label': '보훈대상자 수 / 의료자원 수'})
plt.title('권역별 보훈대상자 대비 의료자원 비율', fontsize=14)
plt.tight_layout()
plt.savefig('imgs/EDA_병원시설/002/보훈대상자_대비_의료자원_비율_히트맵.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-4. 권역별 전문의/보훈대상자 비율 비교
plt.figure(figsize=(12, 6))
specialist_ratios = []
domains = []

for domain in hospital_domain.index:
    if domain in veteran_totals.index:
        veterans = veteran_totals[domain]
        specialists = hospital_domain.loc[domain, specialist_cols].sum()
        if specialists > 0:
            ratio = specialists / veterans * 10000  # 1만명당 전문의 수
            specialist_ratios.append(ratio)
            domains.append(domain)

# 내림차순 정렬
sorted_data = sorted(zip(domains, specialist_ratios), key=lambda x: x[1], reverse=True)
sorted_domains = [item[0] for item in sorted_data]
sorted_ratios = [item[1] for item in sorted_data]

plt.bar(sorted_domains, sorted_ratios, color='lightgreen', alpha=0.7)
plt.title('권역별 1만명당 전문의 수', fontsize=14)
plt.xlabel('권역')
plt.ylabel('1만명당 전문의 수')
plt.xticks(rotation=30)
for i, v in enumerate(sorted_ratios):
    plt.text(i, v + max(sorted_ratios)*0.01, f'{v:.1f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('imgs/EDA_병원시설/002/권역별_1만명당_전문의수.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-5. 권역별 병상/보훈대상자 비율 비교
plt.figure(figsize=(12, 6))
bed_ratios = []
domains = []

for domain in hospital_domain.index:
    if domain in veteran_totals.index:
        veterans = veteran_totals[domain]
        beds = hospital_domain.loc[domain, '일반입원실_일반']
        if beds > 0:
            ratio = beds / veterans * 10000  # 1만명당 병상 수
            bed_ratios.append(ratio)
            domains.append(domain)

# 내림차순 정렬
sorted_data = sorted(zip(domains, bed_ratios), key=lambda x: x[1], reverse=True)
sorted_domains = [item[0] for item in sorted_data]
sorted_ratios = [item[1] for item in sorted_data]

plt.bar(sorted_domains, sorted_ratios, color='lightblue', alpha=0.7)
plt.title('권역별 1만명당 병상 수', fontsize=14)
plt.xlabel('권역')
plt.ylabel('1만명당 병상 수')
plt.xticks(rotation=30)
for i, v in enumerate(sorted_ratios):
    plt.text(i, v + max(sorted_ratios)*0.01, f'{v:.1f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('imgs/EDA_병원시설/002/권역별_1만명당_병상수.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-6. 보훈대상자 대비 병상 수 산점도
plt.figure(figsize=(10, 8))
x_values = []
y_values = []
labels = []

for domain in hospital_domain.index:
    if domain in veteran_totals.index:
        beds = hospital_domain.loc[domain, '일반입원실_일반']
        veterans = veteran_totals[domain]
        if beds > 0 and veterans > 0:
            x_values.append(beds)
            y_values.append(veterans)
            labels.append(domain)

plt.scatter(x_values, y_values, s=100, alpha=0.7, color='steelblue')
for i, label in enumerate(labels):
    plt.annotate(label, (x_values[i], y_values[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

# 회귀선 추가
if len(x_values) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    line = slope * np.array(x_values) + intercept
    plt.plot(x_values, line, color='red', linestyle='--', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.0f}')
    
    # R² 값 계산
    r_squared = r_value ** 2
    equation_text = f'y = {slope:.2f}x + {intercept:.0f}\nR² = {r_squared:.3f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=12)

plt.xlabel('일반병상 수')
plt.ylabel('보훈대상자 수')
plt.title('권역별 병상 수 vs 보훈대상자 수', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('imgs/EDA_병원시설/002/병상수_vs_보훈대상자_산점도.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 적절성 평가
print("\n=== 권역별 의료자원 적절성 평가 ===")
for domain in hospital_domain.index:
    if domain in veteran_totals.index:
        veterans = veteran_totals[domain]
        beds = hospital_domain.loc[domain, '일반입원실_일반']
        specialists = hospital_domain.loc[domain, specialist_cols].sum()  # 모든 전문의 합계
        
        print(f"\n{domain}:")
        print(f"  보훈대상자: {veterans:,}명")
        print(f"  일반병상: {beds:.0f}개")
        print(f"  전문의 총수: {specialists:.0f}명")
        
        if beds > 0:
            bed_ratio = veterans / beds
            print(f"  보훈대상자/병상 비율: {bed_ratio:.1f}")
            
            if bed_ratio > 100:
                print("  → 병상 부족 (보훈대상자 대비)")
            elif bed_ratio < 50:
                print("  → 병상 여유")
            else:
                print("  → 적정 수준")
        
        if specialists > 0:
            specialist_ratio = veterans / specialists
            print(f"  보훈대상자/전문의 비율: {specialist_ratio:.1f}")
            
            if specialist_ratio > 1000:
                print("  → 전문의 부족 (보훈대상자 대비)")
            elif specialist_ratio < 500:
                print("  → 전문의 여유")
            else:
                print("  → 적정 수준")

# 8. 결과 저장
# 권역별 보훈대상자 수
veteran_totals_df = pd.DataFrame(list(veteran_totals.items()), columns=['권역', '보훈대상자수'])
veteran_totals_df.to_csv('new_merged_data/보훈대상자_권역별_총인원.csv', index=False, encoding='utf-8-sig')

# 권역별 의료자원 대비 보훈대상자 비율
ratio_df.to_csv('new_merged_data/보훈대상자_대비_의료자원_비율.csv', encoding='utf-8-sig')

print("\n=== 처리 완료 ===")
print("1. 권역별 보훈대상자 총인원: new_merged_data/보훈대상자_권역별_총인원.csv")
print("2. 보훈대상자 대비 의료자원 비율: new_merged_data/보훈대상자_대비_의료자원_비율.csv")
print("3. 시각화 파일: imgs/EDA_병원시설/002/ 폴더")