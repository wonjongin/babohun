import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def clean_number_string(value):
    """숫자 문자열에서 쉼표 제거 및 정수 변환"""
    if pd.isna(value) or value == '-' or value == '':
        return 0
    
    value_str = str(value).strip()
    value_str = value_str.replace(',', '')
    value_str = re.sub(r'[^\d]', '', value_str)
    
    if value_str == '':
        return 0
    
    return int(value_str)

def perform_proportion_test(count1, total1, count2, total2, alpha=0.05):
    """비율 검정 수행 (두 집단 간 비율 차이 검정) - 직접 구현"""
    
    # 비율 계산
    p1 = count1 / total1
    p2 = count2 / total2
    
    # 비율 차이
    diff = p1 - p2
    
    # 통합 비율 계산
    p_pooled = (count1 + count2) / (total1 + total2)
    
    # 표준오차 계산
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/total1 + 1/total2))
    
    # Z-통계량 계산
    z_stat = diff / se
    
    # p-value 계산 (양측 검정)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    # 신뢰구간 계산 (95%)
    se_ci = np.sqrt(p1 * (1-p1) / total1 + p2 * (1-p2) / total2)
    ci_lower = diff - 1.96 * se_ci
    ci_upper = diff + 1.96 * se_ci
    
    return {
        'proportion1': p1,
        'proportion2': p2,
        'difference': diff,
        'z_statistic': z_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha
    }

def analyze_age_proportions_vs_bohun():
    """연령대별 비율 검정 (보훈병원 이용자 vs 보훈대상자)"""
    print("=== 연령대별 비율 검정 (보훈병원 이용자 vs 보훈대상자) ===")
    
    # 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/연령대별_총이용자수.csv')
    total_data = pd.read_csv('new_merged_data/보훈대상자_10세단위_전체인원.csv')
    
    # 100대 제외, 0~90대만 사용
    valid_ages = [f'{i}대' for i in range(0, 100, 10)]
    hosp_df = hospital_data[hospital_data['연령대'].isin(valid_ages)].copy()
    total_df = total_data[total_data['연령대'].isin(valid_ages)].copy()
    
    # 컬럼명 통일
    hosp_df = hosp_df.rename(columns={'총이용자수': '병원이용자수'})
    total_df = total_df.rename(columns={'인원수': '전체대상자수'})
    
    # 병합
    merged_df = pd.merge(hosp_df, total_df, on='연령대', how='inner')
    
    # 전체 합계
    total_hospital = merged_df['병원이용자수'].sum()
    total_bohun = merged_df['전체대상자수'].sum()
    
    print(f"전체 병원이용자: {total_hospital:,}명")
    print(f"전체 보훈대상자: {total_bohun:,}명")
    print(f"전체 이용률: {(total_hospital/total_bohun*100):.2f}%")
    
    # 연령대별 비율 검정
    results = []
    for _, row in merged_df.iterrows():
        age = row['연령대']
        hospital_count = row['병원이용자수']
        bohun_count = row['전체대상자수']
        
        test_result = perform_proportion_test(
            hospital_count, total_hospital,
            bohun_count, total_bohun
        )
        
        results.append({
            '연령대': age,
            '병원이용자수': hospital_count,
            '보훈대상자수': bohun_count,
            '병원비율': test_result['proportion1'],
            '보훈비율': test_result['proportion2'],
            '비율차이': test_result['difference'],
            'Z통계량': test_result['z_statistic'],
            'p_value': test_result['p_value'],
            '신뢰구간_하한': test_result['ci_lower'],
            '신뢰구간_상한': test_result['ci_upper'],
            '유의성': test_result['significant']
        })
    
    results_df = pd.DataFrame(results)
    
    # 결과 출력
    print("\n=== 연령대별 비율 검정 결과 ===")
    print("-" * 100)
    print(f"{'연령대':<8} {'병원비율':<10} {'보훈비율':<10} {'비율차이':<10} {'Z통계량':<10} {'p-value':<10} {'유의성':<8}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        significance = "유의함" if row['유의성'] else "무의미"
        print(f"{row['연령대']:<8} {row['병원비율']:<10.4f} {row['보훈비율']:<10.4f} "
              f"{row['비율차이']:<10.4f} {row['Z통계량']:<10.2f} {row['p_value']:<10.4f} {significance:<8}")
    
    return results_df

def analyze_gender_proportions_vs_bohun():
    """성별 비율 검정 (보훈병원 이용자 vs 보훈대상자)"""
    print("\n=== 성별 비율 검정 (보훈병원 이용자 vs 보훈대상자) ===")
    
    # 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/전체_성별_이용자수.csv')
    total_data = pd.read_csv('new_merged_data/국가보훈부_보훈대상자 성별연령별 실인원현황.csv')
    
    # 2024-12-31 기준일만 필터링
    total_data = total_data[total_data['기준일'] == '2024-12-31'].copy()
    
    # 숫자 데이터 정리
    total_data['남'] = total_data['남'].apply(clean_number_string)
    total_data['여'] = total_data['여'].apply(clean_number_string)
    
    # 성별별 총합 계산
    total_male = total_data['남'].sum()
    total_female = total_data['여'].sum()
    
    total_gender_df = pd.DataFrame({
        '성별': ['남', '여'],
        '전체대상자수': [total_male, total_female]
    })
    
    # 병원 이용자 데이터 컬럼명 통일
    hospital_data = hospital_data.rename(columns={'이용자수': '병원이용자수'})
    
    # 병합
    merged_df = pd.merge(hospital_data, total_gender_df, on='성별', how='inner')
    
    # 전체 합계
    total_hospital = merged_df['병원이용자수'].sum()
    total_bohun = merged_df['전체대상자수'].sum()
    
    print(f"전체 병원이용자: {total_hospital:,}명")
    print(f"전체 보훈대상자: {total_bohun:,}명")
    print(f"전체 이용률: {(total_hospital/total_bohun*100):.2f}%")
    
    # 성별 비율 검정
    results = []
    for _, row in merged_df.iterrows():
        gender = row['성별']
        hospital_count = row['병원이용자수']
        bohun_count = row['전체대상자수']
        
        test_result = perform_proportion_test(
            hospital_count, total_hospital,
            bohun_count, total_bohun
        )
        
        results.append({
            '성별': gender,
            '병원이용자수': hospital_count,
            '보훈대상자수': bohun_count,
            '병원비율': test_result['proportion1'],
            '보훈비율': test_result['proportion2'],
            '비율차이': test_result['difference'],
            'Z통계량': test_result['z_statistic'],
            'p_value': test_result['p_value'],
            '신뢰구간_하한': test_result['ci_lower'],
            '신뢰구간_상한': test_result['ci_upper'],
            '유의성': test_result['significant']
        })
    
    results_df = pd.DataFrame(results)
    
    # 결과 출력
    print("\n=== 성별 비율 검정 결과 ===")
    print("-" * 100)
    print(f"{'성별':<6} {'병원비율':<10} {'보훈비율':<10} {'비율차이':<10} {'Z통계량':<10} {'p-value':<10} {'유의성':<8}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        significance = "유의함" if row['유의성'] else "무의미"
        print(f"{row['성별']:<6} {row['병원비율']:<10.4f} {row['보훈비율']:<10.4f} "
              f"{row['비율차이']:<10.4f} {row['Z통계량']:<10.2f} {row['p_value']:<10.4f} {significance:<8}")
    
    return results_df

def analyze_region_proportions_vs_bohun():
    """지역별 비율 검정 (보훈병원 이용자 vs 보훈대상자)"""
    print("\n=== 지역별 비율 검정 (보훈병원 이용자 vs 보훈대상자) ===")
    
    # 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/지역별_총이용자수.csv')
    total_data = pd.read_csv('new_merged_data/보훈대상자_권역별_총인원.csv')
    
    # 지역명 매핑
    region_mapping = {
        '광주권역': '광주',
        '대구권역': '대구', 
        '대전권역': '대전',
        '부산권역': '부산',
        '인천권역': '인천',
        '중앙권역': '중앙'
    }
    
    # 전체 대상자 데이터의 지역명 변경
    total_data['지역'] = total_data['권역'].map(region_mapping)
    total_data = total_data.drop('권역', axis=1)
    
    # 컬럼명 통일
    hospital_data = hospital_data.rename(columns={'총이용자수': '병원이용자수'})
    total_data = total_data.rename(columns={'보훈대상자수': '전체대상자수'})
    
    # 병합
    merged_df = pd.merge(hospital_data, total_data, on='지역', how='inner')
    
    # 전체 합계
    total_hospital = merged_df['병원이용자수'].sum()
    total_bohun = merged_df['전체대상자수'].sum()
    
    print(f"전체 병원이용자: {total_hospital:,}명")
    print(f"전체 보훈대상자: {total_bohun:,}명")
    print(f"전체 이용률: {(total_hospital/total_bohun*100):.2f}%")
    
    # 지역별 비율 검정
    results = []
    for _, row in merged_df.iterrows():
        region = row['지역']
        hospital_count = row['병원이용자수']
        bohun_count = row['전체대상자수']
        
        test_result = perform_proportion_test(
            hospital_count, total_hospital,
            bohun_count, total_bohun
        )
        
        results.append({
            '지역': region,
            '병원이용자수': hospital_count,
            '보훈대상자수': bohun_count,
            '병원비율': test_result['proportion1'],
            '보훈비율': test_result['proportion2'],
            '비율차이': test_result['difference'],
            'Z통계량': test_result['z_statistic'],
            'p_value': test_result['p_value'],
            '신뢰구간_하한': test_result['ci_lower'],
            '신뢰구간_상한': test_result['ci_upper'],
            '유의성': test_result['significant']
        })
    
    results_df = pd.DataFrame(results)
    
    # 결과 출력
    print("\n=== 지역별 비율 검정 결과 ===")
    print("-" * 100)
    print(f"{'지역':<8} {'병원비율':<10} {'보훈비율':<10} {'비율차이':<10} {'Z통계량':<10} {'p-value':<10} {'유의성':<8}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        significance = "유의함" if row['유의성'] else "무의미"
        print(f"{row['지역']:<8} {row['병원비율']:<10.4f} {row['보훈비율']:<10.4f} "
              f"{row['비율차이']:<10.4f} {row['Z통계량']:<10.2f} {row['p_value']:<10.4f} {significance:<8}")
    
    return results_df

def create_proportion_visualization(age_results, gender_results, region_results):
    """비율 검정 결과 시각화"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 연령대별 비율 비교
    age_data = age_results.copy()
    x = np.arange(len(age_data))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, age_data['병원비율'], width, label='병원이용자 비율', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, age_data['보훈비율'], width, label='보훈대상자 비율', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('연령대')
    ax1.set_ylabel('비율')
    ax1.set_title('연령대별 비율 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(age_data['연령대'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 성별 비율 비교
    gender_data = gender_results.copy()
    x = np.arange(len(gender_data))
    
    bars3 = ax2.bar(x - width/2, gender_data['병원비율'], width, label='병원이용자 비율', alpha=0.8, color='skyblue')
    bars4 = ax2.bar(x + width/2, gender_data['보훈비율'], width, label='보훈대상자 비율', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('성별')
    ax2.set_ylabel('비율')
    ax2.set_title('성별 비율 비교')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gender_data['성별'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 지역별 비율 비교
    region_data = region_results.copy()
    x = np.arange(len(region_data))
    
    bars5 = ax3.bar(x - width/2, region_data['병원비율'], width, label='병원이용자 비율', alpha=0.8, color='skyblue')
    bars6 = ax3.bar(x + width/2, region_data['보훈비율'], width, label='보훈대상자 비율', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('지역')
    ax3.set_ylabel('비율')
    ax3.set_title('지역별 비율 비교')
    ax3.set_xticks(x)
    ax3.set_xticklabels(region_data['지역'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 비율 차이 요약
    all_diffs = []
    all_labels = []
    
    # 연령대별 차이
    for _, row in age_data.iterrows():
        all_diffs.append(row['비율차이'])
        all_labels.append(f"{row['연령대']}")
    
    # 성별 차이
    for _, row in gender_data.iterrows():
        all_diffs.append(row['비율차이'])
        all_labels.append(f"{row['성별']}")
    
    # 지역별 차이
    for _, row in region_data.iterrows():
        all_diffs.append(row['비율차이'])
        all_labels.append(f"{row['지역']}")
    
    colors = ['green' if d > 0 else 'red' for d in all_diffs]
    bars7 = ax4.bar(range(len(all_diffs)), all_diffs, color=colors, alpha=0.7)
    ax4.set_xlabel('변수')
    ax4.set_ylabel('비율 차이 (병원 - 보훈)')
    ax4.set_title('전체 비율 차이 요약')
    ax4.set_xticks(range(len(all_diffs)))
    ax4.set_xticklabels(all_labels, rotation=45)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/formodel/대전제/비율검정_결과.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    
    print("=== 보훈병원 이용자 vs 보훈대상자 비율 검정 ===\n")
    
    # 1. 연령대별 비율 검정
    age_results = analyze_age_proportions_vs_bohun()
    
    # 2. 성별 비율 검정
    gender_results = analyze_gender_proportions_vs_bohun()
    
    # 3. 지역별 비율 검정
    region_results = analyze_region_proportions_vs_bohun()
    
    # 4. 시각화
    print("\n4. 결과 시각화 중...")
    create_proportion_visualization(age_results, gender_results, region_results)
    
    # 5. 종합 분석
    print("\n" + "="*60)
    print("=== 비율 검정 종합 분석 ===")
    print("="*60)
    
    # 유의한 차이를 보이는 변수들
    significant_age = age_results[age_results['유의성'] == True]
    significant_gender = gender_results[gender_results['유의성'] == True]
    significant_region = region_results[region_results['유의성'] == True]
    
    print(f"\n유의한 차이를 보이는 변수:")
    print(f"• 연령대: {len(significant_age)}개 ({len(significant_age)/len(age_results)*100:.1f}%)")
    print(f"• 성별: {len(significant_gender)}개 ({len(significant_gender)/len(gender_results)*100:.1f}%)")
    print(f"• 지역: {len(significant_region)}개 ({len(significant_region)/len(region_results)*100:.1f}%)")
    
    # 가장 큰 차이를 보이는 변수
    all_results = pd.concat([
        age_results[['연령대', '비율차이', 'p_value']].rename(columns={'연령대': '변수'}),
        gender_results[['성별', '비율차이', 'p_value']].rename(columns={'성별': '변수'}),
        region_results[['지역', '비율차이', 'p_value']].rename(columns={'지역': '변수'})
    ], ignore_index=True)
    
    max_diff_idx = all_results['비율차이'].abs().idxmax()
    max_diff = all_results.loc[max_diff_idx]
    print(f"\n가장 큰 비율 차이:")
    print(f"• 변수: {max_diff['변수']}")
    print(f"• 비율 차이: {max_diff['비율차이']:.4f}")
    print(f"• p-value: {max_diff['p_value']:.6f}")
    
    return {
        'age': age_results,
        'gender': gender_results,
        'region': region_results
    }

if __name__ == "__main__":
    results = main() 