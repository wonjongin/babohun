import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """데이터 로드 및 전처리 (연령대별 병원이용자수, 전체대상자수)"""
    # 1. 병원이용자 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/연령대별_총이용자수.csv')
    # 2. 전체 보훈대상자 데이터 로드
    total_data = pd.read_csv('new_merged_data/보훈대상자_10세단위_전체인원.csv')
    return hospital_data, total_data

def calculate_hospital_age_distribution(hospital_data):
    """보훈병원 이용자의 연령대별 분포 계산"""
    
    # 연령대 컬럼들 (20대부터 90대까지)
    age_columns = ['20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대']
    
    # 각 연령대별 총 이용자 수 계산
    age_distribution = {}
    for age in age_columns:
        age_distribution[age] = hospital_data[age].sum()
    
    # DataFrame으로 변환
    hospital_age_df = pd.DataFrame(list(age_distribution.items()), 
                                  columns=['연령대', '이용자수'])
    
    return hospital_age_df

def prepare_contingency_table(hospital_data, total_data):
    """카이제곱 검정을 위한 분할표 준비 (0~90대만, 100대 제외)"""
    # 100대 제외, 0~90대만 사용
    valid_ages = [f'{i}대' for i in range(0, 100, 10)]
    # 병원이용자
    hosp_df = hospital_data[hospital_data['연령대'].isin(valid_ages)].copy()
    # 전체대상자
    total_df = total_data[total_data['연령대'].isin(valid_ages)].copy()
    # 컬럼명 통일
    hosp_df = hosp_df.rename(columns={'총이용자수': '병원이용자수'})
    total_df = total_df.rename(columns={'인원수': '전체대상자수'})
    # 병합
    merged_df = pd.merge(hosp_df, total_df, on='연령대', how='inner')
    # 분할표
    contingency_table = merged_df[['병원이용자수', '전체대상자수']].values
    return merged_df, contingency_table

def perform_chi_square_test(contingency_table):
    """카이제곱 검정 수행"""
    
    # 카이제곱 검정 실행
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected
    }

def create_visualization(merged_df, test_results):
    """결과 시각화"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 연령대별 분포 비교 그래프
    x = np.arange(len(merged_df))
    width = 0.35
    
    ax1.bar(x - width/2, merged_df['병원이용자수'], width, label='병원이용자', alpha=0.8)
    ax1.bar(x + width/2, merged_df['전체대상자수'], width, label='전체대상자', alpha=0.8)
    
    ax1.set_xlabel('연령대')
    ax1.set_ylabel('인원수')
    ax1.set_title('연령대별 분포 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged_df['연령대'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 비율 비교 그래프
    hospital_ratio = merged_df['병원이용자수'] / merged_df['병원이용자수'].sum()
    total_ratio = merged_df['전체대상자수'] / merged_df['전체대상자수'].sum()
    
    ax2.bar(x - width/2, hospital_ratio, width, label='병원이용자 비율', alpha=0.8)
    ax2.bar(x + width/2, total_ratio, width, label='전체대상자 비율', alpha=0.8)
    
    ax2.set_xlabel('연령대')
    ax2.set_ylabel('비율')
    ax2.set_title('연령대별 비율 비교')
    ax2.set_xticks(x)
    ax2.set_xticklabels(merged_df['연령대'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/formodel/대전제/연령대분포_카이제곱검정_결과.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== 보훈병원 이용자 vs 전체 보훈대상자 연령대 분포 카이제곱 검정 (통합 병원이용자수 기준) ===\n")
    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    hospital_data, total_data = load_and_prepare_data()
    print(f"   - 병원이용자 데이터: {len(hospital_data)}개 연령대")
    print(f"   - 전체 대상자 데이터: {len(total_data)}개 연령대")
    # 2. 분할표 준비
    print("\n2. 카이제곱 검정용 분할표 준비 중...")
    merged_df, contingency_table = prepare_contingency_table(hospital_data, total_data)
    print("   분할표:")
    print(merged_df)
    print(f"\n   분할표 형태: {contingency_table.shape}")
    # 3. 카이제곱 검정 수행
    print("\n3. 카이제곱 검정 수행 중...")
    test_results = perform_chi_square_test(contingency_table)
    # 4. 결과 출력
    print("\n=== 검정 결과 ===")
    print(f"카이제곱 통계량: {test_results['chi2_statistic']:.4f}")
    print(f"p-value: {test_results['p_value']:.6f}")
    print(f"자유도: {test_results['degrees_of_freedom']}")
    # 5. 결과 해석
    print("\n=== 결과 해석 ===")
    alpha = 0.05
    if test_results['p_value'] < alpha:
        print(f"p-value ({test_results['p_value']:.6f}) < α ({alpha})")
        print("→ 귀무가설 기각: 보훈병원 이용자와 전체 보훈대상자의 연령대 분포가 유의하게 다릅니다.")
    else:
        print(f"p-value ({test_results['p_value']:.6f}) ≥ α ({alpha})")
        print("→ 귀무가설 채택: 보훈병원 이용자와 전체 보훈대상자의 연령대 분포가 일치합니다.")
    # 6. 시각화
    print("\n4. 결과 시각화 중...")
    create_visualization(merged_df, test_results)
    # 7. 추가 분석 정보
    print("\n=== 추가 분석 정보 ===")
    merged_df['이용률'] = (merged_df['병원이용자수'] / merged_df['전체대상자수'] * 100).round(2)
    print(merged_df[['연령대', '병원이용자수', '전체대상자수', '이용률']])
    print(f"\n전체 이용률: {(merged_df['병원이용자수'].sum() / merged_df['전체대상자수'].sum() * 100):.2f}%")
    return test_results, merged_df

if __name__ == "__main__":
    results, data = main()
