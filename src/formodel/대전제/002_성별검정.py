import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
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
    
    # 문자열로 변환
    value_str = str(value).strip()
    
    # 쉼표 제거
    value_str = value_str.replace(',', '')
    
    # 숫자가 아닌 문자 제거 (공백, 특수문자 등)
    value_str = re.sub(r'[^\d]', '', value_str)
    
    if value_str == '':
        return 0
    
    return int(value_str)

def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    
    # 1. 보훈병원 이용자 성별 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/전체_성별_이용자수.csv')
    
    # 2. 전체 보훈대상자 데이터 로드 및 전처리
    total_data = pd.read_csv('new_merged_data/국가보훈부_보훈대상자 성별연령별 실인원현황.csv')
    
    # 2024-12-31 기준일만 필터링 (2023-12-31이 없으므로 가장 가까운 날짜 사용)
    total_data = total_data[total_data['기준일'] == '2024-12-31'].copy()
    
    # 숫자 데이터 정리
    total_data['남'] = total_data['남'].apply(clean_number_string)
    total_data['여'] = total_data['여'].apply(clean_number_string)
    
    # 성별별 총합 계산
    total_male = total_data['남'].sum()
    total_female = total_data['여'].sum()
    
    # DataFrame으로 변환
    total_gender_df = pd.DataFrame({
        '성별': ['남', '여'],
        '전체대상자수': [total_male, total_female]
    })
    
    return hospital_data, total_gender_df

def prepare_contingency_table(hospital_data, total_data):
    """카이제곱 검정을 위한 분할표 준비"""
    
    # 병원 이용자 데이터 컬럼명 통일
    hospital_data = hospital_data.rename(columns={'이용자수': '병원이용자수'})
    
    # 두 데이터프레임을 성별 기준으로 병합
    merged_df = pd.merge(hospital_data, total_data, on='성별', how='inner')
    
    # 분할표 생성 (병원이용자수 vs 전체대상자수)
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
    
    # 1. 성별 분포 비교 그래프
    x = np.arange(len(merged_df))
    width = 0.35
    
    ax1.bar(x - width/2, merged_df['병원이용자수'], width, label='병원이용자', alpha=0.8)
    ax1.bar(x + width/2, merged_df['전체대상자수'], width, label='전체대상자', alpha=0.8)
    
    ax1.set_xlabel('성별')
    ax1.set_ylabel('인원수')
    ax1.set_title('성별 분포 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged_df['성별'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 비율 비교 그래프
    hospital_ratio = merged_df['병원이용자수'] / merged_df['병원이용자수'].sum()
    total_ratio = merged_df['전체대상자수'] / merged_df['전체대상자수'].sum()
    
    ax2.bar(x - width/2, hospital_ratio, width, label='병원이용자 비율', alpha=0.8)
    ax2.bar(x + width/2, total_ratio, width, label='전체대상자 비율', alpha=0.8)
    
    ax2.set_xlabel('성별')
    ax2.set_ylabel('비율')
    ax2.set_title('성별 비율 비교')
    ax2.set_xticks(x)
    ax2.set_xticklabels(merged_df['성별'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/formodel/대전제/성별분포_카이제곱검정_결과.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    
    print("=== 보훈병원 이용자 vs 전체 보훈대상자 성별 분포 카이제곱 검정 ===\n")
    
    # 1. 데이터 로드 및 전처리
    print("1. 데이터 로드 중...")
    hospital_data, total_data = load_and_prepare_data()
    
    print(f"   - 병원 이용자 데이터: {len(hospital_data)}개 성별")
    print(f"   - 전체 대상자 데이터: {len(total_data)}개 성별")
    
    # 2. 데이터 확인
    print("\n2. 병원 이용자 성별 분포:")
    print(hospital_data)
    
    print("\n3. 전체 보훈대상자 성별 분포:")
    print(total_data)
    
    # 3. 분할표 준비
    print("\n4. 카이제곱 검정용 분할표 준비 중...")
    merged_df, contingency_table = prepare_contingency_table(hospital_data, total_data)
    
    print("   분할표:")
    print(merged_df)
    print(f"\n   분할표 형태: {contingency_table.shape}")
    
    # 4. 카이제곱 검정 수행
    print("\n5. 카이제곱 검정 수행 중...")
    test_results = perform_chi_square_test(contingency_table)
    
    # 5. 결과 출력
    print("\n=== 검정 결과 ===")
    print(f"카이제곱 통계량: {test_results['chi2_statistic']:.4f}")
    print(f"p-value: {test_results['p_value']:.6f}")
    print(f"자유도: {test_results['degrees_of_freedom']}")
    
    # 6. 결과 해석
    print("\n=== 결과 해석 ===")
    alpha = 0.05
    if test_results['p_value'] < alpha:
        print(f"p-value ({test_results['p_value']:.6f}) < α ({alpha})")
        print("→ 귀무가설 기각: 보훈병원 이용자와 전체 보훈대상자의 성별 분포가 유의하게 다릅니다.")
    else:
        print(f"p-value ({test_results['p_value']:.6f}) ≥ α ({alpha})")
        print("→ 귀무가설 채택: 보훈병원 이용자와 전체 보훈대상자의 성별 분포가 일치합니다.")
    
    # 7. 시각화
    print("\n6. 결과 시각화 중...")
    create_visualization(merged_df, test_results)
    
    # 8. 추가 분석 정보
    print("\n=== 추가 분석 정보 ===")
    print("성별별 이용률 (병원이용자수/전체대상자수):")
    merged_df['이용률'] = (merged_df['병원이용자수'] / merged_df['전체대상자수'] * 100).round(2)
    print(merged_df[['성별', '병원이용자수', '전체대상자수', '이용률']])
    
    print(f"\n전체 이용률: {(merged_df['병원이용자수'].sum() / merged_df['전체대상자수'].sum() * 100):.2f}%")
    
    # 9. 성별 비율 비교
    print("\n=== 성별 비율 비교 ===")
    hospital_male_ratio = (merged_df[merged_df['성별'] == '남']['병원이용자수'].iloc[0] / 
                          merged_df['병원이용자수'].sum() * 100).round(2)
    hospital_female_ratio = (merged_df[merged_df['성별'] == '여']['병원이용자수'].iloc[0] / 
                            merged_df['병원이용자수'].sum() * 100).round(2)
    
    total_male_ratio = (merged_df[merged_df['성별'] == '남']['전체대상자수'].iloc[0] / 
                       merged_df['전체대상자수'].sum() * 100).round(2)
    total_female_ratio = (merged_df[merged_df['성별'] == '여']['전체대상자수'].iloc[0] / 
                         merged_df['전체대상자수'].sum() * 100).round(2)
    
    print(f"병원 이용자 - 남성: {hospital_male_ratio}%, 여성: {hospital_female_ratio}%")
    print(f"전체 대상자 - 남성: {total_male_ratio}%, 여성: {total_female_ratio}%")
    
    return test_results, merged_df

if __name__ == "__main__":
    results, data = main()
