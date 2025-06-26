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
    """데이터 로드 및 전처리"""
    
    # 1. 보훈병원 이용자 지역별 데이터 로드
    hospital_data = pd.read_csv('new_merged_data/지역별_총이용자수.csv')
    
    # 2. 전체 보훈대상자 지역별 데이터 로드
    total_data = pd.read_csv('new_merged_data/보훈대상자_권역별_총인원.csv')
    
    return hospital_data, total_data

def normalize_region_names(hospital_data, total_data):
    """지역명 정규화 (권역명과 지역명 매칭)"""
    
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
    
    # 병원 이용자 데이터 컬럼명 통일
    hospital_data = hospital_data.rename(columns={'총이용자수': '병원이용자수'})
    total_data = total_data.rename(columns={'보훈대상자수': '전체대상자수'})
    
    return hospital_data, total_data

def prepare_contingency_table(hospital_data, total_data):
    """카이제곱 검정을 위한 분할표 준비"""
    
    # 두 데이터프레임을 지역 기준으로 병합
    merged_df = pd.merge(hospital_data, total_data, on='지역', how='inner')
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 1. 지역별 분포 비교 그래프
    x = np.arange(len(merged_df))
    width = 0.35
    
    ax1.bar(x - width/2, merged_df['병원이용자수'], width, label='병원이용자', alpha=0.8)
    ax1.bar(x + width/2, merged_df['전체대상자수'], width, label='전체대상자', alpha=0.8)
    
    ax1.set_xlabel('지역')
    ax1.set_ylabel('인원수')
    ax1.set_title('지역별 분포 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged_df['지역'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 비율 비교 그래프
    hospital_ratio = merged_df['병원이용자수'] / merged_df['병원이용자수'].sum()
    total_ratio = merged_df['전체대상자수'] / merged_df['전체대상자수'].sum()
    
    ax2.bar(x - width/2, hospital_ratio, width, label='병원이용자 비율', alpha=0.8)
    ax2.bar(x + width/2, total_ratio, width, label='전체대상자 비율', alpha=0.8)
    
    ax2.set_xlabel('지역')
    ax2.set_ylabel('비율')
    ax2.set_title('지역별 비율 비교')
    ax2.set_xticks(x)
    ax2.set_xticklabels(merged_df['지역'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/formodel/대전제/지역별분포_카이제곱검정_결과.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    
    print("=== 보훈병원 이용자 vs 전체 보훈대상자 지역별 분포 카이제곱 검정 ===\n")
    
    # 1. 데이터 로드 및 전처리
    print("1. 데이터 로드 중...")
    hospital_data, total_data = load_and_prepare_data()
    
    print(f"   - 병원 이용자 데이터: {len(hospital_data)}개 지역")
    print(f"   - 전체 대상자 데이터: {len(total_data)}개 지역")
    
    # 2. 지역명 정규화
    print("\n2. 지역명 정규화 중...")
    hospital_data, total_data = normalize_region_names(hospital_data, total_data)
    
    # 3. 데이터 확인
    print("\n3. 병원 이용자 지역별 분포:")
    print(hospital_data)
    
    print("\n4. 전체 보훈대상자 지역별 분포:")
    print(total_data)
    
    # 4. 분할표 준비
    print("\n5. 카이제곱 검정용 분할표 준비 중...")
    merged_df, contingency_table = prepare_contingency_table(hospital_data, total_data)
    
    print("   분할표:")
    print(merged_df)
    print(f"\n   분할표 형태: {contingency_table.shape}")
    
    # 5. 카이제곱 검정 수행
    print("\n6. 카이제곱 검정 수행 중...")
    test_results = perform_chi_square_test(contingency_table)
    
    # 6. 결과 출력
    print("\n=== 검정 결과 ===")
    print(f"카이제곱 통계량: {test_results['chi2_statistic']:.4f}")
    print(f"p-value: {test_results['p_value']:.6f}")
    print(f"자유도: {test_results['degrees_of_freedom']}")
    
    # 7. 결과 해석
    print("\n=== 결과 해석 ===")
    alpha = 0.05
    if test_results['p_value'] < alpha:
        print(f"p-value ({test_results['p_value']:.6f}) < α ({alpha})")
        print("→ 귀무가설 기각: 보훈병원 이용자와 전체 보훈대상자의 지역별 분포가 유의하게 다릅니다.")
    else:
        print(f"p-value ({test_results['p_value']:.6f}) ≥ α ({alpha})")
        print("→ 귀무가설 채택: 보훈병원 이용자와 전체 보훈대상자의 지역별 분포가 일치합니다.")
    
    # 8. 시각화
    print("\n7. 결과 시각화 중...")
    create_visualization(merged_df, test_results)
    
    # 9. 추가 분석 정보
    print("\n=== 추가 분석 정보 ===")
    print("지역별 이용률 (병원이용자수/전체대상자수):")
    merged_df['이용률'] = (merged_df['병원이용자수'] / merged_df['전체대상자수'] * 100).round(2)
    print(merged_df[['지역', '병원이용자수', '전체대상자수', '이용률']])
    
    print(f"\n전체 이용률: {(merged_df['병원이용자수'].sum() / merged_df['전체대상자수'].sum() * 100):.2f}%")
    
    # 10. 지역별 순위 분석
    print("\n=== 지역별 순위 분석 ===")
    merged_df_sorted = merged_df.sort_values('이용률', ascending=False)
    print("이용률 순위 (높은 순):")
    for i, (_, row) in enumerate(merged_df_sorted.iterrows(), 1):
        print(f"{i}위: {row['지역']} ({row['이용률']}%)")
    
    # 11. 지역별 비율 비교
    print("\n=== 지역별 비율 비교 ===")
    hospital_ratio = merged_df['병원이용자수'] / merged_df['병원이용자수'].sum() * 100
    total_ratio = merged_df['전체대상자수'] / merged_df['전체대상자수'].sum() * 100
    
    for i, (_, row) in enumerate(merged_df.iterrows()):
        region = row['지역']
        hosp_pct = hospital_ratio.iloc[i]
        total_pct = total_ratio.iloc[i]
        print(f"{region}: 병원이용자 {hosp_pct:.1f}% vs 전체대상자 {total_pct:.1f}%")
    
    return test_results, merged_df

if __name__ == "__main__":
    results, data = main()
