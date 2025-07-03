import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from datetime import datetime
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== 병상 분배 휴리스틱 균등화 모델 ===")
print("📊 가동률 균등화를 위한 반복적 조정 시스템")
print()

# --------------------------------------------------
# 1) 데이터 로드 및 전처리
# --------------------------------------------------
print("1/5: 데이터 로드 및 전처리 중...")

# 병원 통합 데이터 로드 (병상 현황)
df_hospital = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')

# 입원 예측 데이터 로드
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv')

# 최근 연도(2024)만 사용
df_pred = df_pred[df_pred['예측연도'] == 2024]

# 병원명 컬럼명 통일 및 매칭 수정
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_hospital['병원명'] = df_hospital['병원명'].astype(str).str.strip()

# 병원명 매칭 수정: "중앙" → "서울"
df_pred['병원명'] = df_pred['병원명'].replace('중앙', '서울')

# 병상 관련 컬럼 추출 (전문의수로 끝나지 않는 컬럼들)
bed_columns = [col for col in df_hospital.columns if not col.endswith('_전문의수') and col != '병원명']
df_hospital['총병상수'] = df_hospital[bed_columns].sum(axis=1)

# 병원별 예측 환자 수 집계
hospital_patients = df_pred.groupby('병원명')['XGB예측'].sum().reset_index()
hospital_patients.columns = ['병원명', '총예측환자수']

# 병원별 현재 상황 분석
data = []
total_beds = 0
total_patients = 0

for idx, row in hospital_patients.iterrows():
    병원 = row['병원명']
    예측환자수 = row['총예측환자수']
    hosp_row = df_hospital[df_hospital['병원명'] == 병원]
    
    if len(hosp_row) > 0:
        현재병상수 = float(hosp_row['총병상수'].iloc[0])
        if pd.notnull(현재병상수) and 현재병상수 > 0:
            data.append({
                '병원명': 병원,
                '현재병상수': 현재병상수,
                '예측환자수': 예측환자수
            })
            total_beds += 현재병상수
            total_patients += 예측환자수
            print(f"✅ {병원}: 병상 {현재병상수}개, 예측환자 {예측환자수:.0f}명")
        else:
            print(f"⚠️ {병원}: 병상 데이터 없음")
    else:
        print(f"❌ {병원}: 병원 데이터 없음")

current_df = pd.DataFrame(data)

print(f"\n✅ 데이터 로드 및 집계 완료: 병원 수 {len(current_df)}개")
print(f"  - 총 병상 수: {total_beds:.0f}개")
print(f"  - 총 예측 환자 수: {total_patients:.0f}명")
print()

# --------------------------------------------------
# 2) 휴리스틱 가동률 균등화 알고리즘
# --------------------------------------------------
print("2/5: 휴리스틱 가동률 균등화 알고리즘 실행 중...")

def calculate_utilization(beds, patients):
    """가동률 계산"""
    daily_patients = patients / 365
    return (daily_patients / beds) * 100

def redistribute_beds(current_df, target_utilization=70.0, max_iterations=100):
    """병상 재분배 휴리스틱 알고리즘"""
    
    # 초기 상태 복사
    df = current_df.copy()
    df['최적병상수'] = df['현재병상수'].copy()
    
    # 현재 가동률 계산
    df['현재_가동률'] = df.apply(lambda row: calculate_utilization(row['현재병상수'], row['예측환자수']), axis=1)
    
    print(f"초기 가동률 분포:")
    for _, row in df.iterrows():
        print(f"  {row['병원명']}: {row['현재_가동률']:.1f}%")
    
    print(f"\n목표 가동률: {target_utilization}%")
    print("반복적 조정 시작...")
    
    for iteration in range(max_iterations):
        # 현재 가동률 계산
        df['현재_가동률'] = df.apply(lambda row: calculate_utilization(row['최적병상수'], row['예측환자수']), axis=1)
        
        # 가동률 표준편차 계산
        current_std = df['현재_가동률'].std()
        
        if iteration % 10 == 0:
            print(f"반복 {iteration}: 가동률 표준편차 = {current_std:.2f}%")
        
        # 수렴 조건: 표준편차가 충분히 작거나 변화가 없으면 종료
        if current_std < 2.0:
            print(f"수렴 조건 달성 (표준편차 < 2.0%)")
            break
        
        # 가동률이 높은 병원에서 낮은 병원으로 병상 이동
        high_util = df[df['현재_가동률'] > target_utilization + 5].copy()
        low_util = df[df['현재_가동률'] < target_utilization - 5].copy()
        
        if len(high_util) == 0 or len(low_util) == 0:
            print("더 이상 조정할 병원이 없습니다.")
            break
        
        # 가장 높은 가동률 병원에서 가장 낮은 가동률 병원으로 병상 이동
        high_util = high_util.sort_values('현재_가동률', ascending=False)
        low_util = low_util.sort_values('현재_가동률', ascending=True)
        
        for _, high_row in high_util.iterrows():
            for _, low_row in low_util.iterrows():
                # 제약조건 확인: 현재의 80-120% 범위
                high_min = max(1, int(high_row['현재병상수'] * 0.8))
                high_max = int(high_row['현재병상수'] * 1.2)
                low_min = max(1, int(low_row['현재병상수'] * 0.8))
                low_max = int(low_row['현재병상수'] * 1.2)
                
                # 병상 감소 가능 여부 확인
                if df.loc[df['병원명'] == high_row['병원명'], '최적병상수'].iloc[0] > high_min:
                    # 병상 증가 가능 여부 확인
                    if df.loc[df['병원명'] == low_row['병원명'], '최적병상수'].iloc[0] < low_max:
                        # 병상 이동 (1개씩)
                        df.loc[df['병원명'] == high_row['병원명'], '최적병상수'] -= 1
                        df.loc[df['병원명'] == low_row['병원명'], '최적병상수'] += 1
                        break
        
        # 총 병상 수 유지 확인
        total_optimal = df['최적병상수'].sum()
        if abs(total_optimal - total_beds) > 1:
            # 총 병상 수 조정
            diff = int(total_beds - total_optimal)
            if diff > 0:
                # 부족한 경우 가장 낮은 가동률 병원에 추가
                lowest_util_idx = df['현재_가동률'].idxmin()
                max_beds = int(df.loc[lowest_util_idx, '현재병상수'] * 1.2)
                if df.loc[lowest_util_idx, '최적병상수'] + diff <= max_beds:
                    df.loc[lowest_util_idx, '최적병상수'] += diff
            else:
                # 초과한 경우 가장 높은 가동률 병원에서 감소
                highest_util_idx = df['현재_가동률'].idxmax()
                min_beds = max(1, int(df.loc[highest_util_idx, '현재병상수'] * 0.8))
                if df.loc[highest_util_idx, '최적병상수'] + diff >= min_beds:
                    df.loc[highest_util_idx, '최적병상수'] += diff
    
    # 최종 가동률 계산
    df['최적_가동률'] = df.apply(lambda row: calculate_utilization(row['최적병상수'], row['예측환자수']), axis=1)
    
    return df

# 휴리스틱 알고리즘 실행
print("휴리스틱 알고리즘 실행 중...")
result_df = redistribute_beds(current_df, target_utilization=70.0)

print("✅ 휴리스틱 알고리즘 완료!")
print()

# --------------------------------------------------
# 3) 결과 분석
# --------------------------------------------------
print("3/5: 결과 분석 중...")

# 변화량 계산
result_df['변화량'] = result_df['최적병상수'] - result_df['현재병상수']
result_df['변화율'] = (result_df['변화량'] / result_df['현재병상수'] * 100)

print("=== 휴리스틱 최적화 결과 ===")
print(result_df[['병원명', '현재병상수', '최적병상수', '변화량', '변화율', 
                '현재_가동률', '최적_가동률']].round(2).to_string(index=False))

# 성능 지표 계산
현재_가동률_표준편차 = result_df['현재_가동률'].std()
최적_가동률_표준편차 = result_df['최적_가동률'].std()
가동률_개선도 = (현재_가동률_표준편차 - 최적_가동률_표준편차) / 현재_가동률_표준편차 * 100

print(f"\n📊 성능 지표:")
print(f"  - 현재 가동률 표준편차: {현재_가동률_표준편차:.2f}%")
print(f"  - 최적 가동률 표준편차: {최적_가동률_표준편차:.2f}%")
print(f"  - 가동률 개선도: {가동률_개선도:.1f}%")

print()

# --------------------------------------------------
# 4) 결과 저장
# --------------------------------------------------
print("4/5: 결과 저장 중...")

results_dir = "optimization_results_병상_분배_휴리스틱_균등화"
os.makedirs(results_dir, exist_ok=True)

# 결과 저장
result_df.to_csv(f"{results_dir}/병상_분배_휴리스틱_결과.csv", index=False, encoding='utf-8-sig')
print(f"✅ 결과 저장 완료: {results_dir}/병상_분배_휴리스틱_결과.csv")

print()

# --------------------------------------------------
# 5) 시각화
# --------------------------------------------------
print("5/5: 시각화 생성 중...")

plt.figure(figsize=(15, 10))

# 서브플롯 1: 현재 vs 최적 병상 수 비교
plt.subplot(2, 3, 1)
plt.scatter(result_df['현재병상수'], result_df['최적병상수'], alpha=0.7, s=100)
max_beds = max(result_df['현재병상수'].max(), result_df['최적병상수'].max())
plt.plot([0, max_beds], [0, max_beds], 'r--', alpha=0.5)
plt.xlabel('현재 병상 수')
plt.ylabel('최적 병상 수')
plt.title('현재 vs 최적 병상 수 (휴리스틱)')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 병상 변화량
plt.subplot(2, 3, 2)
colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in result_df['변화량']]
plt.barh(result_df['병원명'], result_df['변화량'], color=colors, alpha=0.7)
plt.xlabel('병상 수 변화량')
plt.title('병원별 병상 수 변화량 (휴리스틱)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 3: 가동률 비교
plt.subplot(2, 3, 3)
x = np.arange(len(result_df))
width = 0.35
plt.bar(x - width/2, result_df['현재_가동률'], width, label='현재', alpha=0.7)
plt.bar(x + width/2, result_df['최적_가동률'], width, label='최적', alpha=0.7)
plt.xlabel('병원')
plt.ylabel('병상가동률 (%)')
plt.title('현재 vs 최적 병상가동률 (휴리스틱)')
plt.xticks(x, list(result_df['병원명']), rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 가동률 개선도
plt.subplot(2, 3, 4)
개선도 = result_df['최적_가동률'] - result_df['현재_가동률']
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in 개선도]
plt.barh(result_df['병원명'], 개선도, color=colors, alpha=0.7)
plt.xlabel('가동률 개선도 (%)')
plt.title('병원별 가동률 개선도 (휴리스틱)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 5: 현재 vs 최적 가동률 산점도
plt.subplot(2, 3, 5)
plt.scatter(result_df['현재_가동률'], result_df['최적_가동률'], 
           alpha=0.7, s=100, c=result_df['변화량'], cmap='RdYlBu')
plt.colorbar(label='변화량')
max_util = max(result_df['현재_가동률'].max(), result_df['최적_가동률'].max())
plt.plot([0, max_util], [0, max_util], 'r--', alpha=0.5)
plt.xlabel('현재 병상가동률 (%)')
plt.ylabel('최적 병상가동률 (%)')
plt.title('현재 vs 최적 가동률 비교 (휴리스틱)')
plt.grid(True, alpha=0.3)

# 서브플롯 6: 가동률 분포 비교
plt.subplot(2, 3, 6)
plt.hist([result_df['현재_가동률'], result_df['최적_가동률']], 
         label=['현재', '최적'], alpha=0.7, bins=10)
plt.xlabel('병상가동률 (%)')
plt.ylabel('병원 수')
plt.title('가동률 분포 비교 (휴리스틱)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{results_dir}/병상_분배_휴리스틱_시각화.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 모든 결과가 {results_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 병상 분배 휴리스틱 균등화 완료!")
print("="*60) 