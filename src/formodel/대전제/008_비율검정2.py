import statsmodels.api as sm
import statsmodels.stats.proportion as smp
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('final_merged_data/연도별 진료인원.csv')

# 데이터 확인
print("데이터 구조 확인:")
print(df.head())
print(f"\n데이터 형태: {df.shape}")
print(f"연도 범위: {df['년도'].min()} ~ {df['년도'].max()}")
print(f"지역: {df['지역'].unique()}")

p0 = 0.90  # 기준 비율 (90%)
alpha = 0.05

years = df['년도'].unique()
total_x, total_n = 0, 0

print("📊 연도별 보훈대상자 비율 검정 결과 (기준: 90%)")
print("-" * 70)

for year in sorted(years):
    df_year = df[df['년도'] == year]
    x = df_year.loc[df_year['구분'].isin(['감면', '국비']), '연인원'].sum()  # 보훈대상자 수 (연인원 기준)
    n = x + df_year.loc[df_year['구분'] == '일반', '연인원'].sum()           # 전체 이용자 수
    
    z_stat, p_value = sm.stats.proportions_ztest(count=x, nobs=n, value=p0, alternative='larger')
    ci_low, ci_upp = smp.proportion_confint(x, n, alpha=0.05, method='wilson')
    prop = x / n
    
    print(f"{year}년 ▶ 보훈대상자 비율: {prop:.3%} | Z = {z_stat:.3f}, p = {p_value:.4f}, CI = ({ci_low:.3%}, {ci_upp:.3%})")
    if p_value < alpha:
        print(f"  → ✅ 90%를 초과함 (통계적으로 유의)")
    else:
        print(f"  → ❌ 90%를 초과한다고 보기 어려움")
    
    total_x += x
    total_n += n

# 전체 연도 통합 검정
print("\n📈 전체 연도 통합 비율 검정 (기준: 90%)")
z_stat, p_value = sm.stats.proportions_ztest(count=total_x, nobs=total_n, value=p0, alternative='larger')
ci_low, ci_upp = smp.proportion_confint(total_x, total_n, alpha=0.05, method='wilson')
prop_total = total_x / total_n

print(f"전체 ▶ 보훈대상자 비율: {prop_total:.3%} | Z = {z_stat:.3f}, p = {p_value:.4f}, CI = ({ci_low:.3%}, {ci_upp:.3%})")
if p_value < alpha:
    print(f"  → ✅ 전체적으로 90%를 초과함 (통계적으로 유의)")
else:
    print(f"  → ❌ 전체적으로 90%를 초과한다고 보기 어려움")

# total_x, total_n = 0, 0

# print("\n📊 연도별 보훈대상자 비율 검정 결과 (기준: 90%)")
# print("-" * 80)

# # 연도별로 데이터 그룹화
# for year in sorted(data['년도'].unique()):
#     year_data = data[data['년도'] == year]
    
#     # 해당 연도의 전체 데이터 집계
#     total_감면 = year_data[year_data['구분'] == '감면']['연인원'].sum()
#     total_국비 = year_data[year_data['구분'] == '국비']['연인원'].sum()
#     total_일반 = year_data[year_data['구분'] == '일반']['연인원'].sum()
    
#     # 보훈대상자 수 (감면 + 국비)
#     x = total_감면 + total_국비
#     # 전체 수 (감면 + 국비 + 일반)
#     n = x + total_일반
    
#     # 비율 계산
#     prop = x / n if n > 0 else 0
    
#     # 비율 검정 수행
#     if n > 0 and x > 0:
#         try:
#             z_stat, p_value = sm.stats.proportions_ztest(count=x, nobs=n, value=p0, alternative='larger')
#             ci_low, ci_upp = smp.proportion_confint(x, n, alpha=0.05, method='wilson')
            
#             print(f"{year}년 ▶ 보훈대상자 비율: {prop:.3%} | Z = {z_stat:.3f}, p = {p_value:.4f}, CI = ({ci_low:.3%}, {ci_upp:.3%})")
#             if p_value < alpha:
#                 print(f"  → ✅ 90%를 초과함 (통계적으로 유의)")
#             else:
#                 print(f"  → ❌ 90%를 초과한다고 보기 어려움")
#         except Exception as e:
#             print(f"{year}년 ▶ 보훈대상자 비율: {prop:.3%} | 검정 오류: {e}")
#     else:
#         print(f"{year}년 ▶ 보훈대상자 비율: {prop:.3%} | 데이터 부족으로 검정 불가")
    
#     total_x += x
#     total_n += n

# # 전체 통합 검정
# print("\n📈 전체 연도 통합 비율 검정 (기준: 90%)")
# if total_n > 0 and total_x > 0:
#     try:
#         z_stat, p_value = sm.stats.proportions_ztest(count=total_x, nobs=total_n, value=p0, alternative='larger')
#         ci_low, ci_upp = smp.proportion_confint(total_x, total_n, alpha=0.05, method='wilson')
#         prop_total = total_x / total_n
        
#         print(f"전체 ▶ 보훈대상자 비율: {prop_total:.3%} | Z = {z_stat:.3f}, p = {p_value:.4f}, CI = ({ci_low:.3%}, {ci_upp:.3%})")
#         if p_value < alpha:
#             print(f"  → ✅ 전체적으로 90%를 초과함 (통계적으로 유의)")
#         else:
#             print(f"  → ❌ 전체적으로 90%를 초과한다고 보기 어려움")
#     except Exception as e:
#         print(f"전체 ▶ 검정 오류: {e}")
# else:
#     print("전체 ▶ 데이터 부족으로 검정 불가")

# # 지역별 분석 추가
# print("\n🌍 지역별 보훈대상자 비율 분석")
# print("-" * 80)

# for region in sorted(data['지역'].unique()):
#     region_data = data[data['지역'] == region]
    
#     # 해당 지역의 전체 데이터 집계
#     total_감면 = region_data[region_data['구분'] == '감면']['연인원'].sum()
#     total_국비 = region_data[region_data['구분'] == '국비']['연인원'].sum()
#     total_일반 = region_data[region_data['구분'] == '일반']['연인원'].sum()
    
#     # 보훈대상자 수 (감면 + 국비)
#     x = total_감면 + total_국비
#     # 전체 수 (감면 + 국비 + 일반)
#     n = x + total_일반
    
#     # 비율 계산
#     prop = x / n if n > 0 else 0
    
#     print(f"{region} ▶ 보훈대상자 비율: {prop:.3%} (감면: {total_감면:,}명, 국비: {total_국비:,}명, 일반: {total_일반:,}명)")

# # 구분별 전체 집계
# print("\n📋 구분별 전체 집계")
# print("-" * 80)
# total_감면 = data[data['구분'] == '감면']['연인원'].sum()
# total_국비 = data[data['구분'] == '국비']['연인원'].sum()
# total_일반 = data[data['구분'] == '일반']['연인원'].sum()

# print(f"감면: {total_감면:,}명")
# print(f"국비: {total_국비:,}명")
# print(f"일반: {total_일반:,}명")
# print(f"보훈대상자 총합: {total_감면 + total_국비:,}명")
# print(f"전체 총합: {total_감면 + total_국비 + total_일반:,}명")
# print(f"보훈대상자 비율: {(total_감면 + total_국비) / (total_감면 + total_국비 + total_일반):.3%}")