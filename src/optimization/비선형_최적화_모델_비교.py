import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from typing import Dict, List, Tuple, Any

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== 비선형 최적화 모델 비교 분석 ===")
print("📊 다양한 최적화 기법을 통한 의료자원 분배 최적화")
print()

# --------------------------------------------------
# 1) 현재 문제점 분석
# --------------------------------------------------
print("1/6: 현재 문제점 분석 중...")

print("🔍 현재 최적화 모델들의 문제점:")
print("  ❌ PuLP (선형계획법): 비선형 제약조건 처리 불가")
print("  ❌ 가동률 = 환자수/병상수 형태의 비선형 관계")
print("  ❌ 목적함수에서 비선형 최적화 불가")
print("  ❌ 복잡한 제약조건 표현의 한계")
print()

print("✅ 해결 가능한 대안 최적화 기법들:")
print("  1. SciPy Optimize (비선형 최적화)")
print("  2. Genetic Algorithm (유전 알고리즘)")
print("  3. Particle Swarm Optimization (PSO)")
print("  4. Simulated Annealing (시뮬레이티드 어닐링)")
print("  5. Bayesian Optimization (베이지안 최적화)")
print()

# --------------------------------------------------
# 2) 데이터 로드 및 전처리
# --------------------------------------------------
print("2/6: 데이터 로드 및 전처리 중...")

# 병원 통합 데이터 로드
df_hospital = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')

# 입원 예측 데이터 로드
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv')
df_pred = df_pred[df_pred['예측연도'] == 2024]

# 병원명 매칭
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_hospital['병원명'] = df_hospital['병원명'].astype(str).str.strip()
df_pred['병원명'] = df_pred['병원명'].replace('중앙', '서울')

# 병상 수 계산
bed_columns = [col for col in df_hospital.columns if not col.endswith('_전문의수') and col != '병원명']
df_hospital['총병상수'] = df_hospital[bed_columns].sum(axis=1)

# 병원별 예측 환자 수 집계
hospital_patients = df_pred.groupby('병원명')['XGB예측'].sum().reset_index()
hospital_patients.columns = ['병원명', '총예측환자수']

# 현재 상황 데이터 생성
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

current_df = pd.DataFrame(data)

print(f"✅ 데이터 로드 완료: 병원 {len(current_df)}개")
print(f"  - 총 병상 수: {total_beds:.0f}개")
print(f"  - 총 예측 환자 수: {total_patients:.0f}명")
print()

# --------------------------------------------------
# 3) 비선형 최적화 모델들 구현
# --------------------------------------------------
print("3/6: 비선형 최적화 모델들 구현 중...")

# 목표 가동률
target_utilization = 65.0

# 가동률 계산 함수
def calculate_utilization(beds, patients):
    """병상가동률 계산"""
    daily_patients = patients / 365
    return (daily_patients / (beds + 1)) * 100

# 목적 함수: 가동률 표준편차 최소화
def objective_function(beds_array, patients_array, target_util):
    """목적 함수: 가동률 표준편차 최소화"""
    utilizations = []
    for beds, patients in zip(beds_array, patients_array):
        util = calculate_utilization(beds, patients)
        utilizations.append(util)
    
    return np.std(utilizations)

# 제약조건 함수들
def constraint_total_beds(beds_array, total_beds):
    """총 병상 수 제약조건"""
    return np.sum(beds_array) - total_beds

def constraint_min_beds(beds_array, current_beds_array):
    """최소 병상 수 제약조건 (현재의 60% 이상)"""
    min_beds = current_beds_array * 0.6
    return beds_array - min_beds

def constraint_max_beds(beds_array, current_beds_array):
    """최대 병상 수 제약조건 (현재의 140% 이하)"""
    max_beds = current_beds_array * 1.4
    return max_beds - beds_array

def constraint_min_utilization(beds_array, patients_array):
    """최소 가동률 제약조건 (40% 이상)"""
    utilizations = []
    for beds, patients in zip(beds_array, patients_array):
        util = calculate_utilization(beds, patients)
        utilizations.append(util)
    return np.array(utilizations) - 40

def constraint_max_utilization(beds_array, patients_array):
    """최대 가동률 제약조건 (90% 이하)"""
    utilizations = []
    for beds, patients in zip(beds_array, patients_array):
        util = calculate_utilization(beds, patients)
        utilizations.append(util)
    return 90 - np.array(utilizations)

print("✅ 비선형 최적화 함수들 정의 완료")
print()

# --------------------------------------------------
# 4) SciPy Optimize를 사용한 비선형 최적화
# --------------------------------------------------
print("4/6: SciPy Optimize 비선형 최적화 실행 중...")

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.optimize import NonlinearConstraint, Bounds
    
    # 초기값 설정
    initial_beds = current_df['현재병상수'].values
    patients_array = current_df['예측환자수'].values
    current_beds_array = current_df['현재병상수'].values
    
    # 경계 설정
    bounds = []
    for current_beds in current_beds_array:
        min_beds = max(1, int(current_beds * 0.6))
        max_beds = int(current_beds * 1.4)
        bounds.append((min_beds, max_beds))
    
    # 제약조건 설정
    constraints = [
        # 총 병상 수 제약
        NonlinearConstraint(
            lambda x: constraint_total_beds(x, total_beds),
            lb=0, ub=0
        ),
        # 최소 가동률 제약
        NonlinearConstraint(
            lambda x: constraint_min_utilization(x, patients_array),
            lb=0, ub=np.inf
        ),
        # 최대 가동률 제약
        NonlinearConstraint(
            lambda x: constraint_max_utilization(x, patients_array),
            lb=0, ub=np.inf
        )
    ]
    
    # SLSQP 방법으로 최적화
    result_scipy = minimize(
        lambda x: objective_function(x, patients_array, target_utilization),
        initial_beds,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    print(f"✅ SciPy SLSQP 최적화 완료!")
    print(f"  - 성공 여부: {result_scipy.success}")
    print(f"  - 목적 함수 값: {result_scipy.fun:.4f}")
    print(f"  - 반복 횟수: {result_scipy.nit}")
    
    # Differential Evolution으로도 시도
    result_de = differential_evolution(
        lambda x: objective_function(x, patients_array, target_utilization),
        bounds,
        constraints=constraints,
        maxiter=100,
        popsize=15,
        seed=42,
        workers=1  # 병렬 처리 비활성화로 재현성 확보
    )
    
    print(f"✅ SciPy Differential Evolution 완료!")
    print(f"  - 성공 여부: {result_de.success}")
    print(f"  - 목적 함수 값: {result_de.fun:.4f}")
    print(f"  - 반복 횟수: {result_de.nit}")
    
except ImportError:
    print("⚠️ SciPy가 설치되지 않았습니다.")
    result_scipy = None
    result_de = None

print()

# --------------------------------------------------
# 5) 유전 알고리즘 구현
# --------------------------------------------------
print("5/6: 유전 알고리즘 구현 중...")

class GeneticAlgorithm:
    def __init__(self, pop_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def initialize_population(self, bounds):
        """초기 개체군 생성"""
        population = []
        for _ in range(self.pop_size):
            individual = []
            for (min_val, max_val) in bounds:
                individual.append(np.random.uniform(min_val, max_val))
            population.append(individual)
        return np.array(population)
    
    def fitness_function(self, individual, patients_array, target_util):
        """적합도 함수 (가동률 표준편차의 역수)"""
        try:
            std = objective_function(individual, patients_array, target_util)
            return 1 / (1 + std)  # 표준편차가 작을수록 높은 적합도
        except:
            return 0
    
    def selection(self, population, fitness_scores):
        """토너먼트 선택"""
        selected = []
        for _ in range(len(population)):
            tournament = np.random.choice(len(population), 3, replace=False)
            winner = tournament[np.argmax(fitness_scores[tournament])]
            selected.append(population[winner])
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        """균등 교차"""
        if np.random.random() < self.crossover_rate:
            child = []
            for i in range(len(parent1)):
                if np.random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            return np.array(child)
        return parent1.copy()
    
    def mutation(self, individual, bounds):
        """가우시안 돌연변이"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                min_val, max_val = bounds[i]
                mutated[i] += np.random.normal(0, (max_val - min_val) * 0.1)
                mutated[i] = np.clip(mutated[i], min_val, max_val)
        return mutated
    
    def optimize(self, bounds, patients_array, target_util, total_beds):
        """유전 알고리즘 최적화"""
        population = self.initialize_population(bounds)
        best_fitness = 0
        best_individual = None
        
        for generation in range(self.generations):
            # 적합도 계산
            fitness_scores = []
            for individual in population:
                # 제약조건 검사
                if abs(np.sum(individual) - total_beds) < 1:  # 총 병상 수 제약
                    fitness = self.fitness_function(individual, patients_array, target_util)
                else:
                    fitness = 0  # 제약조건 위반 시 낮은 적합도
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            
            # 최고 적합도 개체 저장
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
            
            # 선택
            selected = self.selection(population, fitness_scores)
            
            # 새로운 개체군 생성
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1 = self.crossover(selected[i], selected[i+1])
                    child2 = self.crossover(selected[i+1], selected[i])
                    
                    child1 = self.mutation(child1, bounds)
                    child2 = self.mutation(child2, bounds)
                    
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            population = np.array(new_population)
            
            if generation % 20 == 0:
                print(f"  세대 {generation}: 최고 적합도 = {best_fitness:.4f}")
        
        return best_individual, best_fitness

# 유전 알고리즘 실행 (재현 가능하도록 시드 설정)
np.random.seed(42)  # 재현 가능한 결과를 위한 시드 설정
ga = GeneticAlgorithm(pop_size=30, generations=50)
result_ga = ga.optimize(bounds, patients_array, target_utilization, total_beds)

print(f"✅ 유전 알고리즘 최적화 완료!")
print(f"  - 최고 적합도: {result_ga[1]:.4f}")
print()

# --------------------------------------------------
# 6) 결과 비교 및 분석
# --------------------------------------------------
print("6/6: 결과 비교 및 분석 중...")

# 결과 저장
results_comparison = {
    '현재상태': {
        '병상수': current_df['현재병상수'].values,
        '가동률': [calculate_utilization(beds, patients) for beds, patients in zip(current_df['현재병상수'], current_df['예측환자수'])],
        '표준편차': np.std([calculate_utilization(beds, patients) for beds, patients in zip(current_df['현재병상수'], current_df['예측환자수'])])
    }
}

if result_scipy is not None and result_scipy.success:
    results_comparison['SciPy_SLSQP'] = {
        '병상수': result_scipy.x,
        '가동률': [calculate_utilization(beds, patients) for beds, patients in zip(result_scipy.x, patients_array)],
        '표준편차': result_scipy.fun
    }

if result_de is not None and result_de.success:
    results_comparison['SciPy_DE'] = {
        '병상수': result_de.x,
        '가동률': [calculate_utilization(beds, patients) for beds, patients in zip(result_de.x, patients_array)],
        '표준편차': result_de.fun
    }

if result_ga[0] is not None:
    results_comparison['Genetic_Algorithm'] = {
        '병상수': result_ga[0],
        '가동률': [calculate_utilization(beds, patients) for beds, patients in zip(result_ga[0], patients_array)],
        '표준편차': objective_function(result_ga[0], patients_array, target_utilization)
    }

# 결과 저장
output_dir = 'optimization_results_비선형_최적화_비교'
os.makedirs(output_dir, exist_ok=True)

# 결과 데이터프레임 생성
comparison_data = []
for method, data in results_comparison.items():
    for i, hospital in enumerate(current_df['병원명']):
        comparison_data.append({
            '최적화방법': method,
            '병원명': hospital,
            '병상수': data['병상수'][i],
            '가동률': data['가동률'][i],
            '예측환자수': current_df.iloc[i]['예측환자수']
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(f'{output_dir}/비선형_최적화_비교_결과.csv', index=False, encoding='utf-8-sig')

# 요약 통계
summary_stats = {}
for method, data in results_comparison.items():
    summary_stats[method] = {
        '가동률_표준편차': data['표준편차'],
        '평균_가동률': np.mean(data['가동률']),
        '최소_가동률': np.min(data['가동률']),
        '최대_가동률': np.max(data['가동률'])
    }

with open(f'{output_dir}/비선형_최적화_비교_요약.json', 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, ensure_ascii=False, indent=2)

# 시각화
plt.figure(figsize=(15, 10))

# 서브플롯 1: 가동률 비교
plt.subplot(2, 3, 1)
methods = list(results_comparison.keys())
utilizations = [results_comparison[method]['가동률'] for method in methods]
plt.boxplot(utilizations, labels=methods)
plt.ylabel('병상가동률 (%)')
plt.title('최적화 방법별 가동률 분포')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 서브플롯 2: 표준편차 비교
plt.subplot(2, 3, 2)
stds = [results_comparison[method]['표준편차'] for method in methods]
plt.bar(methods, stds, alpha=0.7)
plt.ylabel('가동률 표준편차')
plt.title('최적화 방법별 가동률 표준편차')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 서브플롯 3: 병상 수 변화량
plt.subplot(2, 3, 3)
current_beds = results_comparison['현재상태']['병상수']
for method in methods[1:]:  # 현재상태 제외
    optimal_beds = results_comparison[method]['병상수']
    changes = optimal_beds - current_beds
    plt.barh(current_df['병원명'], changes, alpha=0.7, label=method)
plt.xlabel('병상 수 변화량')
plt.title('병상 수 변화량 비교')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 가동률 개선도
plt.subplot(2, 3, 4)
current_std = results_comparison['현재상태']['표준편차']
improvements = []
for method in methods[1:]:
    optimal_std = results_comparison[method]['표준편차']
    improvement = (current_std - optimal_std) / current_std * 100
    improvements.append(improvement)
plt.bar(methods[1:], improvements, alpha=0.7, color='green')
plt.ylabel('개선도 (%)')
plt.title('가동률 균등화 개선도')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 서브플롯 5: 병원별 가동률 비교 (최고 성능 방법)
plt.subplot(2, 3, 5)
best_method = min(methods[1:], key=lambda x: results_comparison[x]['표준편차'])
current_utils = results_comparison['현재상태']['가동률']
optimal_utils = results_comparison[best_method]['가동률']

x = np.arange(len(current_df))
width = 0.35
plt.bar(x - width/2, current_utils, width, label='현재', alpha=0.7)
plt.bar(x + width/2, optimal_utils, width, label=f'최적({best_method})', alpha=0.7)
plt.axhline(y=target_utilization, color='red', linestyle='--', alpha=0.7, label=f'목표({target_utilization}%)')
plt.xlabel('병원')
plt.ylabel('병상가동률 (%)')
plt.title(f'현재 vs 최적 가동률 ({best_method})')
plt.xticks(x, current_df['병원명'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 6: 최적화 방법별 성능 비교
plt.subplot(2, 3, 6)
metrics = ['가동률_표준편차', '평균_가동률', '최대_가동률']
x_pos = np.arange(len(methods))
width = 0.25

for i, metric in enumerate(metrics):
    values = [summary_stats[method][metric] for method in methods]
    plt.bar(x_pos + i*width, values, width, label=metric, alpha=0.7)

plt.xlabel('최적화 방법')
plt.ylabel('값')
plt.title('최적화 방법별 성능 지표')
plt.xticks(x_pos + width, methods, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/비선형_최적화_비교_시각화.png', dpi=300, bbox_inches='tight')
plt.show()

# 결과 출력
print("=== 비선형 최적화 방법 비교 결과 ===")
print(f"📊 현재 상태 가동률 표준편차: {results_comparison['현재상태']['표준편차']:.2f}%")

for method in methods[1:]:
    std = results_comparison[method]['표준편차']
    improvement = (results_comparison['현재상태']['표준편차'] - std) / results_comparison['현재상태']['표준편차'] * 100
    print(f"📊 {method}: 표준편차 {std:.2f}% (개선도: {improvement:.1f}%)")

print(f"\n🏆 최고 성능 방법: {best_method}")
print(f"✅ 모든 결과가 {output_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 비선형 최적화 모델 비교 완료!")
print("="*60) 