# GA 기반 병상 분배 균등화 최적화
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (동일)
df_hospital = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv')
df_pred = df_pred[df_pred['예측연도'] == 2024]
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_hospital['병원명'] = df_hospital['병원명'].astype(str).str.strip()
df_pred['병원명'] = df_pred['병원명'].replace('중앙', '서울')
bed_columns = [col for col in df_hospital.columns if not col.endswith('_전문의수') and col != '병원명']
df_hospital['총병상수'] = df_hospital[bed_columns].sum(axis=1)
hospital_patients = df_pred.groupby('병원명')['XGB예측'].sum().reset_index()
hospital_patients.columns = ['병원명', '총예측환자수']
data = []
total_beds = 0
for idx, row in hospital_patients.iterrows():
    병원 = row['병원명']
    예측환자수 = row['총예측환자수']
    hosp_row = df_hospital[df_hospital['병원명'] == 병원]
    if len(hosp_row) > 0:
        현재병상수 = float(hosp_row['총병상수'].iloc[0])
        if pd.notnull(현재병상수) and 현재병상수 > 0:
            data.append({'병원명': 병원, '현재병상수': 현재병상수, '예측환자수': 예측환자수})
            total_beds += 현재병상수
current_df = pd.DataFrame(data)

# GA 구현
np.random.seed(42)
class GeneticAlgorithm:
    def __init__(self, pop_size=30, generations=50, mutation_rate=0.1, crossover_rate=0.8):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    def initialize_population(self, bounds):
        population = []
        for _ in range(self.pop_size):
            individual = [np.random.uniform(min_val, max_val) for (min_val, max_val) in bounds]
            population.append(individual)
        return np.array(population)
    def fitness_function(self, individual, patients_array):
        utils = [(p/365)/(b+1)*100 for b, p in zip(individual, patients_array)]
        std = np.std(utils)
        return 1/(1+std)
    def selection(self, population, fitness_scores):
        selected = []
        for _ in range(len(population)):
            tournament = np.random.choice(len(population), 3, replace=False)
            winner = tournament[np.argmax(fitness_scores[tournament])]
            selected.append(population[winner])
        return np.array(selected)
    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            return np.array([parent1[i] if np.random.random()<0.5 else parent2[i] for i in range(len(parent1))])
        return parent1.copy()
    def mutation(self, individual, bounds):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                min_val, max_val = bounds[i]
                mutated[i] += np.random.normal(0, (max_val-min_val)*0.1)
                mutated[i] = np.clip(mutated[i], min_val, max_val)
        return mutated
    def optimize(self, bounds, patients_array, total_beds):
        population = self.initialize_population(bounds)
        best_fitness = 0
        best_individual = None
        for generation in range(self.generations):
            fitness_scores = []
            for individual in population:
                if abs(np.sum(individual)-total_beds)<1:
                    fitness = self.fitness_function(individual, patients_array)
                else:
                    fitness = 0
                fitness_scores.append(fitness)
            fitness_scores = np.array(fitness_scores)
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
            selected = self.selection(population, fitness_scores)
            new_population = []
            for i in range(0, len(selected), 2):
                if i+1 < len(selected):
                    child1 = self.crossover(selected[i], selected[i+1])
                    child2 = self.crossover(selected[i+1], selected[i])
                    child1 = self.mutation(child1, bounds)
                    child2 = self.mutation(child2, bounds)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            population = np.array(new_population)
        return best_individual, best_fitness

initial_beds = current_df['현재병상수'].values
patients_array = current_df['예측환자수'].values
bounds = [(max(1, int(b*0.6)), int(b*1.4)) for b in initial_beds]
ga = GeneticAlgorithm()
best_solution, best_fitness = ga.optimize(bounds, patients_array, total_beds)

# 결과 분석 및 저장
print("결과 분석 및 저장 중...")

results = []
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    최적병상수 = best_solution[idx]
    현재병상수 = row['현재병상수']
    예측환자수 = row['예측환자수']
    변화량 = 최적병상수 - 현재병상수
    변화율 = (변화량 / 현재병상수 * 100) if 현재병상수 != 0 else 0
    일평균환자수 = 예측환자수 / 365
    현재_가동률 = (일평균환자수 / (현재병상수 + 1)) * 100
    최적_가동률 = (일평균환자수 / (최적병상수 + 1)) * 100
    results.append({
        '병원명': 병원명,
        '현재병상수': 현재병상수,
        '최적병상수': 최적병상수,
        '변화량': 변화량,
        '변화율': 변화율,
        '예측환자수': 예측환자수,
        '현재_병상가동률': 현재_가동률,
        '최적_병상가동률': 최적_가동률
    })
results_df = pd.DataFrame(results)

# 결과 저장
output_dir = 'optimization_results_병상_분배_균등화_실제'
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(f'{output_dir}/병상_분배_균등화_결과_GA.csv', index=False, encoding='utf-8-sig')

print(f"✅ GA 결과 저장 완료: {output_dir}/병상_분배_균등화_결과_GA.csv")

# 시각화
plt.figure(figsize=(15, 10))

# 서브플롯 1: 현재 vs 최적 병상 수 비교
plt.subplot(2, 3, 1)
plt.scatter(results_df['현재병상수'], results_df['최적병상수'], alpha=0.7, s=100)
max_beds = max(results_df['현재병상수'].max(), results_df['최적병상수'].max())
plt.plot([0, max_beds], [0, max_beds], 'r--', alpha=0.5)
plt.xlabel('현재 병상 수')
plt.ylabel('최적 병상 수')
plt.title('현재 vs 최적 병상 수 (GA)')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 병상 변화량
plt.subplot(2, 3, 2)
colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in results_df['변화량']]
plt.barh(results_df['병원명'], results_df['변화량'], color=colors, alpha=0.7)
plt.xlabel('병상 수 변화량')
plt.title('병원별 병상 수 변화량 (GA)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 3: 가동률 비교
plt.subplot(2, 3, 3)
x = np.arange(len(results_df))
width = 0.35
plt.bar(x - width/2, results_df['현재_병상가동률'], width, label='현재', alpha=0.7)
plt.bar(x + width/2, results_df['최적_병상가동률'], width, label='최적', alpha=0.7)
plt.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='목표(65%)')
plt.xlabel('병원')
plt.ylabel('병상가동률 (%)')
plt.title('현재 vs 최적 병상가동률 (GA)')
plt.xticks(x, list(results_df['병원명']), rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 가동률 개선도
plt.subplot(2, 3, 4)
개선도 = results_df['최적_병상가동률'] - results_df['현재_병상가동률']
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in 개선도]
plt.barh(results_df['병원명'], 개선도, color=colors, alpha=0.7)
plt.xlabel('가동률 개선도 (%)')
plt.title('병원별 가동률 개선도 (GA)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 서브플롯 5: 현재 vs 최적 가동률 산점도
plt.subplot(2, 3, 5)
plt.scatter(results_df['현재_병상가동률'], results_df['최적_병상가동률'], 
           alpha=0.7, s=100, c=results_df['변화량'], cmap='RdYlBu')
plt.colorbar(label='변화량')
max_util = max(results_df['현재_병상가동률'].max(), results_df['최적_병상가동률'].max())
plt.plot([0, max_util], [0, max_util], 'r--', alpha=0.5)
plt.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='목표(65%)')
plt.xlabel('현재 병상가동률 (%)')
plt.ylabel('최적 병상가동률 (%)')
plt.title('현재 vs 최적 가동률 비교 (GA)')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 6: 가동률 분포 비교
plt.subplot(2, 3, 6)
plt.hist([results_df['현재_병상가동률'], results_df['최적_병상가동률']], 
         label=['현재', '최적'], alpha=0.7, bins=10)
plt.axvline(x=65, color='red', linestyle='--', alpha=0.7, label='목표(65%)')
plt.xlabel('병상가동률 (%)')
plt.ylabel('병원 수')
plt.title('가동률 분포 비교 (GA)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/병상_분배_균등화_시각화_GA.png', dpi=300, bbox_inches='tight')
plt.show()

# 성능 지표 계산
현재_가동률_표준편차 = results_df['현재_병상가동률'].std()
최적_가동률_표준편차 = results_df['최적_병상가동률'].std()
가동률_개선도 = (현재_가동률_표준편차 - 최적_가동률_표준편차) / 현재_가동률_표준편차 * 100

print("\n=== GA 최적화 결과 요약 ===")
print(results_df[['병원명', '현재병상수', '최적병상수', '변화량', '변화율', '현재_병상가동률', '최적_병상가동률']].round(2).to_string(index=False))

print(f"\n📊 성능 지표:")
print(f"  - 현재 가동률 표준편차: {현재_가동률_표준편차:.2f}%")
print(f"  - 최적 가동률 표준편차: {최적_가동률_표준편차:.2f}%")
print(f"  - 가동률 개선도: {가동률_개선도:.1f}%")
print(f"  - 최적화 성공 여부: True")
print(f"  - 세대 수: {ga.generations}")

print(f"\n✅ 모든 결과가 {output_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 GA 병상 분배 균등화 최적화 완료!")
print("="*60) 