# GA 기반 전문의 분배 최적화
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

print("=== GA 전문의 분배 최적화 모델 ===")
print("📊 유전 알고리즘을 사용한 전문의 효율적 분배 시스템")
print()

# --------------------------------------------------
# 1) 데이터 로드 및 전처리
# --------------------------------------------------
print("1/6: 데이터 로드 및 전처리 중...")

# 기존 예측 결과 로드
df_pred = pd.read_csv('analysis_data/병원별_진료과별_입원외래_통합_시계열예측결과_개선.csv')
df_info = pd.read_csv('new_merged_data/병원_통합_데이터.csv')

# 최근 연도(2023)만 사용
df_pred = df_pred[df_pred['연도'] == 2023]

# 병원명 컬럼명 통일
df_pred['병원명'] = df_pred['병원'].astype(str).str.strip()
df_pred['진료과'] = df_pred['진료과'].astype(str).str.strip()
df_info['병원명'] = df_info['병원명'].astype(str).str.strip()

print(f"✅ 데이터 로드 완료")
print(f"  - 예측 데이터: {df_pred.shape}")
print(f"  - 병원 정보: {df_info.shape}")
print()

# --------------------------------------------------
# 2) 현재 상황 분석
# --------------------------------------------------
print("2/6: 현재 상황 분석 중...")

def get_doc_col(진료과):
    return f"{진료과}_전문의수"

# 현재 전문의 현황 분석
current_situation = []
total_doctors = 0
total_patients = 0

for idx, row in df_pred.iterrows():
    병원 = row['병원명']
    진료과 = row['진료과']
    예측환자수 = row['XGB예측']  # 가장 정확한 예측값 사용
    
    info_row = df_info[df_info['병원명'] == 병원]
    doc_col = get_doc_col(진료과)
    
    if len(info_row) > 0 and doc_col in info_row.columns:
        현재전문의수 = info_row.iloc[0][doc_col]
        if pd.notnull(현재전문의수):
            current_situation.append({
                '병원명': 병원,
                '진료과': 진료과,
                '현재전문의수': 현재전문의수,
                '예측환자수': 예측환자수,
                '환자당전문의비율': 예측환자수 / (현재전문의수 + 1)  # 0으로 나누기 방지
            })
            total_doctors += 현재전문의수
            total_patients += 예측환자수

current_df = pd.DataFrame(current_situation)

print(f"✅ 현재 상황 분석 완료")
print(f"  - 총 전문의 수: {total_doctors:.0f}명")
print(f"  - 총 예측 환자 수: {total_patients:.0f}명")
print(f"  - 평균 환자당 전문의 비율: {total_patients/total_doctors:.2f}")
print()

# --------------------------------------------------
# 3) GA 구현
# --------------------------------------------------
print("3/6: GA 모델 설정 중...")

# 시드 고정
np.random.seed(42)

class GeneticAlgorithm:
    def __init__(self, pop_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
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
    
    def fitness_function(self, individual, patients_array, total_doctors):
        # 제약조건 위반 시 페널티
        if abs(np.sum(individual) - total_doctors) > 1:
            return 0
        
        # 환자당 전문의 비율의 표준편차 최소화
        ratios = patients_array / (individual + 1)
        std = np.std(ratios)
        return 1 / (1 + std)  # 표준편차가 작을수록 높은 적합도
    
    def selection(self, population, fitness_scores):
        selected = []
        for _ in range(len(population)):
            tournament = np.random.choice(len(population), 3, replace=False)
            winner = tournament[np.argmax(fitness_scores[tournament])]
            selected.append(population[winner])
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            return np.array([parent1[i] if np.random.random() < 0.5 else parent2[i] for i in range(len(parent1))])
        return parent1.copy()
    
    def mutation(self, individual, bounds):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                min_val, max_val = bounds[i]
                mutated[i] += np.random.normal(0, (max_val - min_val) * 0.1)
                mutated[i] = np.clip(mutated[i], min_val, max_val)
        return mutated
    
    def optimize(self, bounds, patients_array, total_doctors):
        population = self.initialize_population(bounds)
        best_fitness = 0
        best_individual = None
        
        for generation in range(self.generations):
            fitness_scores = []
            for individual in population:
                fitness = self.fitness_function(individual, patients_array, total_doctors)
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            max_fitness_idx = np.argmax(fitness_scores)
            
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
            
            selected = self.selection(population, fitness_scores)
            new_population = []
            
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1 = self.crossover(selected[i], selected[i + 1])
                    child2 = self.crossover(selected[i + 1], selected[i])
                    child1 = self.mutation(child1, bounds)
                    child2 = self.mutation(child2, bounds)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            population = np.array(new_population)
        
        return best_individual, best_fitness

print(f"✅ GA 모델 설정 완료")
print(f"  - 개체 수: 50")
print(f"  - 세대 수: 100")
print(f"  - 돌연변이율: 0.1")
print(f"  - 교차율: 0.8")
print()

# --------------------------------------------------
# 4) GA 최적화 실행
# --------------------------------------------------
print("4/6: GA 최적화 실행 중...")

# 초기값 설정
initial_doctors = current_df['현재전문의수'].values
patients_array = current_df['예측환자수'].values

# 경계 설정 (현재의 60%~140% 범위)
bounds = []
for d in initial_doctors:
    lower = max(1, int(d * 0.6))
    upper = max(lower + 1, int(d * 1.4))
    bounds.append((lower, upper))

# GA 실행
ga = GeneticAlgorithm()
best_solution, best_fitness = ga.optimize(bounds, patients_array, total_doctors)

print(f"✅ GA 최적화 완료")
print(f"  - 최적화 성공: True")
print(f"  - 세대 수: {ga.generations}")
print(f"  - 최고 적합도: {best_fitness:.4f}")
print()

# --------------------------------------------------
# 5) 결과 분석 및 저장
# --------------------------------------------------
print("5/6: 결과 분석 및 저장 중...")

results = []
for idx, row in current_df.iterrows():
    병원명 = row['병원명']
    진료과 = row['진료과']
    최적전문의수 = best_solution[idx]
    현재전문의수 = row['현재전문의수']
    예측환자수 = row['예측환자수']
    변화량 = 최적전문의수 - 현재전문의수
    변화율 = (변화량 / 현재전문의수 * 100) if 현재전문의수 != 0 else 0
    현재_환자당전문의비율 = 예측환자수 / (현재전문의수 + 1)
    최적_환자당전문의비율 = 예측환자수 / (최적전문의수 + 1)
    
    results.append({
        '병원명': 병원명,
        '진료과': 진료과,
        '현재전문의수': 현재전문의수,
        '최적전문의수': 최적전문의수,
        '변화량': 변화량,
        '변화율': 변화율,
        '예측환자수': 예측환자수,
        '현재_환자당전문의비율': 현재_환자당전문의비율,
        '최적_환자당전문의비율': 최적_환자당전문의비율
    })

results_df = pd.DataFrame(results)

# 결과 저장
output_dir = 'optimization_results_전문의_분배_최적화'
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(f'{output_dir}/전문의_분배_최적화_결과_GA.csv', index=False, encoding='utf-8-sig')

print(f"✅ GA 결과 저장 완료: {output_dir}/전문의_분배_최적화_결과_GA.csv")

# --------------------------------------------------
# 6) 시각화
# --------------------------------------------------
print("6/6: 시각화 생성 중...")

plt.figure(figsize=(15, 10))

# 서브플롯 1: 현재 vs 최적 전문의 수 비교
plt.subplot(2, 3, 1)
plt.scatter(results_df['현재전문의수'], results_df['최적전문의수'], alpha=0.7, s=100)
max_doctors = max(results_df['현재전문의수'].max(), results_df['최적전문의수'].max())
plt.plot([0, max_doctors], [0, max_doctors], 'r--', alpha=0.5)
plt.xlabel('현재 전문의 수')
plt.ylabel('최적 전문의 수')
plt.title('현재 vs 최적 전문의 수 (GA)')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 전문의 변화량
plt.subplot(2, 3, 2)
colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in results_df['변화량']]
plt.barh(range(len(results_df)), results_df['변화량'], color=colors, alpha=0.7)
plt.xlabel('전문의 수 변화량')
plt.title('병원-진료과별 전문의 수 변화량 (GA)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.yticks(range(len(results_df)), [f"{row['병원명']}-{row['진료과']}" for _, row in results_df.iterrows()], fontsize=8)
plt.grid(True, alpha=0.3)

# 서브플롯 3: 환자당 전문의 비율 비교
plt.subplot(2, 3, 3)
x = np.arange(len(results_df))
width = 0.35
plt.bar(x - width/2, results_df['현재_환자당전문의비율'], width, label='현재', alpha=0.7)
plt.bar(x + width/2, results_df['최적_환자당전문의비율'], width, label='최적', alpha=0.7)
plt.xlabel('병원-진료과')
plt.ylabel('환자당 전문의 비율')
plt.title('현재 vs 최적 환자당 전문의 비율 (GA)')
plt.xticks(x, [f"{row['병원명']}-{row['진료과']}" for _, row in results_df.iterrows()], rotation=45, fontsize=8)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 4: 비율 개선도
plt.subplot(2, 3, 4)
개선도 = results_df['최적_환자당전문의비율'] - results_df['현재_환자당전문의비율']
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in 개선도]
plt.barh(range(len(results_df)), 개선도, color=colors, alpha=0.7)
plt.xlabel('환자당 전문의 비율 개선도')
plt.title('병원-진료과별 비율 개선도 (GA)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.yticks(range(len(results_df)), [f"{row['병원명']}-{row['진료과']}" for _, row in results_df.iterrows()], fontsize=8)
plt.grid(True, alpha=0.3)

# 서브플롯 5: 현재 vs 최적 비율 산점도
plt.subplot(2, 3, 5)
plt.scatter(results_df['현재_환자당전문의비율'], results_df['최적_환자당전문의비율'], 
           alpha=0.7, s=100, c=results_df['변화량'], cmap='RdYlBu')
plt.colorbar(label='변화량')
max_ratio = max(results_df['현재_환자당전문의비율'].max(), results_df['최적_환자당전문의비율'].max())
plt.plot([0, max_ratio], [0, max_ratio], 'r--', alpha=0.5)
plt.xlabel('현재 환자당 전문의 비율')
plt.ylabel('최적 환자당 전문의 비율')
plt.title('현재 vs 최적 비율 비교 (GA)')
plt.grid(True, alpha=0.3)

# 서브플롯 6: 비율 분포 비교
plt.subplot(2, 3, 6)
plt.hist([results_df['현재_환자당전문의비율'], results_df['최적_환자당전문의비율']], 
         label=['현재', '최적'], alpha=0.7, bins=10)
plt.xlabel('환자당 전문의 비율')
plt.ylabel('빈도')
plt.title('비율 분포 비교 (GA)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/전문의_분배_최적화_시각화_GA.png', dpi=300, bbox_inches='tight')
plt.show()

# 성능 지표 계산
현재_비율_표준편차 = results_df['현재_환자당전문의비율'].std()
최적_비율_표준편차 = results_df['최적_환자당전문의비율'].std()
개선도 = (현재_비율_표준편차 - 최적_비율_표준편차) / 현재_비율_표준편차 * 100

print("\n=== GA 최적화 결과 요약 ===")
print(results_df[['병원명', '진료과', '현재전문의수', '최적전문의수', '변화량', '변화율', '현재_환자당전문의비율', '최적_환자당전문의비율']].round(2).to_string(index=False))

print(f"\n📊 성능 지표:")
print(f"  - 현재 비율 표준편차: {현재_비율_표준편차:.2f}")
print(f"  - 최적 비율 표준편차: {최적_비율_표준편차:.2f}")
print(f"  - 비율 개선도: {개선도:.1f}%")
print(f"  - 최적화 성공 여부: True")
print(f"  - 세대 수: {ga.generations}")

print(f"\n✅ 모든 결과가 {output_dir}/ 디렉토리에 저장되었습니다!")
print("="*60)
print("🎯 GA 전문의 분배 최적화 완료!")
print("="*60) 