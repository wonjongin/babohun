import simpy
import pandas as pd
import numpy as np
import random
import os

# ---------------------------
# 1. 미래 예측 환자수 데이터 불러오기
# ---------------------------
total_df = pd.read_csv('analysis_data/병원별_진료과별_미래3년_예측결과.csv')
inpatient_df = pd.read_csv('analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv')

# ---------------------------
# 2. 병상/전문의 데이터 불러오기
# ---------------------------
hospital_data = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')
bed_columns = [col for col in hospital_data.columns if any(x in col for x in ['실', '병실', '입원실', '중환자실'])]
doc_columns = [col for col in hospital_data.columns if '전문의수' in col]

beds_capacity = {
    hosp: int(round(hospital_data[hospital_data['병원명'] == hosp][bed_columns].sum(axis=1).values[0]))
    for hosp in hospital_data['병원명'].unique()
}
total_specialists = int(round(hospital_data[doc_columns].sum().sum()))

# ---------------------------
# 3. 병원별 연도별 환자수(입원/외래) 계산 함수
# ---------------------------
def get_hospital_patient_info(target_year=2025, target_col='XGB예측'):
    # 전체 환자수
    total_patients = (
        total_df[total_df['예측연도'] == target_year]
        .groupby('병원')[target_col]
        .sum()
        .to_dict()
    )
    # 입원환자수
    inpatient_patients = (
        inpatient_df[inpatient_df['예측연도'] == target_year]
        .groupby('병원')[target_col]
        .sum()
        .to_dict()
    )
    # 외래환자수 = 전체 - 입원
    hospital_patient_info = {}
    for hospital in total_patients:
        total = total_patients.get(hospital, 0)
        inpatient = inpatient_patients.get(hospital, 0)
        outpatient = max(0, total - inpatient)
        hospital_patient_info[hospital] = {
            'inpatient': int(inpatient),
            'outpatient': int(outpatient)
        }
    return hospital_patient_info

# ---------------------------
# 4. 현실적인 시뮬레이션 클래스 (입원/외래 분리)
# ---------------------------
class RealisticHospitalSimulation:
    def __init__(self, env, beds_capacity, total_specialists, hospital_patient_info, sim_time=8760, verbose=True):
        self.env = env
        self.beds = {h: simpy.Resource(env, capacity=c) for h, c in beds_capacity.items()}
        self.specialists = simpy.Resource(env, capacity=total_specialists)
        self.hospital_patient_info = hospital_patient_info
        self.sim_time = sim_time
        self.patients_served = 0
        self.patients_rejected = 0
        self.wait_times = []
        self.verbose = verbose
        self.inpatient_count = {h: 0 for h in beds_capacity}
        self.outpatient_count = {h: 0 for h in beds_capacity}
        self.env.process(self.patient_arrival_process())

    def patient_arrival_process(self):
        # 1년(8760시간) 동안 시간대별로 환자 도착
        for hospital, info in self.hospital_patient_info.items():
            # 입원환자 도착: 연간 입원환자수를 365일로 나눠서 일별 도착
            daily_inpatients = info['inpatient'] / 365
            for day in range(365):
                num_inpatients = np.random.poisson(daily_inpatients)
                for _ in range(num_inpatients):
                    self.env.process(self.patient_process(hospital, '입원'))
            # 외래환자 도착: 연간 외래환자수를 250일(평일)로 나눠서 일별 도착
            daily_outpatients = info['outpatient'] / 250
            for day in range(250):
                num_outpatients = np.random.poisson(daily_outpatients)
                for _ in range(num_outpatients):
                    self.env.process(self.patient_process(hospital, '외래'))
        yield self.env.timeout(self.sim_time)

    def patient_process(self, hospital, patient_type):
        arrival = self.env.now
        if hospital not in self.beds:
            self.patients_rejected += 1
            return
        if patient_type == '입원':
            with self.beds[hospital].request() as bed_req:
                result = yield bed_req | self.env.timeout(48)
                if bed_req not in result:
                    self.patients_rejected += 1
                    return
                with self.specialists.request() as doc_req:
                    yield doc_req
                    wait_time = self.env.now - arrival
                    self.wait_times.append(wait_time)
                    treatment_time = random.uniform(3, 14)  # 입원: 3~14일
                    yield self.env.timeout(treatment_time)
                    self.patients_served += 1
                    self.inpatient_count[hospital] += 1
        else:
            with self.specialists.request() as doc_req:
                yield doc_req
                wait_time = self.env.now - arrival
                self.wait_times.append(wait_time)
                treatment_time = random.uniform(0.5, 2.0)  # 외래: 0.5~2시간
                yield self.env.timeout(treatment_time)
                self.patients_served += 1
                self.outpatient_count[hospital] += 1

# ---------------------------
# 5. 평가지표 계산 함수
# ---------------------------
def calculate_evaluation_metrics(sim, beds_capacity, total_specialists):
    total_beds = sum(beds_capacity.values())
    bed_utilization = sum(sim.inpatient_count.values()) * 7 / (total_beds * 365) * 100 if total_beds > 0 else 0
    doctors_per_bed = total_specialists / total_beds if total_beds > 0 else 0
    outpatients_per_doc = sum(sim.outpatient_count.values()) / 250 / total_specialists if total_specialists > 0 else 0
    admission_outpatient_ratio = sum(sim.inpatient_count.values()) / sum(sim.outpatient_count.values()) if sum(sim.outpatient_count.values()) > 0 else 0
    total_patients = sim.patients_served + sim.patients_rejected
    rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
    avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0
    metrics = {
        '병상가동률': bed_utilization,
        '병상당의사수': doctors_per_bed,
        '의사당외래환자수': outpatients_per_doc,
        '외래대비입원비율': admission_outpatient_ratio,
        '환자거절률': rejection_rate,
        '평균대기시간': avg_wait,
        '총환자수': total_patients,
        '서비스환자수': sim.patients_served,
        '거절환자수': sim.patients_rejected
    }
    return metrics

# ---------------------------
# 6. 전국평균 비교 함수
# ---------------------------
def evaluate_metrics_against_national_average(metrics):
    national_averages = {
        '병상가동률': 85.0,
        '병상당의사수': 0.1671,
        '의사당외래환자수': 10.63,
        '외래대비입원비율': 0.38
    }
    evaluation = {}
    for metric, value in metrics.items():
        if metric in national_averages:
            national_avg = national_averages[metric]
            if metric == '병상가동률':
                if 85 <= value <= 90:
                    evaluation_text = '적정'
                elif value > 90:
                    evaluation_text = '높음 (병상부족)'
                else:
                    evaluation_text = '낮음 (과잉)'
            elif metric == '병상당의사수':
                if value >= national_avg:
                    evaluation_text = '적정'
                else:
                    evaluation_text = '낮음'
            elif metric == '의사당외래환자수':
                if value <= national_avg:
                    evaluation_text = '적정'
                else:
                    evaluation_text = '높음'
            elif metric == '외래대비입원비율':
                if abs(value - national_avg) <= 0.1:
                    evaluation_text = '적정'
                elif value > national_avg:
                    evaluation_text = '높음'
                else:
                    evaluation_text = '낮음'
            else:
                evaluation_text = 'N/A'
            evaluation[metric] = {
                '시뮬레이션값': value,
                '전국평균': national_avg,
                '평가': evaluation_text
            }
        else:
            evaluation[metric] = {
                '시뮬레이션값': value,
                '전국평균': 'N/A',
                '평가': 'N/A'
            }
    return evaluation

# ---------------------------
# 7. 메인 실행
# ---------------------------
if __name__ == "__main__":
    print("=== 미래 예측 환자수 기반 현실적 시뮬레이션 실행 ===")
    target_year = 2025
    hospital_patient_info = get_hospital_patient_info(target_year=target_year, target_col='XGB예측')
    sim_time = 8760  # 1년
    env = simpy.Environment()
    sim = RealisticHospitalSimulation(env, beds_capacity, total_specialists, hospital_patient_info, sim_time=sim_time, verbose=False)
    env.run(until=sim_time)
    metrics = calculate_evaluation_metrics(sim, beds_capacity, total_specialists)
    evaluation = evaluate_metrics_against_national_average(metrics)
    print("\n--- 결과 요약 ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\n--- 전국평균 비교 ---")
    for k, v in evaluation.items():
        print(f"{k}: {v['시뮬레이션값']} (전국평균: {v['전국평균']}) → {v['평가']}")
    # 결과 저장
    result_dir = 'simulation_results_realistic'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    pd.DataFrame([metrics]).to_csv(os.path.join(result_dir, f'scenario1_result_realistic_{target_year}.csv'), index=False)
    # 평가 리포트 저장
    rows = []
    for metric in evaluation:
        rows.append({
            '지표명': metric,
            '시뮬레이션값': evaluation[metric]['시뮬레이션값'],
            '전국평균': evaluation[metric]['전국평균'],
            '평가': evaluation[metric]['평가']
        })
    pd.DataFrame(rows).to_csv(os.path.join(result_dir, f'scenario1_evaluation_report_realistic_{target_year}.csv'), index=False) 