import simpy
import pandas as pd
import numpy as np
import random

# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
model1_df = pd.read_csv('model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv')
model3_df = pd.read_csv('model_results_진료과진료비_연령지역진료과연계.csv')
model4_df = pd.read_csv('model_results_진료과별병상수_예측모델_연령지역진료과추가/hospital_bed_prediction_results_Ridge_gridcv.csv')
model5_df = pd.read_csv('model_results_진료과_전문의_연령지역진료과/predictions/ElasticNet_predictions.csv')

# ---------------------------
# 2. 병상 수 설정
# ---------------------------
bed_columns = [col for col in model4_df.columns if col.endswith('_예측_Ridge')]
hospitals = model4_df['병원명'].unique()
beds_capacity = {
    hosp: int(round(model4_df[model4_df['병원명'] == hosp][bed_columns].sum(axis=1).values[0]))
    for hosp in hospitals
}

# ---------------------------
# 3. 전문의 수 설정
# ---------------------------
total_specialists = int(round(model5_df['y_predicted'].mean()))

# ---------------------------
# 4. 시나리오별 도착률 가중치 함수 (2~6번만)
# ---------------------------
def adjust_arrival_rates_scenario(df, scenario_id):
    df = df.copy()
    elderly = ['65-69', '70-79', '80-89', '90이상']
    chronic = [
        'I10', 'E11', 'F20', 'F06', 'I20', 'I25', 'I50', 'I65', 'I67',
        'I63', 'I69', 'G81', 'G82', 'C16', 'C18', 'C22', 'C25', 'C34',
        'C61', 'C67', 'D09', 'D41', 'N18'
    ]
    general_inpatient_codes = ['H25', 'J18', 'A09', 'M51', 'Z38']
    general_outpatient_codes = ['K05', 'J20', 'J30', 'I10', 'K02']

    # 연인원 기반 도착률 계산
    df['hourly_arrival'] = df['연인원'] / 8760

    # 구분 설정
    if '구분' not in df.columns:
        df['구분'] = '외래'
    df.loc[df['상병코드'].isin(general_inpatient_codes), '구분'] = '입원(연인원)'
    df.loc[df['상병코드'].isin(general_outpatient_codes), '구분'] = '외래'

    # 시나리오 2~6 반영
    if scenario_id == 2:
        df.loc[df['상병코드'] == 'N40', 'hourly_arrival'] *= 0.5
    elif scenario_id == 3:
        df.loc[df['상병코드'].isin(general_inpatient_codes + general_outpatient_codes), 'hourly_arrival'] *= 1.3
    elif scenario_id == 4:
        df.loc[df['연령대'].isin(elderly) & df['상병코드'].isin(chronic), 'hourly_arrival'] *= 1.7
    elif scenario_id == 5:
        df.loc[df['연령대'].isin(elderly), 'hourly_arrival'] *= 1.3
        df.loc[df['상병코드'] == 'N40', 'hourly_arrival'] *= 0.7
        df.loc[df['상병코드'].isin(general_inpatient_codes + general_outpatient_codes), 'hourly_arrival'] *= 1.1
        df.loc[df['연령대'].isin(elderly) & df['상병코드'].isin(chronic), 'hourly_arrival'] *= 1.4
    elif scenario_id == 6:
        df['hourly_arrival'] *= 0.8  # 예: 전체 수요 감소 시나리오

    return df

# ---------------------------
# 5. 평가지표 계산 함수
# ---------------------------
def calculate_bed_utilization(patient_groups_df, beds_capacity):
    total_admissions = patient_groups_df.loc[patient_groups_df['구분']=='입원(연인원)', '연인원'].sum()
    daily_avg_inpatients = total_admissions / 365
    total_beds = sum(beds_capacity.values())
    bed_utilization_rate = (daily_avg_inpatients / total_beds) * 100 if total_beds > 0 else 0
    return bed_utilization_rate, daily_avg_inpatients, total_beds

def calculate_doctors_per_bed(total_specialists, beds_capacity):
    total_beds = sum(beds_capacity.values())
    return total_specialists / total_beds if total_beds > 0 else 0

def calculate_outpatients_per_doctor(patient_groups_df, total_specialists):
    total_outpatients = patient_groups_df.loc[patient_groups_df['구분']=='외래', '연인원'].sum()
    daily_avg_outpatients = total_outpatients / 365
    return daily_avg_outpatients / total_specialists if total_specialists > 0 else 0

def calculate_admission_to_outpatient_ratio(patient_groups_df):
    total_admissions = patient_groups_df.loc[patient_groups_df['구분']=='입원(연인원)', '연인원'].sum()
    total_outpatients = patient_groups_df.loc[patient_groups_df['구분']=='외래', '연인원'].sum()
    return total_admissions / total_outpatients if total_outpatients > 0 else 0

def calculate_costliness_index(institution_avg_cost, national_avg_cost=1.0):
    if national_avg_cost == 0:
        return None
    return institution_avg_cost / national_avg_cost

def print_evaluation_metrics(sim, patient_groups_df, beds_capacity, total_specialists, institution_avg_cost=None):
    print("\n=== 평가지표 ===")

    bed_util_rate, daily_avg_inpatients, total_beds = calculate_bed_utilization(patient_groups_df, beds_capacity)
    print(f"병상가동률: {bed_util_rate:.2f}% (일평균 입원환자 수: {daily_avg_inpatients:.0f}, 병상수 합계: {total_beds})")

    doctors_per_bed = calculate_doctors_per_bed(total_specialists, beds_capacity)
    print(f"병상당 의사 수: {doctors_per_bed:.4f} 명/병상")

    outpatients_per_doc = calculate_outpatients_per_doctor(patient_groups_df, total_specialists)
    print(f"의사당 외래환자수 (일평균): {outpatients_per_doc:.2f} 명")

    admission_outpatient_ratio = calculate_admission_to_outpatient_ratio(patient_groups_df)
    print(f"외래환자 대비 입원환자비: {admission_outpatient_ratio:.4f}")

    if institution_avg_cost is not None:
        ci = calculate_costliness_index(institution_avg_cost)
        print(f"고가도지표 (CI): {ci:.3f}")

    total_patients = sim.patients_served + sim.patients_rejected
    rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
    avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0

    print(f"총 환자 수: {total_patients}")
    print(f"환자 거절률: {rejection_rate:.2f}%")
    print(f"평균 대기 시간: {avg_wait:.2f} 시간")

# ---------------------------
# 6. 시뮬레이션 클래스 정의
# ---------------------------
class HospitalSimulation:
    def __init__(self, env, beds_capacity, total_specialists, patient_groups_df, verbose=True):
        self.env = env
        self.beds = {h: simpy.Resource(env, capacity=c) for h, c in beds_capacity.items()}
        self.specialists = simpy.Resource(env, capacity=total_specialists)
        self.patient_groups = patient_groups_df
        self.patients_served = 0
        self.patients_rejected = 0
        self.wait_times = []
        self.verbose = verbose

        self.specialist_usage_timeline = []
        self.bed_usage_timeline = {h: [] for h in self.beds}

        self.env.process(self.monitor_specialist_usage())
        self.env.process(self.monitor_bed_usage())

    def monitor_specialist_usage(self):
        while True:
            yield self.env.timeout(1)
            usage = self.specialists.count
            self.specialist_usage_timeline.append((self.env.now, usage))

    def monitor_bed_usage(self):
        while True:
            yield self.env.timeout(1)
            for hosp, res in self.beds.items():
                self.bed_usage_timeline[hosp].append((self.env.now, res.count))

    def patient_arrival_process(self):
        while True:
            valid_groups = self.patient_groups[self.patient_groups['hourly_arrival'] > 0]
            if valid_groups.empty:
                yield self.env.timeout(1)
                continue

            group = valid_groups.sample(weights=valid_groups['hourly_arrival'], replace=True).iloc[0]
            inter_arrival = np.random.exponential(1 / group['hourly_arrival'])
            yield self.env.timeout(inter_arrival)

            hospital = group['병원명']
            duration = self.get_treatment_time(group)
            pid = f"Pt-{self.env.now:.2f}"
            self.env.process(self.patient_process(pid, hospital, duration))

    def get_treatment_time(self, group):
        if '구분' in group and group['구분'] == '입원(연인원)':
            return random.uniform(5, 10)
        return random.uniform(1, 3)

    def patient_process(self, pid, hospital, duration):
        arrival = self.env.now
        if self.verbose:
            print(f"{arrival:.2f} - {pid} 도착 (병원: {hospital})")

        if hospital not in self.beds:
            if self.verbose:
                print(f"{self.env.now:.2f} - {pid} 병원 없음 → 거절")
            self.patients_rejected += 1
            return

        bed_res = self.beds[hospital]
        with bed_res.request() as bed_req:
            result = yield bed_req | self.env.timeout(1)
            if bed_req not in result:
                if self.verbose:
                    print(f"{self.env.now:.2f} - {pid} 병상 없음 → 거절")
                self.patients_rejected += 1
                return

            with self.specialists.request() as doc_req:
                yield doc_req
                wait_time = self.env.now - arrival
                self.wait_times.append(wait_time)
                if self.verbose:
                    print(f"{self.env.now:.2f} - {pid} 진료 시작 (대기: {wait_time:.2f}시간)")
                yield self.env.timeout(duration)
                if self.verbose:
                    print(f"{self.env.now:.2f} - {pid} 진료 종료")
                self.patients_served += 1

def compute_avg_utilization(timeline, capacity):
    if len(timeline) < 2:
        return 0
    total_time = timeline[-1][0] - timeline[0][0]
    usage_sum = 0
    for i in range(len(timeline) - 1):
        interval = timeline[i+1][0] - timeline[i][0]
        usage_sum += timeline[i][1] * interval
    avg_util = usage_sum / total_time / capacity if capacity > 0 else 0
    return avg_util

# ---------------------------
# 시나리오 실행 함수
# ---------------------------
import os

def run_simulation_by_scenario(scenario_id, sim_time=2, verbose=False, save_csv=True, output_dir='simulation_results'):
    df_scenario = adjust_arrival_rates_scenario(model3_df, scenario_id)

    env = simpy.Environment()
    sim = HospitalSimulation(env, beds_capacity, total_specialists, df_scenario, verbose=verbose)
    env.process(sim.patient_arrival_process())
    env.run(until=sim_time)

    avg_specialist_util = compute_avg_utilization(sim.specialist_usage_timeline, sim.specialists.capacity)

    avg_bed_utils = []
    for hosp, timeline in sim.bed_usage_timeline.items():
        capacity = sim.beds[hosp].capacity
        avg_bed_utils.append(compute_avg_utilization(timeline, capacity))
    avg_bed_util = np.mean(avg_bed_utils) if avg_bed_utils else 0

    total_patients = sim.patients_served + sim.patients_rejected
    rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
    avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0

    print(f"\n--- 시나리오 {scenario_id} 결과 ---")
    print(f"총 진료 완료 환자 수: {sim.patients_served}")
    print(f"총 거절 환자 수: {sim.patients_rejected}")
    print(f"평균 대기 시간: {avg_wait:.2f} 시간")
    print(f"평균 병상 사용률 (시간 가중): {avg_bed_util:.2f}")
    print(f"평균 전문의 사용률 (시간 가중): {avg_specialist_util:.2f}")

    print_evaluation_metrics(sim, df_scenario, beds_capacity, total_specialists, institution_avg_cost=None)

    if save_csv:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        result_dict = {
            'scenario_id': scenario_id,
            'patients_served': sim.patients_served,
            'patients_rejected': sim.patients_rejected,
            'total_patients': total_patients,
            'rejection_rate_percent': rejection_rate,
            'average_wait_time_hours': avg_wait,
            'average_bed_utilization': avg_bed_util,
            'average_specialist_utilization': avg_specialist_util,
            'bed_capacity_total': sum(beds_capacity.values()),
            'total_specialists': total_specialists,
        }
        df_result = pd.DataFrame([result_dict])
        csv_path = os.path.join(output_dir, f'simulation_scenario_{scenario_id}.csv')
        df_result.to_csv(csv_path, index=False)
        print(f"결과를 {csv_path} 에 저장했습니다.")

# ---------------------------
# 메인 실행 (시나리오 2~6)
# ---------------------------
if __name__ == "__main__":
    for scenario in range(2, 7):
        run_simulation_by_scenario(scenario_id=scenario, sim_time=2, verbose=True) # sim_time
