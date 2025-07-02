import simpy
import pandas as pd
import numpy as np
import random

# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
model1_df = pd.read_csv('model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv')
# model3_df = pd.read_csv('model_results_진료과진료비_연령지역진료과연계.csv')
model3_df = pd.read_csv('model_results_v3_연령지역진료과_확장/prediction_results_2.csv')
model4_df = pd.read_csv('model_results_진료과별병상수_예측모델_연령지역진료과추가/hospital_bed_prediction_results_Ridge_gridcv.csv')
model5_df = pd.read_csv('model_results_진료과_전문의_연령지역진료과/predictions/ElasticNet_predictions.csv')

# model3_df에서 y_pred_Original_XGBoost 컬럼을 사용하여 진료비 예측값으로 설정
if 'y_pred_Original_XGBoost' in model3_df.columns:
    model3_df['진료비(천원)'] = model3_df['y_pred_Original_XGBoost']
    print("y_pred_Original_XGBoost 컬럼을 진료비 예측값으로 사용합니다.")
else:
    print("경고: y_pred_Original_XGBoost 컬럼을 찾을 수 없습니다. 기존 진료비(천원) 컬럼을 사용합니다.")

# model3_df에 연령대 정보 추가 (model1_df에서 가져오기)
if 'original_연령대' in model3_df.columns:
    model3_df['age_group'] = model3_df['original_연령대']
    print("original_연령대 컬럼을 age_group으로 매핑했습니다.")
elif 'age_group' in model1_df.columns and 'age_group' not in model3_df.columns:
    # model1_df에서 상병코드별 연령대 정보를 가져와서 model3_df에 추가
    age_mapping = model1_df.groupby('disease_group')['age_group'].first().to_dict()
    model3_df['age_group'] = model3_df['original_상병코드'].map(lambda x: age_mapping.get(x, None))
    print("연령대 정보를 model1_df에서 가져와서 model3_df에 추가했습니다.")

# model3_df에 병원명 컬럼명 확인 및 수정
if 'original_병원명' in model3_df.columns:
    model3_df['병원명'] = model3_df['original_병원명']
    print("original_병원명 컬럼을 병원명으로 매핑했습니다.")
elif '병원명' not in model3_df.columns:
    print("경고: 병원명 컬럼을 찾을 수 없습니다.")

# ---------------------------
# 2. 병상 수 설정 (실제 데이터 사용)
# ---------------------------
# 실제 병원 데이터에서 병상수 계산
hospital_data = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')
bed_columns = [col for col in hospital_data.columns if any(x in col for x in ['실', '병실', '입원실', '중환자실'])]
beds_capacity = {
    hosp: int(round(hospital_data[hospital_data['병원명'] == hosp][bed_columns].sum(axis=1).values[0]))
    for hosp in hospital_data['병원명'].unique()
}

# ---------------------------
# 3. 전문의 수 설정 (실제 데이터 사용)
# ---------------------------
doc_columns = [col for col in hospital_data.columns if '전문의수' in col]
total_specialists = int(round(hospital_data[doc_columns].sum().sum()))

print(f"실제 병상수: {sum(beds_capacity.values())}")
print(f"실제 전문의수: {total_specialists}")

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

    # 연인원 기반 도착률 계산 (개선된 버전)
    # 연인원을 일일 평균으로 변환 후 시간당 도착률 계산
    df['daily_patients'] = df['연인원'] / 365
    df['hourly_arrival'] = df['daily_patients'] / 24  # 24시간 기준

    # 구분 설정
    if '구분' not in df.columns:
        df['구분'] = '외래'
    
    # 상병코드 컬럼명 확인 및 수정
    disease_code_col = 'original_상병코드' if 'original_상병코드' in df.columns else '상병코드'
    age_group_col = 'age_group' if 'age_group' in df.columns else '연령대'
    
    df.loc[df[disease_code_col].isin(general_inpatient_codes), '구분'] = '입원(연인원)'
    df.loc[df[disease_code_col].isin(general_outpatient_codes), '구분'] = '외래'

    # 시나리오 2~6 반영
    if scenario_id == 2:
        df.loc[df[disease_code_col] == 'N40', 'hourly_arrival'] *= 0.5
    elif scenario_id == 3:
        df.loc[df[disease_code_col].isin(general_inpatient_codes + general_outpatient_codes), 'hourly_arrival'] *= 1.3
    elif scenario_id == 4:
        df.loc[df[age_group_col].isin(elderly) & df[disease_code_col].isin(chronic), 'hourly_arrival'] *= 1.7
    elif scenario_id == 5:
        df.loc[df[age_group_col].isin(elderly), 'hourly_arrival'] *= 1.3
        df.loc[df[disease_code_col] == 'N40', 'hourly_arrival'] *= 0.7
        df.loc[df[disease_code_col].isin(general_inpatient_codes + general_outpatient_codes), 'hourly_arrival'] *= 1.1
        df.loc[df[age_group_col].isin(elderly) & df[disease_code_col].isin(chronic), 'hourly_arrival'] *= 1.4
    elif scenario_id == 6:
        df['hourly_arrival'] *= 0.8  # 예: 전체 수요 감소 시나리오

    return df

# ---------------------------
# 5. 평가지표 계산 함수
# ---------------------------
def calculate_bed_utilization(patient_groups_df, beds_capacity):
    # 상병코드별 평균입원일수 데이터 로드
    try:
        avg_length_of_stay_df = pd.read_csv('new_merged_data/상병코드별_평균입원일수.csv')
        avg_length_of_stay_dict = dict(zip(avg_length_of_stay_df['주상병코드'], avg_length_of_stay_df['평균입원일수']))
    except FileNotFoundError:
        print("경고: 상병코드별_평균입원일수.csv 파일을 찾을 수 없습니다. 기본값 7일을 사용합니다.")
        avg_length_of_stay_dict = {}
    
    # 입원 환자 데이터만 필터링
    inpatient_df = patient_groups_df.loc[patient_groups_df['구분']=='입원(연인원)'].copy()
    
    # 상병코드 컬럼명 확인
    disease_code_col = 'original_상병코드' if 'original_상병코드' in inpatient_df.columns else '상병코드'
    
    # 상병코드별 평균입원일수 매핑 (상위 3자리 코드 사용)
    def get_avg_los(disease_code):
        if pd.isna(disease_code):
            return 7.0  # 기본값
        code_3digit = str(disease_code)[:3]
        return avg_length_of_stay_dict.get(code_3digit, 7.0)  # 기본값 7일
    
    # 각 상병코드별로 평균입원일수를 곱하여 총 재원일수 계산
    inpatient_df['평균입원일수'] = inpatient_df[disease_code_col].apply(get_avg_los)
    inpatient_df['총재원일수'] = inpatient_df['연인원'] * inpatient_df['평균입원일수']
    
    # 일평균 재원환자수 계산 (총재원일수 / 365)
    daily_avg_inpatients = inpatient_df['총재원일수'].sum() / 365
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

def get_evaluation_metrics_dict(sim, patient_groups_df, beds_capacity, total_specialists, institution_avg_cost=None):
    """
    평가지표 계산 결과를 딕셔너리로 반환
    """
    bed_util_rate, daily_avg_inpatients, total_beds = calculate_bed_utilization(patient_groups_df, beds_capacity)
    doctors_per_bed = calculate_doctors_per_bed(total_specialists, beds_capacity)
    outpatients_per_doc = calculate_outpatients_per_doctor(patient_groups_df, total_specialists)
    admission_outpatient_ratio = calculate_admission_to_outpatient_ratio(patient_groups_df)
    
    total_patients = sim.patients_served + sim.patients_rejected
    rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
    avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0
    
    metrics = {
        '병상가동률': bed_util_rate,
        '병상당의사수': doctors_per_bed,
        '의사당외래환자수': outpatients_per_doc,
        '외래대비입원비율': admission_outpatient_ratio,
        '환자거절률': rejection_rate,
        '평균대기시간': avg_wait,
        '총환자수': total_patients,
        '서비스환자수': sim.patients_served,
        '거절환자수': sim.patients_rejected
    }
    
    if institution_avg_cost is not None:
        metrics['고가도지표'] = calculate_costliness_index(institution_avg_cost)
    
    return metrics

def evaluate_metrics_against_national_average(metrics):
    """
    전국평균과 비교하여 평가
    """
    national_averages = {
        '병상가동률': 85.0,  # 85~90%가 이상적
        '병상당의사수': 0.1671,  # 전국평균
        '의사당외래환자수': 10.63,  # 전국평균
        '외래대비입원비율': 0.38,  # 전국평균
        '고가도지표': 1.0  # 전국평균
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
            elif metric == '고가도지표':
                if value >= 1.2:
                    evaluation_text = '고가도'
                elif 0.8 <= value <= 1.2:
                    evaluation_text = '일반'
                else:
                    evaluation_text = '저가도'
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

def save_evaluation_report(metrics_list, evaluations_list, filename, scenario_keys=None):
    """
    평가지표 평가 리포트를 CSV로 저장
    """
    rows = []
    for i, (metrics, evaluation) in enumerate(zip(metrics_list, evaluations_list)):
        key = scenario_keys[i] if scenario_keys else (i+1)
        for metric in evaluation:
            rows.append({
                '시나리오': key,
                '지표명': metric,
                '시뮬레이션값': evaluation[metric]['시뮬레이션값'],
                '전국평균': evaluation[metric]['전국평균'],
                '평가': evaluation[metric]['평가']
            })
    
    report_df = pd.DataFrame(rows)
    report_df.to_csv(f'simulation_results/{filename}', index=False, encoding='utf-8-sig')
    print(f"평가 리포트 저장 완료: simulation_results/{filename}")

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

    # 평가지표 계산 및 평가
    metrics = get_evaluation_metrics_dict(sim, df_scenario, beds_capacity, total_specialists)
    evaluation = evaluate_metrics_against_national_average(metrics)

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
            '병상가동률': metrics['병상가동률'],
            '병상당의사수': metrics['병상당의사수'],
            '의사당외래환자수': metrics['의사당외래환자수'],
            '외래대비입원비율': metrics['외래대비입원비율']
        }
        df_result = pd.DataFrame([result_dict])
        csv_path = os.path.join(output_dir, f'simulation_scenario_{scenario_id}.csv')
        df_result.to_csv(csv_path, index=False)
        print(f"결과를 {csv_path} 에 저장했습니다.")

    return df_result, metrics, evaluation

# ---------------------------
# 메인 실행 (시나리오 2~6)
# ---------------------------
if __name__ == "__main__":
    metrics_list = []
    evaluations_list = []
    scenario_keys = []
    for scenario in range(2, 7):
        df_result, metrics, evaluation = run_simulation_by_scenario(scenario_id=scenario, sim_time=8760, verbose=False, save_csv=True, output_dir='simulation_results')
        metrics_list.append(metrics)
        evaluations_list.append(evaluation)
        scenario_keys.append(scenario)
    # 평가지표 평가 리포트 저장
    save_evaluation_report(metrics_list, evaluations_list, 'evaluation_report_scenario2to6.csv', scenario_keys=scenario_keys)
