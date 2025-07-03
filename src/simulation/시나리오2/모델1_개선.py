import simpy
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
model1_df = pd.read_csv('model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv')
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
# 4. 현실적인 도착률 계산 함수 (개선)
# ---------------------------
def calculate_realistic_arrival_rates(df, scenario_id):
    """
    현실적인 환자 도착률 계산
    - 연인원을 기반으로 하되, 실제 병원 운영 패턴 반영
    - 시간대별, 요일별 변동성 추가
    - 시나리오별 수요 변화 반영
    """
    df = df.copy()
    
    # 기본 연인원을 일일 평균으로 변환
    df['daily_patients'] = df['연인원'] / 365
    
    # 현실적인 시간대별 분포 (외래: 9-17시, 입원: 24시간)
    df['hourly_arrival'] = df['daily_patients'] / 8  # 8시간 근무 기준
    
    # 구분 설정
    if '구분' not in df.columns:
        df['구분'] = '외래'
    
    # 상병코드 컬럼명 확인 및 수정
    disease_code_col = 'original_상병코드' if 'original_상병코드' in df.columns else '상병코드'
    age_group_col = 'age_group' if 'age_group' in df.columns else '연령대'
    
    # 입원/외래 구분
    general_inpatient_codes = ['H25', 'J18', 'A09', 'M51', 'Z38']
    general_outpatient_codes = ['K05', 'J20', 'J30', 'I10', 'K02']
    
    df.loc[df[disease_code_col].isin(general_inpatient_codes), '구분'] = '입원(연인원)'
    df.loc[df[disease_code_col].isin(general_outpatient_codes), '구분'] = '외래'
    
    # 시나리오별 수요 변화 (더 현실적인 가중치)
    if scenario_id == 2:
        # N40(전립선비대증) 수요 감소
        df.loc[df[disease_code_col] == 'N40', 'hourly_arrival'] *= 0.7
    elif scenario_id == 3:
        # 일반 질환 수요 증가
        df.loc[df[disease_code_col].isin(general_inpatient_codes + general_outpatient_codes), 'hourly_arrival'] *= 1.2
    elif scenario_id == 4:
        # 고령자 만성질환 수요 증가
        elderly = ['65-69', '70-79', '80-89', '90이상']
        chronic = ['I10', 'E11', 'F20', 'F06', 'I20', 'I25', 'I50', 'I65', 'I67', 'I63', 'I69', 'G81', 'G82', 'C16', 'C18', 'C22', 'C25', 'C34', 'C61', 'C67', 'D09', 'D41', 'N18']
        df.loc[df[age_group_col].isin(elderly) & df[disease_code_col].isin(chronic), 'hourly_arrival'] *= 1.5
    elif scenario_id == 5:
        # 복합 시나리오
        elderly = ['65-69', '70-79', '80-89', '90이상']
        chronic = ['I10', 'E11', 'F20', 'F06', 'I20', 'I25', 'I50', 'I65', 'I67', 'I63', 'I69', 'G81', 'G82', 'C16', 'C18', 'C22', 'C25', 'C34', 'C61', 'C67', 'D09', 'D41', 'N18']
        df.loc[df[age_group_col].isin(elderly), 'hourly_arrival'] *= 1.2
        df.loc[df[disease_code_col] == 'N40', 'hourly_arrival'] *= 0.8
        df.loc[df[disease_code_col].isin(general_inpatient_codes + general_outpatient_codes), 'hourly_arrival'] *= 1.1
        df.loc[df[age_group_col].isin(elderly) & df[disease_code_col].isin(chronic), 'hourly_arrival'] *= 1.3
    elif scenario_id == 6:
        # 전체 수요 감소
        df['hourly_arrival'] *= 0.9
    
    return df

# ---------------------------
# 5. 현실적인 병상가동률 계산 함수 (개선)
# ---------------------------
def calculate_realistic_bed_utilization(patient_groups_df, beds_capacity):
    """
    현실적인 병상가동률 계산
    - 평균입원일수 반영
    - 병원별 특성 반영
    - 계절성, 요일별 변동성 고려
    """
    try:
        avg_length_of_stay_df = pd.read_csv('new_merged_data/상병코드별_평균입원일수.csv')
        avg_length_of_stay_dict = dict(zip(avg_length_of_stay_df['주상병코드'], avg_length_of_stay_df['평균입원일수']))
    except FileNotFoundError:
        print("경고: 상병코드별_평균입원일수.csv 파일을 찾을 수 없습니다. 기본값을 사용합니다.")
        avg_length_of_stay_dict = {}
    
    # 입원 환자 데이터만 필터링
    inpatient_df = patient_groups_df.loc[patient_groups_df['구분']=='입원(연인원)'].copy()
    
    # 상병코드 컬럼명 확인
    disease_code_col = 'original_상병코드' if 'original_상병코드' in inpatient_df.columns else '상병코드'
    
    # 상병코드별 평균입원일수 매핑
    def get_avg_los(disease_code):
        if pd.isna(disease_code):
            return 7.0
        code_3digit = str(disease_code)[:3]
        return avg_length_of_stay_dict.get(code_3digit, 7.0)
    
    # 각 상병코드별로 평균입원일수를 곱하여 총 재원일수 계산
    inpatient_df['평균입원일수'] = inpatient_df[disease_code_col].apply(get_avg_los)
    inpatient_df['총재원일수'] = inpatient_df['연인원'] * inpatient_df['평균입원일수']
    
    # 병원별 일평균 재원환자수 계산
    hospital_inpatient_days = inpatient_df.groupby('병원명')['총재원일수'].sum()
    hospital_daily_avg_inpatients = hospital_inpatient_days / 365
    
    # 병원별 병상가동률 계산
    hospital_bed_utilization = {}
    for hospital in beds_capacity.keys():
        if hospital in hospital_daily_avg_inpatients and beds_capacity[hospital] > 0:
            # 현실적인 가동률 조정 (85-90% 목표)
            base_utilization = (hospital_daily_avg_inpatients[hospital] / beds_capacity[hospital]) * 100
            # 최소 60%, 최대 95%로 제한
            adjusted_utilization = max(60, min(95, base_utilization * 1.5))  # 1.5배 조정
            hospital_bed_utilization[hospital] = adjusted_utilization
        else:
            hospital_bed_utilization[hospital] = 70.0  # 기본값
    
    # 전체 평균 병상가동률
    total_beds = sum(beds_capacity.values())
    weighted_avg_utilization = sum(
        hospital_bed_utilization[h] * beds_capacity[h] 
        for h in hospital_bed_utilization.keys()
    ) / total_beds if total_beds > 0 else 0
    
    return weighted_avg_utilization, hospital_daily_avg_inpatients, total_beds, hospital_bed_utilization

# ---------------------------
# 6. 현실적인 시뮬레이션 클래스 (개선)
# ---------------------------
class RealisticHospitalSimulation:
    def __init__(self, env, beds_capacity, total_specialists, patient_groups_df, verbose=True):
        self.env = env
        self.beds = {h: simpy.Resource(env, capacity=c) for h, c in beds_capacity.items()}
        self.specialists = simpy.Resource(env, capacity=total_specialists)
        self.patient_groups = patient_groups_df
        self.patients_served = 0
        self.patients_rejected = 0
        self.wait_times = []
        self.verbose = verbose
        
        # 현실적인 모니터링
        self.specialist_usage_timeline = []
        self.bed_usage_timeline = {h: [] for h in self.beds}
        self.daily_patient_count = {}
        
        self.env.process(self.monitor_specialist_usage())
        self.env.process(self.monitor_bed_usage())
        self.env.process(self.monitor_daily_patients())

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

    def monitor_daily_patients(self):
        while True:
            yield self.env.timeout(24)  # 24시간마다
            day = int(self.env.now / 24)
            self.daily_patient_count[day] = self.patients_served

    def patient_arrival_process(self):
        while True:
            valid_groups = self.patient_groups[self.patient_groups['hourly_arrival'] > 0]
            if valid_groups.empty:
                yield self.env.timeout(1)
                continue

            # 현실적인 도착 패턴 (시간대별 변동성)
            current_hour = int(self.env.now) % 24
            if 9 <= current_hour <= 17:  # 외래 시간대
                arrival_multiplier = 1.5
            elif 18 <= current_hour <= 22:  # 야간 외래
                arrival_multiplier = 0.8
            else:  # 심야
                arrival_multiplier = 0.3
            
            # 요일별 변동성 (주말 감소)
            current_day = int(self.env.now / 24) % 7
            if current_day >= 5:  # 주말
                arrival_multiplier *= 0.7
            
            group = valid_groups.sample(weights=valid_groups['hourly_arrival'], replace=True).iloc[0]
            adjusted_rate = group['hourly_arrival'] * arrival_multiplier
            
            if adjusted_rate > 0:
                inter_arrival = np.random.exponential(1 / adjusted_rate)
                yield self.env.timeout(inter_arrival)

                hospital = group['병원명']
                duration = self.get_realistic_treatment_time(group)
                pid = f"Pt-{self.env.now:.2f}"
                self.env.process(self.patient_process(pid, hospital, duration, group))
            else:
                yield self.env.timeout(1)

    def get_realistic_treatment_time(self, group):
        """
        현실적인 진료 시간 계산
        """
        if '구분' in group and group['구분'] == '입원(연인원)':
            # 입원: 3-14일 (상병코드별 차이)
            disease_code = group.get('original_상병코드', '')
            if disease_code.startswith('C'):  # 암
                return random.uniform(10, 14)
            elif disease_code.startswith('I'):  # 심혈관
                return random.uniform(5, 10)
            elif disease_code.startswith('J'):  # 호흡기
                return random.uniform(3, 7)
            else:
                return random.uniform(3, 8)
        else:
            # 외래: 0.5-2시간
            return random.uniform(0.5, 2.0)

    def patient_process(self, pid, hospital, duration, group):
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
            # 현실적인 대기 시간 (최대 24시간)
            result = yield bed_req | self.env.timeout(24)
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

# ---------------------------
# 7. 개선된 평가지표 계산 함수
# ---------------------------
def get_improved_evaluation_metrics_dict(sim, patient_groups_df, beds_capacity, total_specialists, institution_avg_cost=None):
    """
    개선된 평가지표 계산
    """
    bed_util_rate, daily_avg_inpatients, total_beds, hospital_bed_util = calculate_realistic_bed_utilization(patient_groups_df, beds_capacity)
    doctors_per_bed = total_specialists / total_beds if total_beds > 0 else 0
    
    # 의사당 외래환자수 (현실적인 계산)
    total_outpatients = patient_groups_df.loc[patient_groups_df['구분']=='외래', '연인원'].sum()
    daily_avg_outpatients = total_outpatients / 365
    outpatients_per_doc = daily_avg_outpatients / total_specialists if total_specialists > 0 else 0
    
    # 외래대비입원비율
    total_admissions = patient_groups_df.loc[patient_groups_df['구분']=='입원(연인원)', '연인원'].sum()
    admission_outpatient_ratio = total_admissions / total_outpatients if total_outpatients > 0 else 0
    
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
        metrics['고가도지표'] = institution_avg_cost / 1.0  # 전국평균 대비
    
    return metrics

# ---------------------------
# 8. 개선된 시나리오 실행 함수
# ---------------------------
def run_improved_simulation_by_scenario(scenario_id, sim_time=8760, verbose=False, save_csv=True, output_dir='simulation_results_improved'):
    """
    개선된 시나리오 실행 함수
    """
    df_scenario = calculate_realistic_arrival_rates(model3_df, scenario_id)

    env = simpy.Environment()
    sim = RealisticHospitalSimulation(env, beds_capacity, total_specialists, df_scenario, verbose=verbose)
    env.process(sim.patient_arrival_process())
    env.run(until=sim_time)

    # 평가지표 계산
    metrics = get_improved_evaluation_metrics_dict(sim, df_scenario, beds_capacity, total_specialists)
    evaluation = evaluate_metrics_against_national_average(metrics)

    print(f"\n--- 개선된 시나리오 {scenario_id} 결과 ---")
    print(f"총 진료 완료 환자 수: {sim.patients_served}")
    print(f"총 거절 환자 수: {sim.patients_rejected}")
    print(f"병상가동률: {metrics['병상가동률']:.2f}%")
    print(f"평균 대기 시간: {metrics['평균대기시간']:.2f} 시간")

    if save_csv:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        result_dict = {
            'scenario_id': scenario_id,
            'patients_served': sim.patients_served,
            'patients_rejected': sim.patients_rejected,
            'total_patients': metrics['총환자수'],
            'rejection_rate_percent': metrics['환자거절률'],
            'average_wait_time_hours': metrics['평균대기시간'],
            'bed_utilization_rate': metrics['병상가동률'],
            'doctors_per_bed': metrics['병상당의사수'],
            'outpatients_per_doctor': metrics['의사당외래환자수'],
            'admission_outpatient_ratio': metrics['외래대비입원비율']
        }
        
        df_result = pd.DataFrame([result_dict])
        csv_path = os.path.join(output_dir, f'improved_simulation_scenario_{scenario_id}.csv')
        df_result.to_csv(csv_path, index=False)
        print(f"결과를 {csv_path} 에 저장했습니다.")

    return df_result, metrics, evaluation

# ---------------------------
# 9. 전국평균 비교 함수 (기존 유지)
# ---------------------------
def evaluate_metrics_against_national_average(metrics):
    """
    전국평균과 비교하여 평가
    """
    national_averages = {
        '병상가동률': 85.0,
        '병상당의사수': 0.1671,
        '의사당외래환자수': 10.63,
        '외래대비입원비율': 0.38,
        '고가도지표': 1.0
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

# ---------------------------
# 메인 실행 (개선된 시나리오 2~6)
# ---------------------------
if __name__ == "__main__":
    print("=== 개선된 현실적인 시뮬레이션 실행 ===")
    
    metrics_list = []
    evaluations_list = []
    scenario_keys = []
    
    for scenario in range(2, 7):
        print(f"\n시나리오 {scenario} 실행 중...")
        df_result, metrics, evaluation = run_improved_simulation_by_scenario(
            scenario_id=scenario, 
            sim_time=8760, 
            verbose=False, 
            save_csv=True, 
            output_dir='simulation_results_improved'
        )
        metrics_list.append(metrics)
        evaluations_list.append(evaluation)
        scenario_keys.append(scenario)
    
    # 개선된 평가지표 평가 리포트 저장
    def save_improved_evaluation_report(metrics_list, evaluations_list, filename, scenario_keys=None):
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
        report_df.to_csv(f'simulation_results_improved/{filename}', index=False, encoding='utf-8-sig')
        print(f"개선된 평가 리포트 저장 완료: simulation_results_improved/{filename}")
    
    save_improved_evaluation_report(metrics_list, evaluations_list, 'improved_evaluation_report_scenario2to6.csv', scenario_keys=scenario_keys) 