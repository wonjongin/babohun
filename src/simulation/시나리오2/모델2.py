import simpy
import pandas as pd
import numpy as np
import random

# --- 통합 데이터에서 병상/전문의 수 추출 ---
hosp_df = pd.read_csv('new_merged_data/병원_통합_데이터_호스피스 삭제.csv')

bed_columns = [col for col in hosp_df.columns if not col.endswith('_전문의수') and col not in ['병원명']]
specialist_columns = [col for col in hosp_df.columns if col.endswith('_전문의수')]

hosp_df['총병상수'] = hosp_df[bed_columns].sum(axis=1)
hosp_df['총전문의수'] = hosp_df[specialist_columns].sum(axis=1)

beds_capacity_dict = dict(zip(hosp_df['병원명'], hosp_df['총병상수']))
specialists_capacity_dict = dict(zip(hosp_df['병원명'], hosp_df['총전문의수']))

# 진료비 데이터 로드
cost_df = pd.read_csv('model_results_v3_시계열_확장/prediction_results_2.csv')

class HospitalSim:
    def __init__(self, env, beds_capacity, specialists_capacity, patients_info):
        self.env = env
        self.beds = {h: simpy.Resource(env, capacity=int(c)) for h, c in beds_capacity.items()}
        self.specialists = {h: simpy.Resource(env, capacity=int(c)) for h, c in specialists_capacity.items()}
        self.patients_info = patients_info
        self.patients_served = 0
        self.patients_rejected = 0
        self.wait_times = []
        self.total_cost = 0
        self.bed_occupancy_time = {hospital: 0 for hospital in beds_capacity}
        # 평가지표 계산을 위한 추가 데이터
        self.inpatient_count = {hospital: 0.0 for hospital in beds_capacity}
        self.outpatient_count = {hospital: 0.0 for hospital in beds_capacity}
        self.total_treatment_time = {hospital: 0.0 for hospital in beds_capacity}

    def patient_arrival(self):
        while True:
            for hospital, info in self.patients_info.items():
                inpatient_rate = info['inpatient'] / 8760
                outpatient_rate = info['outpatient'] / 8760
                cost_in = info['cost_in']
                cost_out = info['cost_out']
                num_inpatients = np.random.poisson(inpatient_rate)
                num_outpatients = np.random.poisson(outpatient_rate)
                for _ in range(num_inpatients):
                    self.env.process(self.patient_process(hospital, '입원(연인원)', cost_in))
                for _ in range(num_outpatients):
                    self.env.process(self.patient_process(hospital, '외래', cost_out))
            yield self.env.timeout(0.1)

    def patient_process(self, hospital, patient_type, cost):
        arrival = self.env.now
        if patient_type == '입원(연인원)':
            if hospital not in self.beds or hospital not in self.specialists:
                self.patients_rejected += 1
                return
            with self.beds[hospital].request() as bed_req:
                result = yield bed_req | self.env.timeout(24)
                if bed_req not in result:
                    self.patients_rejected += 1
                    return
                start_time = self.env.now
                with self.specialists[hospital].request() as doc_req:
                    result = yield doc_req | self.env.timeout(24)
                    if doc_req not in result:
                        self.patients_rejected += 1
                        return
                    self.wait_times.append(self.env.now - arrival)
                    treatment_time = random.uniform(5, 10)
                    yield self.env.timeout(treatment_time)
                    self.total_cost += cost
                    self.patients_served += 1
                    self.inpatient_count[hospital] += 1
                    self.total_treatment_time[hospital] += treatment_time
                end_time = self.env.now
                self.bed_occupancy_time[hospital] += (end_time - start_time)
        else:
            if hospital not in self.specialists:
                self.patients_rejected += 1
                return
            with self.specialists[hospital].request() as doc_req:
                result = yield doc_req | self.env.timeout(24)
                if doc_req not in result:
                    self.patients_rejected += 1
                    return
                self.wait_times.append(self.env.now - arrival)
                treatment_time = random.uniform(1, 3)
                yield self.env.timeout(treatment_time)
                self.total_cost += cost
                self.patients_served += 1
                self.outpatient_count[hospital] += 1
                self.total_treatment_time[hospital] += treatment_time

def calculate_healthcare_metrics(beds_capacity, specialists_capacity, patients_info, sim):
    """
    의료 평가지표 계산 함수 (병상가동률: 평균입원일수 반영)
    """
    metrics = {}
    
    # 1. 병상 관련 지표
    # 병상가동률 (총재원일수/365 / 병상수) * 100
    # 상병코드별 평균입원일수 데이터 로드
    try:
        avg_length_of_stay_df = pd.read_csv('new_merged_data/상병코드별_평균입원일수.csv')
        avg_length_of_stay_dict = dict(zip(avg_length_of_stay_df['주상병코드'], avg_length_of_stay_df['평균입원일수']))
    except FileNotFoundError:
        print("경고: 상병코드별_평균입원일수.csv 파일을 찾을 수 없습니다. 기본값 7일을 사용합니다.")
        avg_length_of_stay_dict = {}

    # 병상가동률 계산
    bed_utilization = {}
    for hospital in beds_capacity.keys():
        if beds_capacity[hospital] > 0:
            # 환자 정보에 상병코드가 있으면 평균입원일수 곱해서 총재원일수 계산
            total_inpatients = 0
            total_stay_days = 0
            if hasattr(sim, 'patients_info') and isinstance(sim.patients_info, dict):
                for code, info in sim.patients_info.get(hospital, {}).get('diagnosis', {}).items():
                    code_3digit = str(code)[:3]
                    avg_los = avg_length_of_stay_dict.get(code_3digit, 7.0)
                    total_inpatients += info.get('count', 0)
                    total_stay_days += info.get('count', 0) * avg_los
            # 없으면 기존 방식
            if total_stay_days == 0:
                total_inpatients = sim.inpatient_count.get(hospital, 0)
                total_stay_days = total_inpatients * 7.0
            daily_inpatients = total_stay_days / 365
            bed_utilization[hospital] = (daily_inpatients / beds_capacity[hospital]) * 100
        else:
            bed_utilization[hospital] = 0
    
    # 병상당 의사 수
    bed_per_specialist = {}
    for hospital in beds_capacity.keys():
        if beds_capacity[hospital] > 0:
            bed_per_specialist[hospital] = specialists_capacity[hospital] / beds_capacity[hospital]
        else:
            bed_per_specialist[hospital] = 0
    
    # 2. 환자당 의료인력 지표
    # 의사당 입원환자수 (1일 평균)
    specialist_inpatient_ratio = {}
    for hospital in specialists_capacity.keys():
        if specialists_capacity[hospital] > 0:
            daily_inpatients = sim.inpatient_count[hospital] / 365
            specialist_inpatient_ratio[hospital] = daily_inpatients / specialists_capacity[hospital]
        else:
            specialist_inpatient_ratio[hospital] = 0
    
    # 의사당 외래환자수 (1일 평균)
    specialist_outpatient_ratio = {}
    for hospital in specialists_capacity.keys():
        if specialists_capacity[hospital] > 0:
            daily_outpatients = sim.outpatient_count[hospital] / 365
            specialist_outpatient_ratio[hospital] = daily_outpatients / specialists_capacity[hospital]
        else:
            specialist_outpatient_ratio[hospital] = 0
    
    # 3. 환자 비율 지표
    # 외래환자 대비 입원환자비
    inpatient_outpatient_ratio = {}
    for hospital in beds_capacity.keys():
        if sim.outpatient_count[hospital] > 0:
            inpatient_outpatient_ratio[hospital] = sim.inpatient_count[hospital] / sim.outpatient_count[hospital]
        else:
            inpatient_outpatient_ratio[hospital] = np.nan
    
    # 4. 의료 이용 지표
    # 고가도지표 (Costliness Index)
    total_costs = []
    for hospital in beds_capacity.keys():
        if hospital in patients_info:
            hospital_cost = (sim.inpatient_count[hospital] * patients_info[hospital]['cost_in'] + 
                            sim.outpatient_count[hospital] * patients_info[hospital]['cost_out'])
            total_costs.append(hospital_cost)
    
    national_avg_cost = np.mean(total_costs) if total_costs else 0
    
    ci_index = {}
    for hospital in beds_capacity.keys():
        if hospital in patients_info:
            hospital_cost = (sim.inpatient_count[hospital] * patients_info[hospital]['cost_in'] + 
                            sim.outpatient_count[hospital] * patients_info[hospital]['cost_out'])
            if national_avg_cost > 0:
                ci_index[hospital] = hospital_cost / national_avg_cost
            else:
                ci_index[hospital] = np.nan
        else:
            ci_index[hospital] = np.nan
    
    # 평균값 계산
    metrics = {
        '병상가동률(%)': np.mean(list(bed_utilization.values())),
        '병상당의사수': np.mean(list(bed_per_specialist.values())),
        '의사당입원환자수': np.mean(list(specialist_inpatient_ratio.values())),
        '의사당외래환자수': np.mean(list(specialist_outpatient_ratio.values())),
        '외래대비입원비율': np.nanmean(list(inpatient_outpatient_ratio.values())),
        '고가도지표': np.nanmean(list(ci_index.values())),
        '병상가동률_상세': bed_utilization,
        '병상당의사수_상세': bed_per_specialist,
        '의사당입원환자수_상세': specialist_inpatient_ratio,
        '의사당외래환자수_상세': specialist_outpatient_ratio,
        '외래대비입원비율_상세': inpatient_outpatient_ratio,
        '고가도지표_상세': ci_index
    }
    
    return metrics

def evaluate_against_national_standards(metrics):
    """
    전국 평균과 비교하여 평가
    """
    evaluation = {}
    
    # 전국 평균 기준 (평가지표.txt 참조)
    national_standards = {
        '병상가동률(%)': 67.43,  # 전국 종합병원 평균
        '병상당의사수': 0.1671,  # 전국 평균
        '의사당입원환자수': 4.0327,  # 연인원 기준
        '의사당외래환자수': 10.625,  # 연인원 기준
        '외래대비입원비율': 0.3795,  # 전국 평균
        '고가도지표': 1.0  # 전국 평균
    }
    
    for metric, value in metrics.items():
        if metric.endswith('_상세'):
            continue
            
        if metric in national_standards:
            national_avg = national_standards[metric]
            if national_avg > 0:
                ratio = value / national_avg
                if ratio > 1.2:
                    status = "높음 (전국평균 대비 {:.1%})".format(ratio)
                elif ratio < 0.8:
                    status = "낮음 (전국평균 대비 {:.1%})".format(ratio)
                else:
                    status = "적정 (전국평균 대비 {:.1%})".format(ratio)
            else:
                status = "평가불가"
        else:
            status = "기준없음"
            
        evaluation[metric] = {
            '시뮬레이션값': value,
            '전국평균': national_standards.get(metric, 'N/A'),
            '평가': status
        }
    
    return evaluation

def prepare_patients_info(year):
    # 기존 로직 유지 (진료비 기반 환자 수 추정)
    year_factor = 1 + (year - 2023) * 0.05
    cost_year_df = cost_df[cost_df['년도'] == 2023].copy()
    cost_year_df = cost_year_df.drop_duplicates(subset=["original_병원명", "original_진료과", "년도"], keep="first")
    cost_summary = cost_year_df.groupby('original_병원명')['original_진료비(천원)'].mean()
    cost_in = cost_summary.astype(float).multiply(0.7)
    cost_out = cost_summary.astype(float).multiply(0.3)
    base_patients = 50000
    patients_info = {}
    for hospital in cost_summary.index:
        cost_ratio = cost_summary[hospital] / cost_summary.mean() if cost_summary.mean() > 0 else 1
        total_patients = int(base_patients * cost_ratio * year_factor)
        inpatient_count = int(total_patients * 0.2)
        outpatient_count = total_patients - inpatient_count
        patients_info[hospital] = {
            'inpatient': inpatient_count,
            'outpatient': outpatient_count,
            'cost_in': cost_in.get(hospital, 0),
            'cost_out': cost_out.get(hospital, 0),
        }
    return patients_info

def calculate_evaluation_metrics(beds_capacity, specialists_capacity, patients_info, sim):
    bed_utilizations = []
    for hospital, capacity in beds_capacity.items():
        if capacity > 0:
            utilization = sim.bed_occupancy_time.get(hospital, 0) / (capacity * sim.env.now)
            bed_utilizations.append(utilization)
        else:
            bed_utilizations.append(0)
    avg_bed_utilization = np.mean(bed_utilizations) if bed_utilizations else 0

    bed_per_specialist = {}
    for h in beds_capacity.keys():
        beds = beds_capacity.get(h, 0)
        specs = specialists_capacity.get(h, 0)
        bed_per_specialist[h] = specs / beds if beds > 0 else 0

    specialist_inpatient_ratio = {}
    for h, info in patients_info.items():
        inpatient_daily = info['inpatient'] / 365
        specs = specialists_capacity.get(h, 0)
        specialist_inpatient_ratio[h] = inpatient_daily / specs if specs > 0 else 0

    specialist_outpatient_ratio = {}
    for h, info in patients_info.items():
        outpatient_daily = info['outpatient'] / 365
        specs = specialists_capacity.get(h, 0)
        specialist_outpatient_ratio[h] = outpatient_daily / specs if specs > 0 else 0

    inpatient_outpatient_ratio = {}
    for h, info in patients_info.items():
        inpatient = info['inpatient']
        outpatient = info['outpatient']
        inpatient_outpatient_ratio[h] = inpatient / outpatient if outpatient > 0 else np.nan

    total_costs = [info['cost_in'] + info['cost_out'] for info in patients_info.values()]
    national_avg_cost = np.mean(total_costs) if total_costs else 0

    ci_index = {}
    for h, info in patients_info.items():
        hospital_cost = info['cost_in'] + info['cost_out']
        ci_index[h] = hospital_cost / national_avg_cost if national_avg_cost > 0 else np.nan

    return {
        'avg_bed_utilization': avg_bed_utilization,
        'bed_per_specialist': bed_per_specialist,
        'specialist_inpatient_ratio': specialist_inpatient_ratio,
        'specialist_outpatient_ratio': specialist_outpatient_ratio,
        'inpatient_outpatient_ratio': inpatient_outpatient_ratio,
        'ci_index': ci_index
    }

def print_healthcare_evaluation(metrics, evaluation):
    """
    의료 평가지표 평가 결과를 출력
    """
    print("\n" + "="*60)
    print("의료 평가지표 평가 결과")
    print("="*60)
    
    for metric, eval_info in evaluation.items():
        print(f"\n{metric}:")
        print(f"  시뮬레이션값: {eval_info['시뮬레이션값']:.4f}")
        print(f"  전국평균: {eval_info['전국평균']}")
        print(f"  평가: {eval_info['평가']}")
    
    print("\n" + "="*60)
    print("평가지표 해석:")
    print("="*60)
    print("• 병상가동률: 85~90%가 이상적, 90% 이상은 병상 부족, 70% 이하는 과잉")
    print("• 병상당의사수: 전국평균 0.1671명/병상")
    print("• 의사당입원환자수: 전국평균 4.03명/일")
    print("• 의사당외래환자수: 전국평균 10.63명/일")
    print("• 외래대비입원비율: 전국평균 0.38")
    print("• 고가도지표: 1.0이 전국평균, 1.2 이상은 고가도 의료기관")

def run_scenario_1(start_year=2023, end_year=2027, sim_time=8760):
    results = []
    for year in range(start_year, end_year + 1):
        print(f"\n== 시나리오 1 - {year}년 ==")
        # 연도별 변화 반영 (year_factor)
        year_factor = 1 + (year - 2023) * 0.05
        # 병상/전문의 수는 통합 데이터 기반 + 연도별 증가율
        beds_capacity = {h: int(c * year_factor) for h, c in beds_capacity_dict.items()}
        specialists_capacity = {h: int(c * year_factor) for h, c in specialists_capacity_dict.items()}
        patients_info = prepare_patients_info(year)
        env = simpy.Environment()
        sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info)
        env.process(sim.patient_arrival())
        env.run(until=sim_time)
        total_patients = sim.patients_served + sim.patients_rejected
        rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
        avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0
        metrics = calculate_healthcare_metrics(beds_capacity, specialists_capacity, patients_info, sim)
        evaluation = evaluate_against_national_standards(metrics)
        
        print(f"총환자: {total_patients}, 거절률: {rejection_rate:.2f}%, 평균대기시간: {avg_wait:.2f}, 누적진료비: {sim.total_cost:.2f}")
        
        # 평가지표 출력
        print_healthcare_evaluation(metrics, evaluation)
        
        results.append({
            '연도': year,
            '총환자': total_patients,
            '거절환자': sim.patients_rejected,
            '거절률(%)': rejection_rate,
            '평균대기시간(시간)': avg_wait,
            '누적진료비': sim.total_cost,
            '평균병상가동률': metrics['병상가동률(%)'],
            '병상당의사수': metrics['병상당의사수'],
            '의사당입원환자수': metrics['의사당입원환자수'],
            '의사당외래환자수': metrics['의사당외래환자수'],
            '외래대비입원비율': metrics['외래대비입원비율'],
            '고가도지표': metrics['고가도지표']
        })
    return pd.DataFrame(results)

def run_scenario_2(base_year=2025, sim_time=8760, bed_factors=[0.8, 1.0, 1.2], spec_factors=[0.8, 1.0, 1.2]):
    results = []
    beds_capacity_base = beds_capacity_dict.copy()
    specialists_capacity_base = specialists_capacity_dict.copy()
    patients_info_base = prepare_patients_info(base_year)
    for bf in bed_factors:
        for sf in spec_factors:
            beds_capacity = {h: int(c * bf) for h, c in beds_capacity_base.items()}
            specialists_capacity = {h: int(c * sf) for h, c in specialists_capacity_base.items()}
            env = simpy.Environment()
            sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info_base)
            env.process(sim.patient_arrival())
            env.run(until=sim_time)
            total_patients = sim.patients_served + sim.patients_rejected
            rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
            avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0
            metrics = calculate_healthcare_metrics(beds_capacity, specialists_capacity, patients_info_base, sim)
            evaluation = evaluate_against_national_standards(metrics)
            
            print(f"[병상배율 {bf}, 전문의배율 {sf}] 총환자: {total_patients}, 거절률: {rejection_rate:.2f}%, 평균대기시간: {avg_wait:.2f}, 누적진료비: {sim.total_cost:.2f}")
            
            # 평가지표 출력
            print(f"  병상가동률: {metrics['병상가동률(%)']:.2f}%, 병상당의사수: {metrics['병상당의사수']:.4f}")
            
            results.append({
                '병상배율': bf,
                '전문의배율': sf,
                '총환자': total_patients,
                '거절환자': sim.patients_rejected,
                '거절률(%)': rejection_rate,
                '평균대기시간(시간)': avg_wait,
                '누적진료비': sim.total_cost,
                '평균병상가동률': metrics['병상가동률(%)'],
                '병상당의사수': metrics['병상당의사수'],
                '의사당입원환자수': metrics['의사당입원환자수'],
                '의사당외래환자수': metrics['의사당외래환자수'],
                '외래대비입원비율': metrics['외래대비입원비율'],
                '고가도지표': metrics['고가도지표']
            })
    return pd.DataFrame(results)

def run_scenario_3(year=2025, sim_time=8760, bed_changes=[-50, 0, 50], spec_changes=[-10, 0, 10]):
    results = []
    beds_capacity_base = beds_capacity_dict.copy()
    specialists_capacity_base = specialists_capacity_dict.copy()
    patients_info = prepare_patients_info(year)
    for bed_change in bed_changes:
        for spec_change in spec_changes:
            beds_capacity = {h: max(0, int(c + bed_change)) for h, c in beds_capacity_base.items()}
            specialists_capacity = {h: max(0, int(c + spec_change)) for h, c in specialists_capacity_base.items()}
            env = simpy.Environment()
            sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info)
            env.process(sim.patient_arrival())
            env.run(until=sim_time)
            total_patients = sim.patients_served + sim.patients_rejected
            rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
            avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0
            metrics = calculate_healthcare_metrics(beds_capacity, specialists_capacity, patients_info, sim)
            evaluation = evaluate_against_national_standards(metrics)
            
            print(f"[병상변화 {bed_change}, 전문의변화 {spec_change}] 총환자: {total_patients}, 거절률: {rejection_rate:.2f}%, 평균대기시간: {avg_wait:.2f}, 누적진료비: {sim.total_cost:.2f}")
            
            # 평가지표 출력
            print(f"  병상가동률: {metrics['병상가동률(%)']:.2f}%, 병상당의사수: {metrics['병상당의사수']:.4f}")
            
            results.append({
                '병상변화': bed_change,
                '전문의변화': spec_change,
                '총환자': total_patients,
                '거절환자': sim.patients_rejected,
                '거절률(%)': rejection_rate,
                '평균대기시간(시간)': avg_wait,
                '누적진료비': sim.total_cost,
                '평균병상가동률': metrics['병상가동률(%)'],
                '병상당의사수': metrics['병상당의사수'],
                '의사당입원환자수': metrics['의사당입원환자수'],
                '의사당외래환자수': metrics['의사당외래환자수'],
                '외래대비입원비율': metrics['외래대비입원비율'],
                '고가도지표': metrics['고가도지표']
            })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # 시나리오 1 실행 및 결과 저장
    df_scenario_1 = run_scenario_1(sim_time=8760)
    df_scenario_1.to_csv("simulation_results_model2/scenario1_result.csv", index=False)

    # 평가지표 및 평가 리포트 저장 함수
    def save_evaluation_report(metrics_list, evaluations_list, filename, sim_keys=None):
        rows = []
        for i, (metrics, evaluation) in enumerate(zip(metrics_list, evaluations_list)):
            key = sim_keys[i] if sim_keys else (i+1)
            for metric in evaluation:
                rows.append({
                    '시뮬레이션회차': key,
                    '지표명': metric,
                    '시뮬레이션값': evaluation[metric]['시뮬레이션값'],
                    '전국평균': evaluation[metric]['전국평균'],
                    '평가': evaluation[metric]['평가']
                })
        pd.DataFrame(rows).to_csv(filename, index=False)

    # 시나리오1 리포트 저장
    metrics_list = []
    evaluations_list = []
    sim_keys = []
    for year in range(2023, 2028):
        year_factor = 1 + (year - 2023) * 0.05
        beds_capacity = {h: int(c * year_factor) for h, c in beds_capacity_dict.items()}
        specialists_capacity = {h: int(c * year_factor) for h, c in specialists_capacity_dict.items()}
        patients_info = prepare_patients_info(year)
        env = simpy.Environment()
        sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info)
        env.process(sim.patient_arrival())
        env.run(until=8760)
        metrics = calculate_healthcare_metrics(beds_capacity, specialists_capacity, patients_info, sim)
        evaluation = evaluate_against_national_standards(metrics)
        metrics_list.append(metrics)
        evaluations_list.append(evaluation)
        sim_keys.append(year)
    save_evaluation_report(metrics_list, evaluations_list, "simulation_results_model2/scenario1_evaluation_report.csv", sim_keys)

    # 시나리오2 실행 및 결과 저장
    df_scenario_2 = run_scenario_2(sim_time=8760)
    df_scenario_2.to_csv('simulation_results_model2/scenario2_result.csv', index=False)
    # 시나리오2 리포트 저장 (병상배율, 전문의배율 조합별)
    metrics_list2 = []
    evaluations_list2 = []
    sim_keys2 = []
    bed_factors = [0.8, 1.0, 1.2]
    spec_factors = [0.8, 1.0, 1.2]
    patients_info_base = prepare_patients_info(2025)
    for bf in bed_factors:
        for sf in spec_factors:
            beds_capacity = {h: int(c * bf) for h, c in beds_capacity_dict.items()}
            specialists_capacity = {h: int(c * sf) for h, c in specialists_capacity_dict.items()}
            env = simpy.Environment()
            sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info_base)
            env.process(sim.patient_arrival())
            env.run(until=8760)
            metrics = calculate_healthcare_metrics(beds_capacity, specialists_capacity, patients_info_base, sim)
            evaluation = evaluate_against_national_standards(metrics)
            metrics_list2.append(metrics)
            evaluations_list2.append(evaluation)
            sim_keys2.append(f"bed_{bf}_spec_{sf}")
    save_evaluation_report(metrics_list2, evaluations_list2, "simulation_results_model2/scenario2_evaluation_report.csv", sim_keys2)

    # 시나리오3 실행 및 결과 저장
    df_scenario_3 = run_scenario_3(sim_time=8760)
    df_scenario_3.to_csv('simulation_results_model2/scenario3_result.csv', index=False)
    # 시나리오3 리포트 저장 (병상변화, 전문의변화 조합별)
    metrics_list3 = []
    evaluations_list3 = []
    sim_keys3 = []
    bed_changes = [-50, 0, 50]
    spec_changes = [-10, 0, 10]
    patients_info = prepare_patients_info(2025)
    for bed_change in bed_changes:
        for spec_change in spec_changes:
            beds_capacity = {h: max(0, int(c + bed_change)) for h, c in beds_capacity_dict.items()}
            specialists_capacity = {h: max(0, int(c + spec_change)) for h, c in specialists_capacity_dict.items()}
            env = simpy.Environment()
            sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info)
            env.process(sim.patient_arrival())
            env.run(until=8760)
            metrics = calculate_healthcare_metrics(beds_capacity, specialists_capacity, patients_info, sim)
            evaluation = evaluate_against_national_standards(metrics)
            metrics_list3.append(metrics)
            evaluations_list3.append(evaluation)
            sim_keys3.append(f"bedchg_{bed_change}_specchg_{spec_change}")
    save_evaluation_report(metrics_list3, evaluations_list3, "simulation_results_model2/scenario3_evaluation_report.csv", sim_keys3)
