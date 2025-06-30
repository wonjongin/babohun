import simpy
import pandas as pd
import numpy as np
import random

# --- 데이터 불러오기 ---
inpatient_df = pd.read_csv('analysis_data/병원별_진료과별_입원_미래3년_예측결과.csv')
combined_patient_df = pd.read_csv('analysis_data/병원별_진료과별_미래3년_예측결과.csv')
cost_df = pd.read_csv('model_results_진료과진료비_시계열/진료비_구간예측결과_시계열연계.csv')
bed_df = pd.read_csv('model_results_진료과별병상수_예측모델_시계열추가_3개년/hospital_bed_prediction_results_Ridge_gridcv.csv')
specialist_df = pd.read_csv('model_results_진료과_전문의/predictions/Ridge_predictions.csv')

class HospitalSim:
    def __init__(self, env, beds_capacity, specialists_capacity, patients_info):
        self.env = env
        self.beds = {h: simpy.Resource(env, capacity=c) for h, c in beds_capacity.items()}
        self.specialists = {h: simpy.Resource(env, capacity=c) for h, c in specialists_capacity.items()}
        self.patients_info = patients_info

        self.patients_served = 0
        self.patients_rejected = 0
        self.wait_times = []
        self.total_cost = 0

        # 병상별 누적 점유 시간 저장
        self.bed_occupancy_time = {hospital: 0 for hospital in beds_capacity}

    def patient_arrival(self):
        while True:
            for hospital, info in self.patients_info.items():
                inpatient_rate = info['inpatient'] / 8760  # 시간당 도착률
                outpatient_rate = info['outpatient'] / 8760
                cost_in = info['cost_in']
                cost_out = info['cost_out']

                num_inpatients = np.random.poisson(inpatient_rate)
                num_outpatients = np.random.poisson(outpatient_rate)

                for _ in range(num_inpatients):
                    self.env.process(self.patient_process(hospital, '입원(연인원)', cost_in))
                for _ in range(num_outpatients):
                    self.env.process(self.patient_process(hospital, '외래', cost_out))

            yield self.env.timeout(1)  # 1시간 단위

    def patient_process(self, hospital, patient_type, cost):
        arrival = self.env.now

        if patient_type == '입원(연인원)':
            if hospital not in self.beds or hospital not in self.specialists:
                self.patients_rejected += 1
                return
            with self.beds[hospital].request() as bed_req:
                result = yield bed_req | self.env.timeout(1)
                if bed_req not in result:
                    self.patients_rejected += 1
                    return
                start_time = self.env.now
                with self.specialists[hospital].request() as doc_req:
                    result = yield doc_req | self.env.timeout(1)
                    if doc_req not in result:
                        self.patients_rejected += 1
                        bed_req.release()
                        return
                    self.wait_times.append(self.env.now - arrival)
                    treatment_time = random.uniform(5, 10)
                    yield self.env.timeout(treatment_time)
                    self.total_cost += cost
                    self.patients_served += 1
                end_time = self.env.now
                self.bed_occupancy_time[hospital] += (end_time - start_time)
        else:  # 외래
            if hospital not in self.specialists:
                self.patients_rejected += 1
                return
            with self.specialists[hospital].request() as doc_req:
                result = yield doc_req | self.env.timeout(1)
                if doc_req not in result:
                    self.patients_rejected += 1
                    return
                self.wait_times.append(self.env.now - arrival)
                yield self.env.timeout(random.uniform(1, 3))
                self.total_cost += cost
                self.patients_served += 1

def prepare_patients_info(year):
    in_df = inpatient_df[inpatient_df['연도'] == year].copy()
    in_df['가중예측값'] = 0.2 * in_df['ARIMA예측'] + 0.3 * in_df['RF예측'] + 0.5 * in_df['XGB예측']
    inpatient_sum = in_df.groupby('병원명')['가중예측값'].sum()

    comb_df = combined_patient_df[combined_patient_df['연도'] == year].copy()
    comb_df['가중예측값'] = 0.2 * comb_df['ARIMA예측'] + 0.3 * comb_df['RF예측'] + 0.5 * comb_df['XGB예측']
    combined_sum = comb_df.groupby('병원명')['가중예측값'].sum()

    outpatient = (combined_sum - inpatient_sum).clip(lower=0)

    cost_year_df = cost_df[cost_df['연도'] == year].copy()
    cost_summary = cost_year_df.groupby(['병원명', '구분'])['가중예측값'].sum().unstack(fill_value=0)

    hospitals = sorted(set(inpatient_sum.index) & set(combined_sum.index) & set(cost_summary.index))
    patients_info = {}
    for h in hospitals:
        patients_info[h] = {
            'inpatient': inpatient_sum.get(h, 0),
            'outpatient': outpatient.get(h, 0),
            'cost_in': cost_summary.loc[h].get('입원(연인원)', 0),
            'cost_out': cost_summary.loc[h].get('외래', 0),
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

def run_scenario_1(start_year=2023, end_year=2027, sim_time=2):
    results = []
    for year in range(start_year, end_year + 1):
        print(f"\n== 시나리오 1 - {year}년 ==")
        beds_year = bed_df[bed_df['연도'] == year].set_index('병원명')
        specs_year = specialist_df[specialist_df['연도'] == year].set_index('병원명')

        beds_capacity = beds_year['병상_총_예측'].to_dict()
        specialists_capacity = specs_year['y_predicted'].astype(int).to_dict()

        patients_info = prepare_patients_info(year)

        env = simpy.Environment()
        sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info)
        env.process(sim.patient_arrival())
        env.run(until=sim_time)

        total_patients = sim.patients_served + sim.patients_rejected
        rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
        avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0

        metrics = calculate_evaluation_metrics(beds_capacity, specialists_capacity, patients_info, sim)

        print(f"총환자: {total_patients}, 거절률: {rejection_rate:.2f}%, 평균대기시간: {avg_wait:.2f}, 누적진료비: {sim.total_cost:.2f}")
        print(f"평균 병상 가동률: {metrics['avg_bed_utilization']:.3f}")
        print("병상당 의사 수:")
        for h, v in metrics['bed_per_specialist'].items():
            print(f"  {h}: {v:.3f}")
        print("의사당 일평균 입원환자 수:")
        for h, v in metrics['specialist_inpatient_ratio'].items():
            print(f"  {h}: {v:.2f}")
        print("의사당 일평균 외래환자 수:")
        for h, v in metrics['specialist_outpatient_ratio'].items():
            print(f"  {h}: {v:.2f}")
        print("외래 대비 입원환자 비율:")
        for h, v in metrics['inpatient_outpatient_ratio'].items():
            print(f"  {h}: {v:.2f}")
        print("고가도 지표 (CI):")
        for h, v in metrics['ci_index'].items():
            print(f"  {h}: {v:.3f}")

        results.append({
            '연도': year,
            '총환자': total_patients,
            '거절환자': sim.patients_rejected,
            '거절률(%)': rejection_rate,
            '평균대기시간(시간)': avg_wait,
            '누적진료비': sim.total_cost,
            '평균병상가동률': metrics['avg_bed_utilization'],
        })

    return pd.DataFrame(results)

def run_scenario_2(base_year=2025, sim_time=2, bed_factors=[0.8, 1.0, 1.2], spec_factors=[0.8, 1.0, 1.2]):
    results = []
    beds_year = bed_df[bed_df['연도'] == base_year].set_index('병원명')
    specs_year = specialist_df[specialist_df['연도'] == base_year].set_index('병원명')
    patients_info_base = prepare_patients_info(base_year)

    for bf in bed_factors:
        for sf in spec_factors:
            beds_capacity = {h: int(c * bf) for h, c in beds_year['병상_총_예측'].items()}
            specialists_capacity = {h: int(c * sf) for h, c in specs_year['y_predicted'].items()}

            env = simpy.Environment()
            sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info_base)
            env.process(sim.patient_arrival())
            env.run(until=sim_time)

            total_patients = sim.patients_served + sim.patients_rejected
            rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
            avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0

            metrics = calculate_evaluation_metrics(beds_capacity, specialists_capacity, patients_info_base, sim)

            print(f"[병상배율 {bf}, 전문의배율 {sf}] 총환자: {total_patients}, 거절률: {rejection_rate:.2f}%, 평균대기시간: {avg_wait:.2f}, 누적진료비: {sim.total_cost:.2f}")
            print(f"평균 병상 가동률: {metrics['avg_bed_utilization']:.3f}")

            results.append({
                '병상배율': bf,
                '전문의배율': sf,
                '총환자': total_patients,
                '거절환자': sim.patients_rejected,
                '거절률(%)': rejection_rate,
                '평균대기시간(시간)': avg_wait,
                '누적진료비': sim.total_cost,
                '평균병상가동률': metrics['avg_bed_utilization'],
            })

    return pd.DataFrame(results)


def run_scenario_3(year=2025, sim_time=2, bed_changes=[-50, 0, 50], spec_changes=[-10, 0, 10]): # sim_time이 시뮬 돌리는 시간(단위: 1시간)
    results = []
    beds_year = bed_df[bed_df['연도'] == year].set_index('병원명')
    specs_year = specialist_df[specialist_df['연도'] == year].set_index('병원명')
    patients_info = prepare_patients_info(year)

    for bed_change in bed_changes:
        for spec_change in spec_changes:
            beds_capacity = {h: max(0, int(c + bed_change)) for h, c in beds_year['병상_총_예측'].items()}
            specialists_capacity = {h: max(0, int(c + spec_change)) for h, c in specs_year['y_predicted'].items()}

            env = simpy.Environment()
            sim = HospitalSim(env, beds_capacity, specialists_capacity, patients_info)
            env.process(sim.patient_arrival())
            env.run(until=sim_time)

            total_patients = sim.patients_served + sim.patients_rejected
            rejection_rate = (sim.patients_rejected / total_patients * 100) if total_patients else 0
            avg_wait = np.mean(sim.wait_times) if sim.wait_times else 0

            metrics = calculate_evaluation_metrics(beds_capacity, specialists_capacity, patients_info, sim)

            print(f"[병상변화 {bed_change}, 전문의변화 {spec_change}] 총환자: {total_patients}, 거절률: {rejection_rate:.2f}%, 평균대기시간: {avg_wait:.2f}, 누적진료비: {sim.total_cost:.2f}")
            print(f"평균 병상 가동률: {metrics['avg_bed_utilization']:.3f}")

            results.append({
                '병상변화량': bed_change,
                '전문의변화량': spec_change,
                '총환자': total_patients,
                '거절환자': sim.patients_rejected,
                '거절률(%)': rejection_rate,
                '평균대기시간(시간)': avg_wait,
                '누적진료비': sim.total_cost,
                '평균병상가동률': metrics['avg_bed_utilization'],
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df_scenario_1 = run_scenario_1(sim_time=2)
    df_scenario_2 = run_scenario_2(sim_time=2)
    df_scenario_3 = run_scenario_3(sim_time=2)

    df_scenario_1.to_csv('scenario1_results.csv', index=False)
    df_scenario_2.to_csv('scenario2_results.csv', index=False)
    df_scenario_3.to_csv('scenario3_results.csv', index=False)
