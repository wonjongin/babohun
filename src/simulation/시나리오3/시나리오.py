import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import random

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class HospitalSpecializationSimulation:
    """특화 병원 운영 및 일반인 유입 유도 시뮬레이션"""
    
    def __init__(self):
        self.results = {}
        self.hospital_data = {}
        self.specialization_data = {}
        # 랜덤 시드 설정
        np.random.seed(42)
        random.seed(42)
        
    def load_model_data(self):
        """모델 예측 결과 데이터 로드"""
        print("=== 모델 예측 결과 데이터 로드 ===")
        
        # 모델 1: 연령지역진료과 예측 결과
        try:
            self.model1_df = pd.read_csv('model_results_연령지역_진료과/Stacking_prediction_results_detailed.csv')
            print(f"모델 1 데이터 로드 완료: {self.model1_df.shape}")
        except FileNotFoundError:
            print("경고: 모델 1 데이터 파일을 찾을 수 없습니다.")
            self.model1_df = None
        
        # 모델 2: 시계열 예측 결과 (3년)
        try:
            self.model2_df = pd.read_csv('analysis_data/병원별_진료과별_미래3년_예측결과.csv')
            print(f"모델 2 데이터 로드 완료: {self.model2_df.shape}")
        except FileNotFoundError:
            print("경고: 모델 2 데이터 파일을 찾을 수 없습니다.")
            self.model2_df = None
        
        # 모델 3: 진료비 예측 결과 (연령지역진료과)
        try:
            self.model3_df = pd.read_csv('model_results_v3_연령지역진료과_확장/prediction_results_2.csv')
            print(f"모델 3 데이터 로드 완료: {self.model3_df.shape}")
        except FileNotFoundError:
            print("경고: 모델 3 데이터 파일을 찾을 수 없습니다.")
            self.model3_df = None
        
        # 모델 4: 병상수 예측 결과
        try:
            self.model4_df = pd.read_csv('model_results_진료과별병상수_예측모델_연령지역진료과추가/hospital_bed_prediction_results_Ridge_gridcv.csv')
            print(f"모델 4 데이터 로드 완료: {self.model4_df.shape}")
        except FileNotFoundError:
            print("경고: 모델 4 데이터 파일을 찾을 수 없습니다.")
            self.model4_df = None
        
        # 모델 5: 전문의 예측 결과
        try:
            self.model5_df = pd.read_csv('model_results_진료과_전문의_연령지역진료과/predictions/ElasticNet_predictions.csv')
            print(f"모델 5 데이터 로드 완료: {self.model5_df.shape}")
        except FileNotFoundError:
            print("경고: 모델 5 데이터 파일을 찾을 수 없습니다.")
            self.model5_df = None
    
    def analyze_veteran_diseases(self):
        """보훈대상자 주요 질환 분석 (모델 1, 2 활용)"""
        print("\n=== 보훈대상자 주요 질환 분석 ===")
        
        if self.model1_df is not None:
            # 상병코드별 빈도 분석
            disease_counts = self.model1_df['disease_group'].value_counts()
            print(f"상위 10개 질환: {disease_counts.head(10)}")
            
            # 진료과별 질환 분포
            if '진료과' in self.model1_df.columns:
                dept_disease = self.model1_df.groupby('진료과')['disease_group'].value_counts()
                print(f"진료과별 주요 질환: {dept_disease.head(10)}")
            
            # 지역별 질환 분포
            if '지역' in self.model1_df.columns:
                region_disease = self.model1_df.groupby('지역')['disease_group'].value_counts()
                print(f"지역별 주요 질환: {region_disease.head(10)}")
        
        return disease_counts if self.model1_df is not None else None
    
    def analyze_high_cost_diseases(self):
        """고진료비 질환 경제성 분석 (모델 3 활용)"""
        print("\n=== 고진료비 질환 경제성 분석 ===")
        
        if self.model3_df is not None:
            # 원본 진료비 데이터가 있는 경우
            if 'original_진료비(천원)' in self.model3_df.columns:
                cost_analysis = self.model3_df.groupby('original_상병코드')['original_진료비(천원)'].agg(['mean', 'sum', 'count'])
                cost_analysis = cost_analysis.sort_values('mean', ascending=False)
                print(f"상위 10개 고진료비 질환: {cost_analysis.head(10)}")
                
                # 진료과별 평균 진료비
                if 'original_진료과' in self.model3_df.columns:
                    dept_cost = self.model3_df.groupby('original_진료과')['original_진료비(천원)'].mean().sort_values(ascending=False)
                    print(f"진료과별 평균 진료비: {dept_cost}")
                
                return cost_analysis
            else:
                print("진료비 데이터를 찾을 수 없습니다.")
        
        return None
    
    def analyze_resource_distribution(self):
        """지역별 자원 분배 분석 (모델 4, 5 활용)"""
        print("\n=== 지역별 자원 분배 분석 ===")
        
        resource_analysis = {}
        
        # 병상수 분석 (모델 4)
        if self.model4_df is not None:
            if '지역' in self.model4_df.columns and 'predicted_beds' in self.model4_df.columns:
                bed_distribution = self.model4_df.groupby('지역')['predicted_beds'].agg(['mean', 'sum'])
                print(f"지역별 병상수 분배: {bed_distribution}")
                resource_analysis['beds'] = bed_distribution
        
        # 전문의 분석 (모델 5)
        if self.model5_df is not None:
            if '지역' in self.model5_df.columns and 'predicted_doctors' in self.model5_df.columns:
                doctor_distribution = self.model5_df.groupby('지역')['predicted_doctors'].agg(['mean', 'sum'])
                print(f"지역별 전문의 분배: {doctor_distribution}")
                resource_analysis['doctors'] = doctor_distribution
        
        return resource_analysis
    
    def simulate_specialization_hospital(self, target_diseases, target_region, simulation_days=365):
        """특화 병원 운영 시뮬레이션"""
        print(f"\n=== 특화 병원 운영 시뮬레이션 ({target_region}, {target_diseases}) ===")
        
        # 기본 병원 설정
        hospital_config = {
            'name': f'{target_region}특화병원',
            'region': target_region,
            'specialization': target_diseases,
            'total_beds': 200,
            'total_doctors': 50,
            'veteran_ratio': 0.7,  # 보훈대상자 비율
            'general_ratio': 0.3,  # 일반인 비율
            'daily_patients': 100,
            'avg_stay_days': 7,
            'bed_utilization': 0.8
        }
        
        # 시뮬레이션 결과 저장
        daily_results = []
        
        # 시계열 트렌드 설정
        base_daily_patients = hospital_config['daily_patients']
        seasonal_factor = 1.0
        trend_factor = 1.0
        
        for day in range(simulation_days):
            # 계절성 요인 (겨울철 환자 증가)
            seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * day / 365)
            
            # 장기 트렌드 (점진적 증가)
            trend_factor = 1.0 + 0.001 * day
            
            # 랜덤 변동성
            random_factor = np.random.normal(1.0, 0.1)
            
            # 최종 일일 환자 수 계산
            adjusted_daily_patients = int(base_daily_patients * seasonal_factor * trend_factor * random_factor)
            adjusted_daily_patients = max(80, min(120, adjusted_daily_patients))  # 범위 제한
            
            # 보훈대상자와 일반인 비율에 약간의 변동성 추가
            veteran_ratio_variation = np.random.normal(hospital_config['veteran_ratio'], 0.05)
            veteran_ratio_variation = max(0.65, min(0.75, veteran_ratio_variation))
            
            daily_veteran = int(adjusted_daily_patients * veteran_ratio_variation)
            daily_general = adjusted_daily_patients - daily_veteran
            
            # 병상 가동률에 변동성 추가
            bed_utilization_variation = np.random.normal(hospital_config['bed_utilization'], 0.05)
            bed_utilization_variation = max(0.7, min(0.9, bed_utilization_variation))
            occupied_beds = int(hospital_config['total_beds'] * bed_utilization_variation)
            
            # 수익 계산 (보훈대상자 vs 일반인) - 약간의 변동성 추가
            veteran_revenue_per_patient = np.random.normal(50000, 5000)
            general_revenue_per_patient = np.random.normal(80000, 8000)
            
            veteran_revenue = daily_veteran * veteran_revenue_per_patient
            general_revenue = daily_general * general_revenue_per_patient
            total_revenue = veteran_revenue + general_revenue
            
            # 의료진 효율성 - 시간에 따른 개선
            base_efficiency = min(1.0, hospital_config['total_doctors'] / (daily_veteran + daily_general) * 0.1)
            efficiency_improvement = 1.0 + 0.0005 * day  # 시간에 따른 효율성 개선
            doctor_efficiency = min(1.0, base_efficiency * efficiency_improvement)
            
            # 특화 효과 - 시간에 따른 강화
            specialization_effect = 1.2 + 0.0003 * day  # 점진적 특화 효과 강화
            
            daily_result = {
                'day': day + 1,
                'veteran_patients': daily_veteran,
                'general_patients': daily_general,
                'total_patients': daily_veteran + daily_general,
                'occupied_beds': occupied_beds,
                'bed_utilization_rate': bed_utilization_variation,
                'veteran_revenue': veteran_revenue,
                'general_revenue': general_revenue,
                'total_revenue': total_revenue,
                'doctor_efficiency': doctor_efficiency,
                'specialization_effect': specialization_effect
            }
            
            daily_results.append(daily_result)
        
        # 결과 요약
        results_df = pd.DataFrame(daily_results)
        summary = {
            'total_revenue': results_df['total_revenue'].sum(),
            'avg_daily_patients': results_df['total_patients'].mean(),
            'avg_bed_utilization': results_df['bed_utilization_rate'].mean(),
            'veteran_ratio_maintained': results_df['veteran_patients'].sum() / results_df['total_patients'].sum(),
            'general_ratio_achieved': results_df['general_patients'].sum() / results_df['total_patients'].sum()
        }
        
        print(f"시뮬레이션 결과 요약:")
        print(f"- 총 수익: {summary['total_revenue']:,.0f}원")
        print(f"- 평균 일일 환자: {summary['avg_daily_patients']:.1f}명")
        print(f"- 평균 병상 가동률: {summary['avg_bed_utilization']:.1%}")
        print(f"- 보훈대상자 비율 유지: {summary['veteran_ratio_maintained']:.1%}")
        print(f"- 일반인 유입 비율: {summary['general_ratio_achieved']:.1%}")
        
        return results_df, summary
    
    def simulate_regional_policy(self, target_regions, simulation_days=365):
        """지역별 일반인 유입 정책 시뮬레이션"""
        print(f"\n=== 지역별 일반인 유입 정책 시뮬레이션 ===")
        
        regional_results = {}
        
        for region in target_regions:
            print(f"\n--- {region} 지역 정책 시뮬레이션 ---")
            
            # 지역별 정책 설정
            policy_config = {
                'region': region,
                'promotion_intensity': 0.8,  # 홍보 강도
                'accessibility_improvement': 0.6,  # 접근성 개선
                'service_quality': 0.7,  # 서비스 품질
                'existing_beds': 150,
                'existing_doctors': 30,
                'veteran_patients': 80,
                'general_patients': 20  # 초기 일반인 환자
            }
            
            # 정책 효과 계산
            policy_effect = (policy_config['promotion_intensity'] + 
                           policy_config['accessibility_improvement'] + 
                           policy_config['service_quality']) / 3
            
            # 일반인 유입 증가율
            general_increase_rate = policy_effect * 0.5  # 정책 효과의 50%
            
            daily_results = []
            
            # 지역별 특성에 따른 차별화
            region_factors = {
                '부산': {'growth_rate': 1.2, 'seasonal_impact': 0.15},
                '대구': {'growth_rate': 1.1, 'seasonal_impact': 0.12},
                '인천': {'growth_rate': 1.3, 'seasonal_impact': 0.18}
            }
            
            region_factor = region_factors.get(region, {'growth_rate': 1.0, 'seasonal_impact': 0.1})
            
            for day in range(simulation_days):
                # 일반인 환자 수 증가 (점진적 + 랜덤 변동)
                base_growth_factor = 1 + (general_increase_rate * (day / simulation_days))
                
                # 계절성 요인
                seasonal_factor = 1.0 + region_factor['seasonal_impact'] * np.sin(2 * np.pi * day / 365)
                
                # 랜덤 변동성
                random_factor = np.random.normal(1.0, 0.08)
                
                # 지역별 성장률 적용
                region_growth = region_factor['growth_rate'] + 0.0002 * day
                
                # 최종 성장 팩터
                final_growth_factor = base_growth_factor * seasonal_factor * random_factor * region_growth
                
                current_general = int(policy_config['general_patients'] * final_growth_factor)
                
                # 보훈대상자 비율 유지 (최소 60%)
                max_general = int(policy_config['veteran_patients'] * 0.67)  # 보훈대상자 대비 최대 67%
                current_general = min(current_general, max_general)
                current_general = max(15, current_general)  # 최소 15명 유지
                
                total_patients = policy_config['veteran_patients'] + current_general
                
                # 의료 접근성 개선 효과 (시간에 따른 점진적 개선)
                accessibility_bonus = policy_config['accessibility_improvement'] * 0.1 * (1 + 0.0005 * day)
                
                # 서비스 품질 향상 효과 (시간에 따른 점진적 개선)
                quality_bonus = policy_config['service_quality'] * 0.15 * (1 + 0.0003 * day)
                
                # 정책 효과성 (시간에 따른 변화)
                policy_effectiveness = policy_effect * (1 + 0.0001 * day)
                
                daily_result = {
                    'day': day + 1,
                    'region': region,
                    'veteran_patients': policy_config['veteran_patients'],
                    'general_patients': current_general,
                    'total_patients': total_patients,
                    'general_ratio': current_general / total_patients,
                    'accessibility_effect': accessibility_bonus,
                    'quality_effect': quality_bonus,
                    'policy_effectiveness': policy_effectiveness
                }
                
                daily_results.append(daily_result)
            
            regional_results[region] = pd.DataFrame(daily_results)
            
            # 지역별 결과 요약
            region_summary = {
                'final_general_ratio': daily_results[-1]['general_ratio'],
                'avg_total_patients': regional_results[region]['total_patients'].mean(),
                'policy_effectiveness': policy_effect
            }
            
            print(f"- 최종 일반인 비율: {region_summary['final_general_ratio']:.1%}")
            print(f"- 평균 총 환자 수: {region_summary['avg_total_patients']:.1f}명")
            print(f"- 정책 효과성: {region_summary['policy_effectiveness']:.1%}")
        
        return regional_results
    
    def generate_policy_recommendations(self):
        """정책 제안 생성"""
        print("\n=== 정책 제안 생성 ===")
        
        recommendations = {
            'specialization_strategy': {
                'title': '질환별 특화 병원 구축 전략',
                'description': '보훈대상자 주요 질환을 중심으로 한 특화 병원 운영',
                'target_diseases': ['심혈관질환', '호흡기질환', '근골격계질환'],
                'expected_benefits': ['전문성 강화', '진료 품질 향상', '브랜드 가치 증대']
            },
            'general_access_strategy': {
                'title': '일반인 진료 접근성 개선',
                'description': '의료자원이 부족한 지역 대상 보훈병원 일반인 진료 개방',
                'target_regions': ['지방 중소도시', '의료 취약 지역'],
                'implementation_methods': [
                    '홍보 강화',
                    '접근성 개선',
                    '맞춤형 서비스 제공'
                ]
            },
            'resource_optimization': {
                'title': '지역별 자원 최적 배분',
                'description': '기존 자원 내에서 효율적인 의료 서비스 제공',
                'focus_areas': ['병상 가동률 향상', '전문의 효율성 증대', '지역 간 격차 완화']
            }
        }
        
        print("📋 정책 제안 요약:")
        for key, rec in recommendations.items():
            print(f"\n🔹 {rec['title']}")
            print(f"   {rec['description']}")
            if 'target_diseases' in rec:
                print(f"   대상 질환: {', '.join(rec['target_diseases'])}")
            if 'target_regions' in rec:
                print(f"   대상 지역: {', '.join(rec['target_regions'])}")
            if 'expected_benefits' in rec:
                print(f"   기대 효과: {', '.join(rec['expected_benefits'])}")
        
        return recommendations
    
    def run_complete_simulation(self):
        """전체 시나리오 시뮬레이션 실행"""
        print("=== 시나리오 3: 특화 병원 운영 및 일반인 유입 유도 시뮬레이션 ===")
        
        # 1. 데이터 로드
        self.load_model_data()
        
        # 2. 보훈대상자 주요 질환 분석
        disease_analysis = self.analyze_veteran_diseases()
        
        # 3. 고진료비 질환 경제성 분석
        cost_analysis = self.analyze_high_cost_diseases()
        
        # 4. 지역별 자원 분배 분석
        resource_analysis = self.analyze_resource_distribution()
        
        # 5. 특화 병원 시뮬레이션
        target_diseases = ['심혈관질환', '호흡기질환']
        target_region = '서울'
        specialization_results, spec_summary = self.simulate_specialization_hospital(
            target_diseases, target_region, simulation_days=365
        )
        
        # 6. 지역별 정책 시뮬레이션
        target_regions = ['부산', '대구', '인천']
        regional_results = self.simulate_regional_policy(target_regions, simulation_days=365)
        
        # 7. 정책 제안 생성
        recommendations = self.generate_policy_recommendations()
        
        # 8. 결과 저장
        self.save_simulation_results(
            specialization_results, regional_results, 
            spec_summary, recommendations
        )
        
        print("\n✅ 시나리오 3 시뮬레이션 완료!")
        
        return {
            'specialization_results': specialization_results,
            'regional_results': regional_results,
            'spec_summary': spec_summary,
            'recommendations': recommendations
        }
    
    def save_simulation_results(self, specialization_results, regional_results, spec_summary, recommendations):
        """시뮬레이션 결과 저장"""
        print("\n=== 시뮬레이션 결과 저장 ===")
        
        # 결과 디렉토리 생성
        output_dir = "simulation_results_scenario3"
        os.makedirs(output_dir, exist_ok=True)
        
        # 특화 병원 시뮬레이션 결과 저장
        specialization_results.to_csv(f"{output_dir}/specialization_hospital_simulation.csv", 
                                    index=False, encoding='utf-8-sig')
        
        # 지역별 정책 시뮬레이션 결과 저장
        for region, results in regional_results.items():
            results.to_csv(f"{output_dir}/regional_policy_{region}_simulation.csv", 
                          index=False, encoding='utf-8-sig')
        
        # 요약 결과 저장
        summary_df = pd.DataFrame([spec_summary])
        summary_df.to_csv(f"{output_dir}/simulation_summary.csv", 
                         index=False, encoding='utf-8-sig')
        
        # 정책 제안 저장
        with open(f"{output_dir}/policy_recommendations.txt", 'w', encoding='utf-8') as f:
            f.write("=== 시나리오 3 정책 제안 ===\n\n")
            for key, rec in recommendations.items():
                f.write(f"🔹 {rec['title']}\n")
                f.write(f"   {rec['description']}\n")
                if 'target_diseases' in rec:
                    f.write(f"   대상 질환: {', '.join(rec['target_diseases'])}\n")
                if 'target_regions' in rec:
                    f.write(f"   대상 지역: {', '.join(rec['target_regions'])}\n")
                if 'expected_benefits' in rec:
                    f.write(f"   기대 효과: {', '.join(rec['expected_benefits'])}\n")
                f.write("\n")
        
        print(f"결과가 {output_dir} 디렉토리에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    # 시뮬레이션 객체 생성
    simulation = HospitalSpecializationSimulation()
    
    # 전체 시나리오 실행
    results = simulation.run_complete_simulation()
    
    return results

if __name__ == "__main__":
    main()
