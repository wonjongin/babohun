import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 결과 폴더 생성
results_dir = Path('results/모델3예측결과')
results_dir.mkdir(parents=True, exist_ok=True)

# 데이터 로드
print("데이터 로딩 중...")
future_data = pd.read_csv('analysis_data/병원별_진료과별_미래3년_예측결과.csv')
existing_predictions = pd.read_csv('model_results_진료과_전문의_연령지역진료과/predictions/ElasticNet_predictions.csv')

# ElasticNet 모델 로드
print("모델 로딩 중...")
model_path = 'model_results_진료과_전문의_연령지역진료과/models/ElasticNet_model.pkl'
model = joblib.load(model_path)

# 기존 예측 데이터에서 필요한 컬럼들 추출
print("데이터 전처리 중...")
existing_features = existing_predictions.drop(['전문의수', 'y_actual', 'y_predicted', 'prediction_error', 'absolute_error', 'model'], axis=1)

# 미래 데이터를 기존 모델 입력 형태로 변환
def prepare_future_data(future_df, existing_features):
    """미래 데이터를 모델 입력 형태로 변환"""
    # 기존 특성 컬럼들을 가져옴 (전문의수 제외)
    feature_columns = [col for col in existing_features.columns if col not in ['전문의수', 'y_actual', 'y_predicted', 'prediction_error', 'absolute_error', 'model']]
    
    processed_data = []
    
    for _, row in future_df.iterrows():
        hospital = row['병원']
        dept = row['진료과']
        arima_pred = row['ARIMA예측']
        
        # 기존 데이터의 첫 번째 행을 템플릿으로 사용
        features = existing_features.iloc[0].copy()
        
        # 미래 예측 데이터로 업데이트
        features['ARIMA예측'] = arima_pred
        features['RF예측'] = row['RF예측']
        features['XGB예측'] = row['XGB예측']
        features['실제값'] = 0.0  # 미래 데이터는 실제값이 없음
        features['실제값_log'] = 0.0
        
        # 예측값 관련 특성들 업데이트
        features['예측값_평균'] = (arima_pred + row['RF예측'] + row['XGB예측']) / 3
        features['예측값_표준편차'] = np.std([arima_pred, row['RF예측'], row['XGB예측']])
        features['예측값_최대'] = max(arima_pred, row['RF예측'], row['XGB예측'])
        features['예측값_최소'] = min(arima_pred, row['RF예측'], row['XGB예측'])
        features['가중예측값'] = arima_pred * 0.4 + row['RF예측'] * 0.3 + row['XGB예측'] * 0.3
        
        # 로그 변환
        features['ARIMA예측_log'] = np.log(arima_pred + 1)
        features['RF예측_log'] = np.log(row['RF예측'] + 1)
        features['XGB예측_log'] = np.log(row['XGB예측'] + 1)
        
        # 병상당 예측 환자수
        features['병상당예측환자수'] = arima_pred / features['총병상수']
        
        # 병원명 더미 변수 업데이트
        features['병원명_대구'] = 1.0 if hospital == '대구' else 0.0
        features['병원명_대전'] = 1.0 if hospital == '대전' else 0.0
        features['병원명_부산'] = 1.0 if hospital == '부산' else 0.0
        features['병원명_서울'] = 1.0 if hospital == '중앙' else 0.0
        features['병원명_인천'] = 1.0 if hospital == '인천' else 0.0
        
        # 진료과 더미 변수 초기화 후 해당 진료과만 1로 설정
        dept_columns = [col for col in feature_columns if col.startswith('진료과_')]
        for col in dept_columns:
            features[col] = 0.0
        
        # 해당 진료과 매핑
        dept_mapping = {
            '가정의학과': '진료과_내과계열',  # 가정의학과는 내과계열로 매핑
            '내과': '진료과_내과계열',
            '산부인과': '진료과_외과계열',
            '소아청소년과': '진료과_소아계열',
            '신경과': '진료과_내과계열',
            '신경외과': '진료과_외과계열',
            '안과': '진료과_외과계열',
            '외과': '진료과_외과계열',
            '이비인후과': '진료과_외과계열',
            '재활의학과': '진료과_외과계열',
            '정형외과': '진료과_외과계열',
            '치과': '진료과_외과계열',
            '피부과': '진료과_외과계열',
            '응급의학과': '진료과_외과계열',
            '정신건강의학과': '진료과_정신계열',
            '핵의학과': '진료과_내과계열'
        }
        
        if dept in dept_mapping:
            features[dept_mapping[dept]] = 1.0
        
        processed_data.append(features)
    
    return pd.DataFrame(processed_data)

# 미래 데이터 전처리
print("미래 데이터 전처리 중...")
future_processed = prepare_future_data(future_data, existing_features)

# 기존 예측과 미래 예측 비교를 위해 기존 데이터도 준비
print("기존 예측 데이터 준비 중...")
existing_features_clean = existing_features.drop('전문의수', axis=1, errors='ignore').copy()

# 모델 예측 실행
print("미래 데이터로 예측 중...")
future_predictions = model.predict(future_processed)

print("기존 데이터로 예측 중...")
existing_predictions_new = model.predict(existing_features_clean)

# 결과 데이터프레임 생성
print("결과 데이터프레임 생성 중...")

# 미래 예측 결과
future_results = future_data.copy()
future_results['예측전문의수'] = future_predictions
future_results['예측연도'] = future_results['예측연도'].astype(int)

# 기존 예측 결과 (실제값과 비교)
existing_results = existing_predictions[['전문의수', 'y_actual', 'y_predicted']].copy()
existing_results['예측오차'] = existing_results['y_predicted'] - existing_results['y_actual']
existing_results['절대오차'] = abs(existing_results['예측오차'])

# 결과 저장
print("결과 저장 중...")

# 미래 예측 결과 저장
future_results.to_csv(results_dir / '미래3년_전문의수_예측결과.csv', index=False, encoding='utf-8-sig')

# 기존 예측 결과 요약
existing_summary = existing_results.describe()
existing_summary.to_csv(results_dir / '기존예측_성능요약.csv', encoding='utf-8-sig')

# 병원별, 진료과별 미래 예측 요약
future_summary = future_results.groupby(['병원', '진료과']).agg({
    '예측전문의수': ['mean', 'std', 'min', 'max'],
    'ARIMA예측': 'mean',
    'RF예측': 'mean',
    'XGB예측': 'mean'
}).round(2)
future_summary.columns = ['예측전문의수_평균', '예측전문의수_표준편차', '예측전문의수_최소', '예측전문의수_최대',
                         'ARIMA예측_평균', 'RF예측_평균', 'XGB예측_평균']
future_summary.to_csv(results_dir / '미래예측_병원별진료과별_요약.csv', encoding='utf-8-sig')

# 시각화
print("시각화 생성 중...")

# 1. 연도별 예측 전문의 수 변화
plt.figure(figsize=(15, 8))
for hospital in future_results['병원'].unique():
    hospital_data = future_results[future_results['병원'] == hospital]
    plt.plot(hospital_data['예측연도'], hospital_data['예측전문의수'], 
             marker='o', label=hospital, linewidth=2, markersize=6)

plt.title('연도별 병원별 예측 전문의 수 변화 (2024-2026)', fontsize=16, fontweight='bold')
plt.xlabel('연도', fontsize=12)
plt.ylabel('예측 전문의 수', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / '연도별_병원별_전문의수_변화.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 진료과별 예측 전문의 수 분포 (2024년)
plt.figure(figsize=(16, 10))
year_2024 = future_results[future_results['예측연도'] == 2024]
pivot_data = year_2024.pivot(index='진료과', columns='병원', values='예측전문의수')

sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': '예측 전문의 수'})
plt.title('2024년 진료과별 병원별 예측 전문의 수', fontsize=16, fontweight='bold')
plt.xlabel('병원', fontsize=12)
plt.ylabel('진료과', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(results_dir / '2024년_진료과별_병원별_전문의수_히트맵.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 기존 vs 미래 예측 비교 (2024년)
plt.figure(figsize=(12, 8))
year_2024_avg = year_2024.groupby('진료과')['예측전문의수'].mean()
year_2024_avg_sorted = year_2024_avg.sort_values(ascending=False)

plt.barh(range(len(year_2024_avg_sorted)), year_2024_avg_sorted.values, color='skyblue', alpha=0.7)
plt.yticks(range(len(year_2024_avg_sorted)), list(year_2024_avg_sorted.index))
plt.xlabel('평균 예측 전문의 수', fontsize=12)
plt.title('2024년 진료과별 평균 예측 전문의 수', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(results_dir / '2024년_진료과별_평균_전문의수.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 예측 모델별 비교 (ARIMA, RF, XGB vs ElasticNet)
plt.figure(figsize=(15, 8))
sample_data = year_2024.head(20)  # 샘플 데이터

x = range(len(sample_data))
plt.plot(x, sample_data['ARIMA예측'], 'o-', label='ARIMA 환자수 예측', alpha=0.7)
plt.plot(x, sample_data['RF예측'], 's-', label='RF 환자수 예측', alpha=0.7)
plt.plot(x, sample_data['XGB예측'], '^-', label='XGB 환자수 예측', alpha=0.7)
plt.plot(x, sample_data['예측전문의수'], 'd-', label='ElasticNet 전문의수 예측', linewidth=2)

plt.title('환자수 예측 vs 전문의수 예측 비교 (2024년 샘플)', fontsize=16, fontweight='bold')
plt.xlabel('샘플 인덱스', fontsize=12)
plt.ylabel('예측값', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / '환자수_vs_전문의수_예측_비교.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 기존 예측 성능 분석
plt.figure(figsize=(12, 8))
plt.scatter(existing_results['y_actual'], existing_results['y_predicted'], alpha=0.6)
plt.plot([existing_results['y_actual'].min(), existing_results['y_actual'].max()], 
         [existing_results['y_actual'].min(), existing_results['y_actual'].max()], 
         'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('실제 전문의 수', fontsize=12)
plt.ylabel('예측 전문의 수', fontsize=12)
plt.title('기존 예측 모델 성능: 실제값 vs 예측값', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / '기존예측_실제값_vs_예측값.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 모든 결과가 {results_dir} 폴더에 저장되었습니다!")
print(f"📊 생성된 파일들:")
print(f"   - 미래3년_전문의수_예측결과.csv")
print(f"   - 기존예측_성능요약.csv")
print(f"   - 미래예측_병원별진료과별_요약.csv")
print(f"   - 연도별_병원별_전문의수_변화.png")
print(f"   - 2024년_진료과별_병원별_전문의수_히트맵.png")
print(f"   - 2024년_진료과별_평균_전문의수.png")
print(f"   - 환자수_vs_전문의수_예측_비교.png")
print(f"   - 기존예측_실제값_vs_예측값.png")
