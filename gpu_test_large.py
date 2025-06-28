import os
import numpy as np
import subprocess
import time

# CUDA 환경 변수 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("=== 대용량 데이터 GPU 테스트 시작 ===")

# 더 큰 데이터 생성 (GPU 사용 강제)
print("대용량 데이터 생성 중...")
X = np.random.random((50000, 50))  # 5만개 샘플, 50개 피처
y = np.random.randint(0, 10, 50000)  # 10개 클래스
print(f"데이터 크기: {X.shape}")

# 1. XGBoost GPU 테스트 (대용량 데이터)
print("\n1. XGBoost GPU 테스트 (대용량 데이터)")
try:
    import xgboost as xgb
    print(f"XGBoost 버전: {xgb.__version__}")
    
    # GPU 사용률 확인 (학습 전)
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split(',')
        print(f"학습 전 GPU 사용률: {gpu_info[0]}%, 메모리: {gpu_info[1]} MB")
    
    # 수정된 GPU 설정으로 모델 생성
    model = xgb.XGBClassifier(
        tree_method="hist",
        device="cuda",
        n_estimators=500,  # 더 많은 트리
        max_depth=8,  # 더 깊은 트리
        learning_rate=0.1,
        random_state=42
    )
    
    print("XGBoost GPU 모델 학습 시작...")
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    print(f"XGBoost 학습 시간: {end_time - start_time:.2f}초")
    
    # GPU 사용률 확인 (학습 후)
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split(',')
        print(f"학습 후 GPU 사용률: {gpu_info[0]}%, 메모리: {gpu_info[1]} MB")
    
    print("✅ XGBoost GPU 테스트 완료")
    
except Exception as e:
    print(f"❌ XGBoost GPU 테스트 실패: {str(e)}")

# 2. LightGBM GPU 테스트 (대용량 데이터)
print("\n2. LightGBM GPU 테스트 (대용량 데이터)")
try:
    import lightgbm as lgb
    print(f"LightGBM 버전: {lgb.__version__}")
    
    # GPU 사용률 확인 (학습 전)
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split(',')
        print(f"학습 전 GPU 사용률: {gpu_info[0]}%, 메모리: {gpu_info[1]} MB")
    
    # 수정된 GPU 설정으로 모델 생성
    model = lgb.LGBMClassifier(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_estimators=500,  # 더 많은 트리
        max_depth=8,  # 더 깊은 트리
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    
    print("LightGBM GPU 모델 학습 시작...")
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    print(f"LightGBM 학습 시간: {end_time - start_time:.2f}초")
    
    # GPU 사용률 확인 (학습 후)
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split(',')
        print(f"학습 후 GPU 사용률: {gpu_info[0]}%, 메모리: {gpu_info[1]} MB")
    
    print("✅ LightGBM GPU 테스트 완료")
    
except Exception as e:
    print(f"❌ LightGBM GPU 테스트 실패: {str(e)}")

print("\n=== 대용량 데이터 GPU 테스트 완료 ===") 