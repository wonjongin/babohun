import os
import numpy as np
import subprocess
import time

# CUDA 환경 변수 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("=== GPU 테스트 시작 ===")

# 1. XGBoost GPU 테스트
print("\n1. XGBoost GPU 테스트")
try:
    import xgboost as xgb
    print(f"XGBoost 버전: {xgb.__version__}")
    
    # 간단한 데이터 생성
    X = np.random.random((1000, 10))
    y = np.random.randint(0, 3, 1000)
    
    # GPU 설정으로 모델 생성
    model = xgb.XGBClassifier(
        tree_method='gpu_hist',
        gpu_id=0,
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
    
    print("XGBoost GPU 모델 학습 시작...")
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    print(f"XGBoost 학습 시간: {end_time - start_time:.2f}초")
    print("✅ XGBoost GPU 테스트 완료")
    
except Exception as e:
    print(f"❌ XGBoost GPU 테스트 실패: {str(e)}")

# 2. LightGBM GPU 테스트
print("\n2. LightGBM GPU 테스트")
try:
    import lightgbm as lgb
    print(f"LightGBM 버전: {lgb.__version__}")
    
    # GPU 설정으로 모델 생성
    model = lgb.LGBMClassifier(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_estimators=100,
        max_depth=3,
        random_state=42,
        verbose=-1
    )
    
    print("LightGBM GPU 모델 학습 시작...")
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    print(f"LightGBM 학습 시간: {end_time - start_time:.2f}초")
    print("✅ LightGBM GPU 테스트 완료")
    
except Exception as e:
    print(f"❌ LightGBM GPU 테스트 실패: {str(e)}")

# 3. GPU 사용률 확인
print("\n3. GPU 사용률 확인")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split(',')
        print(f"GPU 사용률: {gpu_info[0]}%")
        print(f"GPU 메모리 사용: {gpu_info[1]} MB")
    else:
        print("GPU 사용률 확인 실패")
except:
    print("GPU 사용률 확인 실패")

print("\n=== GPU 테스트 완료 ===") 