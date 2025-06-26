import os
import pandas as pd

# 폴더 경로
folder_path = 'final_merged_data'

# 폴더 내 모든 파일 순회
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f'파일명: {filename}')
        try:
            df = pd.read_csv(file_path)
            # 앞 5개 열, 5개 행만 출력
            print(df.iloc[:5, :5])
        except Exception as e:
            print(f'파일 읽기 오류: {e}')
        print('-' * 40)
