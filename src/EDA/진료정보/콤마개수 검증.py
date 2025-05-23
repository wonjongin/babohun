import os
import pandas as pd
import numpy as np

def check_csv_comma_count(filepath, encoding='euc-kr'):
    with open(filepath, encoding=encoding) as f:
        lines = f.readlines()
    
    # 각 행의 콤마 개수 세기
    comma_counts = [line.count(',') for line in lines]
    wrong_files = []
    # 가장 많이 나온 콤마 개수를 표준으로 간주
    from collections import Counter
    count_freq = Counter(comma_counts)
    standard = count_freq.most_common(1)[0][0]
    
    print(f"표준 콤마 개수: {standard}")
    
    # 표준과 다른 행 출력
    for idx, (line, cnt) in enumerate(zip(lines, comma_counts), 1):
        if cnt != standard:
            wrong_files.append(filepath)
            # print(f"행 {idx}: 콤마 {cnt}개 → {line.strip()}")
    
    ll = np.unique(wrong_files)
    print(f"파일 {ll}에서 {len(ll)}개의 행이 표준과 다릅니다.")
    print("검증 완료")

file_list = os.listdir('data/진료정보')

for file in file_list:
    if file.endswith('.csv'):
        file_path = os.path.join('data/진료정보', file)
        print(f"Processing {file}...")
        try:
            df = pd.read_csv(file_path, encoding='euc-kr')
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        # check_csv_comma_count(file_path, encoding='euc-kr')
