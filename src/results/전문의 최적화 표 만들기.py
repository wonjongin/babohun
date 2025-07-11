import pandas as pd

# 데이터 경로
input_path = '../../optimization_results_전문의_분배_최적화/전문의_분배_최적화_결과.csv'
output_top_path = '전문의_분배_최적화_변화량_Top10.csv'
output_bottom_path = '전문의_분배_최적화_변화량_Bottom10.csv'

# 데이터 불러오기
df = pd.read_csv(input_path)

# 변화량 기준 정렬 (내림차순: 상위, 오름차순: 하위)
df_top10 = df.sort_values('변화량', ascending=False).head(10)
df_bottom10 = df.sort_values('변화량', ascending=True).head(10)

# CSV로 저장
df_top10.to_csv(output_top_path, index=False)
df_bottom10.to_csv(output_bottom_path, index=False)

print(f"상위 10개 결과 저장: {output_top_path}")
print(f"하위 10개 결과 저장: {output_bottom_path}")
