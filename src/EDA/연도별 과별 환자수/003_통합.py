import pandas as pd

# 파일 경로
infile_in = 'new_merged_data/연도별진료과별입원환자수.csv'
infile_out = 'new_merged_data/연도별진료과별외래환자수.csv'
outfile = 'new_merged_data/연도별진료과별_입원외래_통합환자수.csv'

# 데이터 불러오기
df_in = pd.read_csv(infile_in)
df_out = pd.read_csv(infile_out)

# 컬럼명 통일(혹시 모를 공백 등)
df_in.columns = df_in.columns.str.strip()
df_out.columns = df_out.columns.str.strip()

# key 컬럼 통일(str로 변환)
for col in ['연도', '병원', '진료과']:
    df_in[col] = df_in[col].astype(str)
    df_out[col] = df_out[col].astype(str)

# merge (outer join, 없는 값은 0)
df_merged = pd.merge(
    df_in, df_out,
    on=['연도', '병원', '진료과'],
    how='outer',
    suffixes=('_입원', '_외래')
)

# 결측치 0으로
for col in ['전체환자수_입원', '전체환자수_외래']:
    if col in df_merged.columns:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
    else:
        df_merged[col] = 0

# 합계 컬럼 추가
df_merged['전체환자수_합계'] = df_merged['전체환자수_입원'] + df_merged['전체환자수_외래']

# 정수형으로 변환
for col in ['전체환자수_입원', '전체환자수_외래', '전체환자수_합계']:
    df_merged[col] = df_merged[col].astype(int)

# 저장
cols = ['연도', '병원', '진료과', '전체환자수_입원', '전체환자수_외래', '전체환자수_합계']
df_merged[cols].to_csv(outfile, index=False, encoding='utf-8')
print(f'저장 완료: {outfile}')
