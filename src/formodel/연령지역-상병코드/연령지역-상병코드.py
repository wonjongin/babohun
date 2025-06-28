import pandas as pd
ekqlseh=pd.read_csv("C:/Users/jenny/babohun/new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv",encoding="utf-8-sig")

ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '연인원'] = \
    ekqlseh.loc[ekqlseh['구분'].str.contains('외래'), '실인원']

mask = ekqlseh['구분'] == '입원(실인원)'
rows_to_drop = ekqlseh[ekqlseh['구분'] == '입원(실인원)'].index
df_dropped = ekqlseh.drop(index=rows_to_drop)

cols_to_drop = ['순위', '상병명', '실인원', '진료비(천원)']
df_result = df_dropped.drop(columns=cols_to_drop)

df_result.to_csv(
    "C:/Users/jenny/babohun/new_merged_data/df_result.csv",
    index=False,
    encoding="utf-8-sig")
