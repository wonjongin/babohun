import pandas as pd


regions = ['중앙', '부산', '인천', '대구', '대전', '광주']
regions_df = {}

for region in regions:
    regions_df[region] = pd.read_csv(f"data/연도별 진료인원 UTF8/한국보훈복지의료공단_년도별 국가유공자 진료인원_{region}보훈병원_20231231.csv")
    # regions_df[region]['년도'] = 2023
    if region != '중앙':
        regions_df[region]['지역'] = region
    else:
        regions_df[region]['지역'] = '서울'



# Merge all dataframes into one
merged_df = pd.concat(regions_df.values(), ignore_index=True)
# Save the merged dataframe to a CSV file
merged_df['구분'] = merged_df['구분'].replace({
"1_국비": "국비",
"2_감면": "감면",
"3_일반": "일반",
})

cols = "년도,지역,구분,자연인,연인원".split(",")
merged_df = merged_df[cols]

merged_df.to_csv('new_merged_data/연도별 진료인원_병원통합.csv', index=False, encoding='utf-8')