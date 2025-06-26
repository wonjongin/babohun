import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정 
mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우용 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

# 데이터 불러오기
Medical = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/진료과정보.csv", encoding="utf-8-sig")

# 어떤 과가 존재하는지
department = Medical["진료과별"].dropna().unique()
# 어떤 과가 존재하는지(지역별로)
seoul_department = sorted(Medical[Medical["지역"] == "서울"]["진료과별"].dropna().unique())
gwangju_department = sorted(Medical[Medical["지역"] == "광주"]["진료과별"].dropna().unique())
daejeon_department = sorted(Medical[Medical["지역"] == "대전"]["진료과별"].dropna().unique())
incheon_department = sorted(Medical[Medical["지역"] == "인천"]["진료과별"].dropna().unique())
busan_department = sorted(Medical[Medical["지역"] == "부산"]["진료과별"].dropna().unique())
daegu_department = sorted(Medical[Medical["지역"] == "대구"]["진료과별"].dropna().unique())
#지역별 괸료과수
region_dept_count = Medical.groupby("지역")["진료과별"].nunique().reset_index()
region_dept_count = region_dept_count.sort_values(by="진료과별", ascending=False)
#지역-진료과별 데이터 정리
Medical["장비유"] = Medical["보유장비"].notna()
deduped = Medical.copy()
deduped["장비유정렬"] = Medical["보유장비"].notna().astype(int)
deduped = deduped.sort_values(by=["지역", "진료과별", "장비유정렬"], ascending=[True, True, False])
deduped_unique = deduped.drop_duplicates(subset=["지역", "진료과별"], keep="first")
region_dept_equipment = (
    Medical[Medical["보유장비"].notna()]
    .groupby(["지역", "진료과별"])["보유장비"]
    .count()
    .reset_index()
    .rename(columns={"보유장비": "보유장비 개수"}))
#시각화
regions = Medical["지역"].dropna().unique().tolist()
regions.append("전체")
for region in regions:
    if region == "전체":
        df = region_dept_equipment.groupby("진료과별")["보유장비 개수"].sum().reset_index()
        title = "전체 지역"
    else:
        df = region_dept_equipment[region_dept_equipment["지역"] == region][["진료과별", "보유장비 개수"]]
        title = f"{region} 지역"

    top5 = df.sort_values(by="보유장비 개수", ascending=False).head(5)
    bottom5 = df.sort_values(by="보유장비 개수", ascending=True).head(5)
    combined = pd.concat([top5, bottom5])

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        combined["진료과별"],
        combined["보유장비 개수"],
        color=["steelblue" if x in top5["진료과별"].values else "lightcoral" for x in combined["진료과별"]]
    )
    plt.title(f"{title} - 진료과별 보유장비 수 (상위 5 / 하위 5)")
    plt.xlabel("보유 장비 개수")
    plt.ylabel("진료과")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
