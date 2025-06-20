import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우용 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

bed = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/병상정보.csv", encoding="utf-8-sig")

#1-1 병원별 병상 수 합계
grouped_bed = bed.groupby("지역")["병상 합계"].sum()  #지역별로 병상 개수 합치기
bed_region = grouped_bed.reset_index()
bed_region = bed_region.sort_values(by="병상 합계", ascending=False)

# 1-1. 시각화
plt.figure(figsize=(10, 6))
plt.barh(bed_region["지역"], bed_region["병상 합계"], color='skyblue')
plt.xlabel("병상 수")
plt.ylabel("지역")
plt.title("병상 수 상위 지역")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 2. 병상 유형별 비율 분석
bed_types = [
    "1인실 병상", "2인실 병상", "3인실 병상", "4인실 병상", "5인실 병상", "6인실 병상", 
    "7인실 병상", "8인실 병상", "9인실 병상", "특실 병상", "격리실 병상", "국비 병상", 
    "사비 병상", "기타 병상"]

bedtyperegion = bed.groupby("지역")[bed_types].sum()
bedtyperegion["총합"] = bedtyperegion.sum(axis=1)
bedtyperegion["1인실 비율(%)"] = (bedtyperegion["1인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["2인실 비율(%)"] = (bedtyperegion["2인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["3인실 비율(%)"] = (bedtyperegion["3인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["4인실 비율(%)"] = (bedtyperegion["4인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["5인실 비율(%)"] = (bedtyperegion["5인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["6인실 비율(%)"] = (bedtyperegion["6인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["7인실 비율(%)"] = (bedtyperegion["7인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["8인실 비율(%)"] = (bedtyperegion["8인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["9인실 비율(%)"] = (bedtyperegion["9인실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["특실 비율(%)"]   = (bedtyperegion["특실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["격리실 비율(%)"] = (bedtyperegion["격리실 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["국비 비율(%)"]   = (bedtyperegion["국비 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["사비 비율(%)"]   = (bedtyperegion["사비 병상"] / bedtyperegion["총합"]) * 100
bedtyperegion["기타 비율(%)"]   = (bedtyperegion["기타 병상"] / bedtyperegion["총합"]) * 100

bed_type_percent_cols = [
    "1인실 비율(%)", "2인실 비율(%)", "3인실 비율(%)", "4인실 비율(%)", "5인실 비율(%)",
    "6인실 비율(%)", "7인실 비율(%)", "8인실 비율(%)", "9인실 비율(%)",
    "특실 비율(%)", "격리실 비율(%)", "국비 비율(%)", "사비 비율(%)", "기타 비율(%)"]

region = "서울"
data = bedtyperegion.loc[region, bed_type_percent_cols]
plt.figure(figsize=(6,6))
plt.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
plt.title(f"{region} 병상 유형 비율")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

region = "부산"
data = bedtyperegion.loc[region, bed_type_percent_cols]
plt.figure(figsize=(6,6))
plt.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
plt.title(f"{region} 병상 유형 비율")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

region = "대전"
data = bedtyperegion.loc[region, bed_type_percent_cols]
plt.figure(figsize=(6,6))
plt.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
plt.title(f"{region} 병상 유형 비율")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

region = "광주"
data = bedtyperegion.loc[region, bed_type_percent_cols]
plt.figure(figsize=(6,6))
plt.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
plt.title(f"{region} 병상 유형 비율")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

region = "대구"
data = bedtyperegion.loc[region, bed_type_percent_cols]
plt.figure(figsize=(6,6))
plt.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
plt.title(f"{region} 병상 유형 비율")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

region = "인천"
data = bedtyperegion.loc[region, bed_type_percent_cols]
plt.figure(figsize=(6,6))
plt.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
plt.title(f"{region} 병상 유형 비율")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
