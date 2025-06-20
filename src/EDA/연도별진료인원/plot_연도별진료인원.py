import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정 
mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우용 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

peopletreated_year = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/연도별 진료인원.csv", encoding="utf-8-sig")
publichealth_year = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/연도별 공공의료병상 활성화 현황_2019.csv", encoding="utf-8-sig")

#연도별 진료인원 변화 추이
peopletreated_year_summary = peopletreated_year.groupby("년도")[["자연인", "연인원"]].sum().reset_index()
# 시각화
plt.figure(figsize=(8, 5))
plt.plot(peopletreated_year_summary["년도"], peopletreated_year_summary["자연인"], marker='o', label="실인원")
plt.plot(peopletreated_year_summary["년도"], peopletreated_year_summary["연인원"], marker='o', label="연인원")
plt.title("연도별 진료인원 변화 추이")
plt.xlabel("년도")
plt.ylabel("인원 수")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#연도별 진료유형별 실인원 추이
peopletreated_year_type = peopletreated_year.groupby(["년도", "구분"])[["자연인", "연인원"]].sum().reset_index()
# 피벗하여 시각화 용이하게
pivot_type = peopletreated_year_type.pivot(index="년도", columns="구분", values="자연인")
#시각화
pivot_type.plot(kind="line", marker="o", figsize=(8,5))
plt.title("연도별 진료유형별 실인원 추이")
plt.xlabel("년도")
plt.ylabel("실인원")
plt.grid(True)
plt.tight_layout()
plt.show()

#지역별 진료인원
peopletreated_yearregion = peopletreated_year.groupby("지역")[["자연인", "연인원"]].sum().sort_values(by="연인원", ascending=False)
#시각화
plt.figure(figsize=(10,6))
plt.barh(peopletreated_yearregion.index, peopletreated_yearregion["연인원"], color='skyblue')
plt.xlabel("연인원")
plt.title("진료인원 상위 지역 (누적 기준)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#병상 가동률
publichealth_year_summary = publichealth_year.groupby("구분명")[["년가동병상수", "입원년인원"]].sum().reset_index()

publichealth_year_summary["가동률(%)"] = (
    publichealth_year_summary["입원년인원"] / publichealth_year_summary["년가동병상수"]) * 100

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(publichealth_year_summary["구분명"], publichealth_year_summary["가동률(%)"], color="orange")
plt.xlabel("공공병상 가동률 (%)")
plt.title("공공병상 가동률 상위 10개 병원")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()