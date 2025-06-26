import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정 
mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우용 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

publichealth_year = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/연도별 공공의료병상 활성화 현황_2019.csv", encoding="utf-8-sig")

#공공의료병상 공급 병원별
publichealth_year_summary = publichealth_year.groupby("구분명")[["년가동병상수", "입원년인원"]].sum().reset_index()
# 시각화
plt.figure(figsize=(10, 6))
plt.barh(publichealth_year_summary["구분명"], publichealth_year_summary["년가동병상수"], color='lightgreen')
plt.xlabel("연간 가동 병상 수")
plt.ylabel("병원명")
plt.title("공공의료병상 공급")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
