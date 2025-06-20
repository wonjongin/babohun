import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정 
mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우용 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

EMR = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/통EMR_부산보훈병원_2023.csv", encoding="utf-8-sig")

EMR["작업일자"] = pd.to_datetime(EMR["작업일자"])
emr_flow = EMR[["작업일자", "입원실인원수", "퇴원실인원수", "금일재원인원수"]].set_index("작업일자")
#시각화
plt.figure(figsize=(12, 6))
plt.plot(emr_flow.index, emr_flow["입원실인원수"], label="입원", marker='o')
plt.plot(emr_flow.index, emr_flow["퇴원실인원수"], label="퇴원", marker='s')
plt.plot(emr_flow.index, emr_flow["금일재원인원수"], label="재원", marker='^')
plt.title("일자별 입·퇴원 및 재원 환자 수 추이")
plt.xlabel("날짜")
plt.ylabel("인원 수")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#입원 환자 수
plt.figure(figsize=(10, 4))
plt.plot(emr_flow.index, emr_flow["입원실인원수"], color="blue", marker='o')
plt.title("일자별 입원 환자 수 추이")
plt.xlabel("날짜")
plt.ylabel("입원 환자 수")
plt.grid(True)
plt.tight_layout()
plt.show()
#일자별 퇴원 환자 수
plt.figure(figsize=(10, 4))
plt.plot(emr_flow.index, emr_flow["퇴원실인원수"], color="red", marker='s')
plt.title("일자별 퇴원 환자 수 추이")
plt.xlabel("날짜")
plt.ylabel("퇴원 환자 수")
plt.grid(True)
plt.tight_layout()
plt.show()
#일자별 재원 환자 수 추이
plt.figure(figsize=(10, 4))
plt.plot(emr_flow.index, emr_flow["금일재원인원수"], color="green", marker='^')
plt.title("일자별 재원 환자 수 추이")
plt.xlabel("날짜")
plt.ylabel("재원 환자 수")
plt.grid(True)
plt.tight_layout()
plt.show()
