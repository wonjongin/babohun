import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import plotly.express as px

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df1 = pd.read_csv("new_merged_data/국가보훈부_보훈대상자 성별연령별 실인원현황.csv", encoding='euc-kr')
df2 = pd.read_csv("new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv")
df3 = pd.read_csv("merged_data/과별 퇴원환자 20대 주진단_열병합.csv")
df4 = pd.read_csv("new_merged_data/만성질환 환자 연령별 현황_인천포함_합계포함_0추가.csv")
df5 = pd.read_csv("merged_data/연령대별 20대상병정보.csv")

## 고령층에서 많이 발생하는 다빈도/만성질환 확인
age_cols = ['65-69', '70-79', '80-89', '90이상']

# 다빈도 기준
df2['고령층합'] = df2[age_cols].sum(axis=1)
top_diseases_elderly_df2 = df2.sort_values(by='고령층합', ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.barh(top_diseases_elderly_df2['상병명'], top_diseases_elderly_df2['고령층합'])
plt.xlabel("고령층 환자 수")
plt.title("고령층에서 많이 발생한 다빈도 질환 (입원 기준)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 만성질환 기준
df4['고령층합'] = df4[['65-69세', '70-79세', '80-89세', '90세 이상']].sum(axis=1)
top_diseases_elderly_df4 = df4.sort_values(by='고령층합', ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.barh(top_diseases_elderly_df4['상병명'], top_diseases_elderly_df4['고령층합'])
plt.xlabel("고령층 환자 수")
plt.title("고령층에서 많이 발생한 만성질환")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

## 퇴원 상병이 다빈도/만성질환과 어떤 관련성이 있는가
common_names_df3_df2 = set(df3['상병명']) & set(df2['상병명'])
common_names_df3_df4 = set(df3['상병명']) & set(df4['상병명'])

print("퇴원환자와 다빈도질환 공통 질병명:", common_names_df3_df2)
print("퇴원환자와 만성질환 공통 질병명:", common_names_df3_df4)

## 젊은 연령층에서 두드러진 특정 상병이 존재하는가
young_df = df5[df5['연령대'] == '59세 이하'].sort_values(by='건수', ascending=False).head(10)
plt.figure(figsize=(10,6))
plt.barh(young_df['상병명'], young_df['건수'])
plt.xlabel("건수")
plt.title("젊은 연령층(59세 이하)에서 많이 발생한 상병")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

## 연령대별 주요 질환 누적/그룹형 그래프
age_order = ['59세 이하', '60-64세', '65-69세', '70-79세', '80-89세', '90세 이상']
top3_df4 = df4.sort_values(by='연령별_합계', ascending=False).head(3)
top3_df4 = top3_df4.set_index('상병명')

plot_df = top3_df4[['59세 이하', '60-64세', '65-69세', '70-79세', '80-89세', '90세 이상']].T
plot_df = plot_df.loc[age_order]

plot_df.plot(kind='bar', figsize=(10,6))
plt.title("연령대별 주요 만성질환 환자 수")
plt.xlabel("연령대")
plt.ylabel("환자 수")
plt.legend(title="질환명")
plt.tight_layout()
plt.show()

## 특정 질환의 연령대별 선/막대 그래프
bp_row = df4[df4['상병명'].str.contains('고혈압')].iloc[0]

age_cols = ['59세 이하', '60-64세', '65-69세', '70-79세', '80-89세', '90세 이상']
bp_vals = bp_row[age_cols].values

plt.figure(figsize=(8, 5))
plt.plot(age_cols, bp_vals, marker='o', linestyle='-')
plt.title("연령대별 고혈압 환자 수")
plt.xlabel("연령대")
plt.ylabel("환자 수")
plt.grid(True)
plt.tight_layout()
plt.show()