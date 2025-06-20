import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("new_merged_data/만성질환 환자 연령별 현황_인천포함_합계포함_0추가.csv")

# 질환별
national_sum = df.groupby(['년도', '지역', '상병명'], as_index=False)['연령별_합계'].sum()
grouped_stats = national_sum.groupby('상병명')['연령별_합계'].agg(['mean', 'std']).reset_index()
print("질환별 전국 평균 환자수 및 표준편차 (지역별 포함):")
print(grouped_stats)

national_yearly_sum = df.groupby(['년도', '상병명'], as_index=False)['연령별_합계'].sum()
pivot = national_yearly_sum.pivot(index='상병명', columns='년도', values='연령별_합계')
def calc_growth(df_pivot, start_year, end_year):
    growth = ((df_pivot[end_year] - df_pivot[start_year]) / df_pivot[start_year]) * 100
    growth_df = growth.reset_index()
    growth_df.columns = ['상병명', f'{start_year}~{end_year} 증가율(%)']
    return growth_df
growth_21_22 = calc_growth(pivot, 2021, 2022)
growth_22_23 = calc_growth(pivot, 2022, 2023)

print("\n2021년 대비 2022년 환자 수 증가율:")
print(growth_21_22)

print("\n2022년 대비 2023년 환자 수 증가율:")
print(growth_22_23)

# 만성질환별 연령대별 환자 수 누적막대 그래프
age_cols = ["59세 이하", "60-64세", "65-69세", "70-79세", "80-89세", "90세 이상"]
df_melt = df.melt(id_vars=["상병명"], value_vars=age_cols,
                  var_name="연령대", value_name="환자수")
pivot = df_melt.pivot_table(index="상병명", columns="연령대", values="환자수", aggfunc='sum').fillna(0)

pivot.loc[:, age_cols].plot(kind='bar', stacked=True)
plt.title("만성질환별 연령대별 환자수 누적 막대 그래프")
plt.ylabel("환자수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
sns.heatmap(pivot.loc[:, age_cols], annot=True, fmt="d", cmap="YlGnBu")
plt.title("만성질환별 연령대별 환자수 히트맵")
plt.xlabel("연령대")
plt.ylabel("만성질환명")
plt.tight_layout()
plt.show()

# 각 만성질환별 노년층 환자 비율 계산 및 시각화
df["노년층_합계"] = df[["65-69세", "70-79세", "80-89세", "90세 이상"]].sum(axis=1)
df["노년층_비율"] = df["노년층_합계"] / df["연령별_합계"]

plt.figure(figsize=(8,5))
sns.barplot(data=df, x="상병명", y="노년층_비율")
plt.title("만성질환별 노년층(65세 이상) 환자 비율")
plt.ylabel("노년층 환자 비율")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 전체 만성질환자 중 연령대별 분포 막대 그래프
age_sum = df[age_cols].sum().reindex(age_cols)

plt.figure(figsize=(8,5))
sns.barplot(x=age_sum.index, y=age_sum.values)
plt.title("전체 만성질환자 중 연령대별 분포")
plt.xlabel("연령대")
plt.ylabel("환자수")
plt.tight_layout()
plt.show()

# 주요 만성질환별 환자 수 증감 추이 선 그래프
df_grouped = df.groupby(["년도", "상병명"])["연령별_합계"].sum().reset_index()
topN = 5
top_diseases = df_grouped.groupby("상병명")["연령별_합계"].sum().nlargest(topN).index

plt.figure(figsize=(12,6))
for disease in top_diseases:
    data = df_grouped[df_grouped["상병명"]==disease]
    plt.plot(data["년도"], data["연령별_합계"], marker='o', label=disease)
    
    # 추세선 (선형회귀) 추가
    X = data["년도"].values.reshape(-1,1)
    y = data["연령별_합계"].values
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    plt.plot(data["년도"], trend, linestyle='--', alpha=0.7)

plt.title(f"주요 만성질환별 환자 수 증가 추이 및 추세선 (Top {topN})")
plt.xlabel("년도")
plt.ylabel("환자 수")
plt.legend()
plt.tight_layout()
plt.show()

# 전국적으로 환자 수 많은 Top N 만성질환 파이차트
total_by_disease = df.groupby("상병명")["연령별_합계"].sum().sort_values(ascending=False)
topN2 = 10
top_diseases2 = total_by_disease.head(topN2)

plt.figure(figsize=(8,8))
plt.pie(top_diseases2, labels=top_diseases2.index, autopct='%1.1f%%', startangle=140)
plt.title(f"전국 주요 만성질환 환자 수 비중 (Top {topN2})")
plt.axis('equal')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=top_diseases2.values, y=top_diseases2.index, palette='viridis')
plt.title(f"전국 주요 만성질환 환자 수 (Top {topN2})")
plt.xlabel("환자 수")
plt.ylabel("만성질환명")
plt.tight_layout()
plt.show()

# 지역별 특정 만성질환 유병률 시각화
target_disease = "당뇨병"
df_disease = df[df["상병명"] == target_disease]

df_region = df_disease.groupby("지역")["연령별_합계"].sum().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=df_region, x="지역", y="연령별_합계",
            hue="지역", palette="coolwarm", legend=False)
plt.title(f"[{target_disease}] 지역별 환자 수 분포")
plt.xlabel("지역")
plt.ylabel("환자 수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 특정질병집중도 분석: 병원별 비율 계산
total_by_hospital = df.groupby("지역")["연령별_합계"].sum().reset_index()
total_by_hospital.rename(columns={"연령별_합계": "전체환자수"}, inplace=True)

target_disease = "고혈압"
disease_by_hospital = df[df["상병명"] == target_disease].groupby("지역")["연령별_합계"].sum().reset_index()
disease_by_hospital.rename(columns={"연령별_합계": f"{target_disease}_환자수"}, inplace=True)

df_ratio = pd.merge(total_by_hospital, disease_by_hospital, on="지역", how="left")
df_ratio[f"{target_disease}_집중도"] = df_ratio[f"{target_disease}_환자수"] / df_ratio["전체환자수"]
df_ratio[f"{target_disease}_집중도"] = df_ratio[f"{target_disease}_집중도"].fillna(0)

df_ratio_sorted = df_ratio.sort_values(by=f"{target_disease}_집중도", ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=df_ratio_sorted, x="지역", y=f"{target_disease}_집중도", palette="mako")
plt.title(f"[{target_disease}] 병원별 집중도 (특정 질병 환자수 / 전체 만성질환 환자수)")
plt.xlabel("지역")
plt.ylabel("질병 집중도 (비율)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()