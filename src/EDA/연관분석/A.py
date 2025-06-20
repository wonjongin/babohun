import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df1 = pd.read_csv("new_merged_data/만성질환 환자 연령별 현황_인천포함_합계포함_0추가.csv")
df2 = pd.read_csv("merged_data/연령대별 20대상병정보.csv")
df3 = pd.read_csv("data/통합EMR시스템_사비 진료과별 입원 진료현황/한국보훈복지의료공단_통EMR_부산보훈병원_20231231.csv", encoding='euc-kr')

print(df1.head(3))
print(df2.head(3))
print(df3.head(3))

## 특정 기간의 EMR 입원/퇴원 사유 중 만성질환 관련 상병 비율
# 만성질환 코드 목록 예시 (df4, df10에서 추출된 코드라고 가정)
chronic_codes = ['I10~I13, I15', 'E10~E14', 'F00~F99, G40~G41']  # 실제로는 정제된 코드 목록 필요

# df1: 입원 데이터 (연도, 지역, 상병코드 포함)
df1['is_chronic'] = df1['코드'].isin(chronic_codes)

# 연도별 만성질환 상병 비율 계산
chronic_ratio_by_year = df1.groupby(['년도', 'is_chronic'])['연령별_합계'].sum().reset_index()
total_by_year = chronic_ratio_by_year.groupby('년도')['연령별_합계'].sum()
chronic_ratio_by_year['비율'] = chronic_ratio_by_year.apply(
    lambda row: (row['연령별_합계'] / total_by_year[row['년도']]) * 100, axis=1
)
pivot_table = chronic_ratio_by_year.pivot(index='년도', columns='is_chronic', values='비율')
pivot_table.columns = ['비만성질환_비율(%)', '만성질환_비율(%)']
print(pivot_table.round(2))
pivot_table.to_csv("src/EDA/연관분석/연도별_만성질환_비율_pivot.csv", encoding='utf-8-sig')

# 시각화: 연도별 만성질환 입원자 비율 막대그래프
chronic_pivot = chronic_ratio_by_year.pivot(index='년도', columns='is_chronic', values='비율')
chronic_pivot.columns = ['비만성질환', '만성질환']
chronic_pivot.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('연도별 입원 환자 중 만성질환 비율')
plt.ylabel('비율 (%)')
plt.xlabel('년도')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

## 일자별 재원 환자 중 특정 연령대 비율 변화
# df3에 연령대별 정보 없음 -> 불가능