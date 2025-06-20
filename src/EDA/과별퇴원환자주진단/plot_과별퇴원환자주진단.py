import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("merged_data/과별 퇴원환자 20대 주진단_열병합.csv")

## 기술통계
# 진료과별/연도별/지역별 실인원 합계, 평균, 표준편차
print(df.groupby('진료과')['실인원'].agg(['sum', 'mean', 'std']))
df.groupby('진료과')['실인원'].agg(['sum', 'mean', 'std']) \
  .reset_index() \
  .to_csv("src/EDA/과별퇴원환자주진단/진료과별_실인원_통계.csv", index=False, encoding='utf-8-sig')

print(df.groupby('년도')['실인원'].agg(['sum', 'mean', 'std']))
df.groupby('년도')['실인원'].agg(['sum', 'mean', 'std']) \
  .reset_index() \
  .to_csv("src/EDA/과별퇴원환자주진단/연도별_실인원_통계.csv", index=False, encoding='utf-8-sig')

print(df.groupby('지역')['실인원'].agg(['sum', 'mean', 'std']))
df.groupby('지역')['실인원'].agg(['sum', 'mean', 'std']) \
  .reset_index() \
  .to_csv("src/EDA/과별퇴원환자주진단/지역별_실인원_통계.csv", index=False, encoding='utf-8-sig')

# 전국 퇴원 질병별 실인원 평균, 표준편차
print(df.groupby('상병명')['실인원'].agg(['mean', 'std']).sort_values('mean', ascending=False))
df.groupby('상병명')['실인원'].agg(['mean', 'std']) \
  .sort_values('mean', ascending=False) \
  .reset_index() \
  .to_csv("src/EDA/과별퇴원환자주진단/상병명별_실인원_평균_표준편차.csv", index=False, encoding='utf-8-sig')

## 시각화
# 진료과별 퇴원환자 실인원 합계 막대그래프
pivot = df.pivot_table(values='실인원', index='진료과', columns='년도', aggfunc='sum')
pivot.plot(kind='bar', figsize=(30, 6))
plt.title('진료과별 퇴원환자 합계 (연도별)')
plt.ylabel('실인원 합계')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 지역별, 진료과별 퇴원환자 히트맵
pivot2 = df.pivot_table(values='실인원', index='지역', columns='진료과', aggfunc='sum')

plt.figure(figsize=(30, 6))
sns.heatmap(pivot2, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('지역별 · 진료과별 퇴원환자 수 Heatmap')
plt.show()

# 연도별 특정 질병 퇴원 환자 수 추이
df['상병명'] = df['상병명'].astype(str)
target = df[df['상병명'].str.contains('고혈압', na=False)]

trend = target.groupby('년도')['실인원'].sum()
trend.plot(marker='o')
plt.title('연도별 고혈압 퇴원환자 추이')
plt.ylabel('실인원')
plt.grid(True)
plt.show()

# 진료과별 Top N 질병
N = 5
top_by_dept = df.groupby(['진료과', '상병명'], as_index=False)['실인원'].sum()
top_by_dept_sorted = top_by_dept.sort_values(['진료과', '실인원'], ascending=[True, False])
top_disease = top_by_dept_sorted.groupby('진료과').head(N)

plt.figure(figsize=(20, 10))
sns.barplot(data=top_disease, x='상병명', y='실인원', hue='진료과')
plt.title(f'진료과별 Top {N} 질병')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 전국 Top N 퇴원 질병
top_total = df.groupby('상병명')['실인원'].sum().sort_values(ascending=False).head(N)

top_total.plot(kind='bar')
plt.title(f'전국 Top {N} 퇴원 질병')
plt.ylabel('실인원 합계')
plt.xticks(rotation=45)
plt.show()

# 과별 주요 진단명 워드 클라우드
text = ' '.join(df[df['진료과'] == '가정의학과']['상병명'].tolist())
wordcloud = WordCloud(font_path='malgun.ttf', background_color='white').generate(text)

plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('가정의학과 주요 진단명 워드클라우드')
plt.show()