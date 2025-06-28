import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rc('font', family='Pretendard')
save_dir = 'imgs/EDA_질병및수술통계/003/'
os.makedirs(save_dir, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv('final_merged_data/질병 및 수술 통계.csv', names=['상병코드','상병명','국비','사비','년도','지역','구분','합계'], header=0)

# 1. 수술 데이터만 필터링
df = df[df['구분'] == '수술']

# 2. 연도별 전체 수술 건수 추이
plt.figure(figsize=(10,5))
sns.lineplot(data=df.groupby('년도')['합계'].sum().reset_index(), x='년도', y='합계', marker='o')
plt.title('연도별 전체 수술 건수 추이')
plt.xlabel('년도')
plt.ylabel('합계')
plt.tight_layout()
plt.savefig(save_dir + '연도별_전체_수술건수_추이.png')
plt.close()

# 3. 지역별 전체 수술 건수
plt.figure(figsize=(10,5))
region_sum = df.groupby('지역')['합계'].sum().sort_values(ascending=False)
sns.barplot(x=region_sum.index, y=region_sum.values, palette='viridis')
plt.title('지역별 전체 수술 건수')
plt.xlabel('지역')
plt.ylabel('합계')
plt.tight_layout()
plt.savefig(save_dir + '지역별_전체_수술건수.png')
plt.close()

# 4. 수술명별 Top 20 수술 건수
top_surgery = df.groupby('상병명')['합계'].sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(12,7))
sns.barplot(x=top_surgery.values, y=top_surgery.index, palette='magma')
plt.title('수술명별 전체 수술 건수 Top 20')
plt.xlabel('합계')
plt.ylabel('수술명')
plt.tight_layout()
plt.savefig(save_dir + '수술명별_전체_수술건수_Top20.png')
plt.close()

# 5. 연도별, 지역별 수술 합계 히트맵
pivot = df.pivot_table(index='지역', columns='년도', values='합계', aggfunc='sum', fill_value=0)
plt.figure(figsize=(12,7))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('연도별 지역별 수술 합계 히트맵')
plt.xlabel('년도')
plt.ylabel('지역')
plt.tight_layout()
plt.savefig(save_dir + '연도별_지역별_수술합계_히트맵.png')
plt.close()

# 6. 국비/사비 비율 (수술만)
total_gukbi = df['국비'].sum()
total_sabi = df['사비'].sum()
plt.figure(figsize=(6,6))
plt.pie([total_gukbi, total_sabi], labels=['국비','사비'], autopct='%1.1f%%', startangle=140)
plt.title('수술 국비/사비 비율')
plt.tight_layout()
plt.savefig(save_dir + '수술_국비_사비_비율.png')
plt.close()

# 7. 수술명별 연도별 추이 (Top 5)
top5 = df.groupby('상병명')['합계'].sum().sort_values(ascending=False).head(5).index
plt.figure(figsize=(12,6))
sns.lineplot(data=df[df['상병명'].isin(top5)].groupby(['상병명','년도'])['합계'].sum().reset_index(),
             x='년도', y='합계', hue='상병명', marker='o')
plt.title('수술명별 연도별 합계 추이 (Top 5)')
plt.xlabel('년도')
plt.ylabel('합계')
plt.tight_layout()
plt.savefig(save_dir + '수술명별_연도별_추이_Top5.png')
plt.close()