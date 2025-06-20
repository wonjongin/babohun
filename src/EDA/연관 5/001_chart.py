import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

plt.rc('font', family='AppleGothic')
os.makedirs('imgs/EDA_연관5/001', exist_ok=True)

# 데이터 불러오기
equip = pd.read_csv('final_merged_data/진료과정보.csv')  # 14번
disease = pd.read_csv('final_merged_data/다빈도 질환 환자 연령별 분포.csv')  # 2번
out = pd.read_csv('final_merged_data/과별 퇴원환자 20대 주진단.csv')  # 3번
stat = pd.read_csv('final_merged_data/질병 및 수술 통계.csv')  # 15번

# 진료과별 대표 장비 추출 (중복 제거)
equip_group = equip.groupby('진료과별')['보유장비'].apply(lambda x: ', '.join(x.dropna().unique())).reset_index()

# 진료과별 주요 질환(상병명) 추출 (퇴원환자 기준)
main_disease = out.groupby('진료과')['상병명'].apply(lambda x: ', '.join(x.dropna().unique()[:5])).reset_index()
main_disease.columns = ['진료과', '주요질환']

# 진료과별 장비-질환 매칭 테이블
table = pd.merge(equip_group, main_disease, left_on='진료과별', right_on='진료과', how='outer')
table = table[['진료과별', '보유장비', '주요질환']]
print(table.head(10))

# 엑셀/CSV로 저장 (선택)
table.to_csv('imgs/EDA_연관5/001/진료과별_장비_주요질환_매칭.csv', index=False)

# 진료과별 환자수 상위 5개 질환 막대그래프
for dept in out['진료과'].unique():
    dept_df = out[out['진료과'] == dept]
    top_disease = dept_df.groupby('상병명')['실인원'].sum().sort_values(ascending=False).head(5)
    plt.figure(figsize=(7,4))
    sns.barplot(x=top_disease.values, y=top_disease.index, palette='Blues_r')
    plt.title(f'{dept} 주요 질환별 환자수 Top5')
    plt.xlabel('환자수')
    plt.ylabel('상병명')
    plt.tight_layout()
    plt.savefig(f'imgs/EDA_연관5/001/{dept}_주요질환_막대.png')
    plt.close()

# # 진료과별 주요 질환 워드클라우드
# for dept in out['진료과'].unique():
#     dept_df = out[out['진료과'] == dept]
#     word_freq = dept_df.groupby('상병명')['실인원'].sum().to_dict()
#     if not word_freq: continue
#     wc = WordCloud(font_path='fonts/Pretendard-Regular.ttf', width=800, height=400, background_color='white')
#     plt.figure(figsize=(8,4))
#     plt.imshow(wc.generate_from_frequencies(word_freq))
#     plt.axis('off')
#     plt.title(f'{dept} 주요 질환 워드클라우드')
#     plt.tight_layout()
#     plt.savefig(f'imgs/EDA_연관5/001/{dept}_주요질환_워드클라우드.png')
#     plt.close()

# 고가 장비 키워드 정의
expensive_equip = ['MRI', 'CT', '컴퓨터전신단층촬영기', '자기공명영상장치']

# 진료과별 고가장비 보유 여부
equip['고가장비보유'] = equip['보유장비'].apply(lambda x: any(eq in str(x) for eq in expensive_equip))
expensive_dept = equip[equip['고가장비보유']]['진료과별'].unique()

# 고가장비 보유/미보유 그룹별 환자수(예: 암 관련)
cancer_keywords = ['암', 'malignant', 'carcinoma', 'neoplasm', 'cancer']
out['암질환'] = out['상병명'].apply(lambda x: any(kw in str(x) for kw in cancer_keywords))
out['고가장비보유'] = out['진료과'].isin(expensive_dept)

# 그룹별 환자수 비교
grouped = out.groupby(['고가장비보유', '암질환'])['실인원'].sum().unstack().fillna(0)
grouped.plot(kind='bar', stacked=True, figsize=(7,5), colormap='Set2')
plt.title('고가장비 보유여부별 암질환 환자수')
plt.xlabel('고가장비 보유여부')
plt.ylabel('환자수')
plt.xticks([0,1], ['미보유', '보유'], rotation=0)
plt.tight_layout()
plt.savefig('imgs/EDA_연관5/001/고가장비_암질환_환자수_비교.png')
plt.close()

# 진료과별 전체 환자수, 암환자수
dept_total = out.groupby('진료과')['실인원'].sum()
dept_cancer = out[out['암질환']].groupby('진료과')['실인원'].sum()
df_scatter = pd.DataFrame({'전체환자수': dept_total, '암환자수': dept_cancer}).fillna(0)
df_scatter['고가장비보유'] = df_scatter.index.isin(expensive_dept)

plt.figure(figsize=(7,5))
sns.scatterplot(data=df_scatter, x='전체환자수', y='암환자수', hue='고가장비보유', style='고가장비보유', s=100)
plt.title('진료과별 전체환자수 vs 암환자수 (고가장비 보유여부)')
plt.xlabel('전체환자수')
plt.ylabel('암환자수')
plt.tight_layout()
plt.savefig('imgs/EDA_연관5/001/진료과별_전체vs암환자수_산점도.png')
plt.close()



# 만성질환 키워드 정의
chronic_keywords = [
    '당뇨', 'diabetes', '고혈압', 'hypertension', '심부전', 'heart failure',
    '만성', 'chronic', '천식', 'asthma', 'COPD', '신부전', 'renal failure',
    '간경변', 'cirrhosis', '관절염', 'arthritis', '뇌졸중', 'stroke', '심근경색', 'myocardial infarction'
]
out['만성질환'] = out['상병명'].apply(lambda x: any(kw.lower() in str(x).lower() for kw in chronic_keywords))

# 고가장비 보유여부는 기존과 동일
out['고가장비보유'] = out['진료과'].isin(expensive_dept)

# 그룹별 환자수 비교
grouped = out.groupby(['고가장비보유', '만성질환'])['실인원'].sum().unstack().fillna(0)
grouped.plot(kind='bar', stacked=True, figsize=(7,5), colormap='Set2')
plt.title('고가장비 보유여부별 만성질환 환자수')
plt.xlabel('고가장비 보유여부')
plt.ylabel('환자수')
plt.xticks([0,1], ['미보유', '보유'], rotation=0)
plt.tight_layout()
plt.savefig('imgs/EDA_연관5/001/고가장비_만성질환_환자수_비교.png')
plt.close()

# 진료과별 전체 환자수, 만성질환 환자수
dept_total = out.groupby('진료과')['실인원'].sum()
dept_chronic = out[out['만성질환']].groupby('진료과')['실인원'].sum()
df_scatter = pd.DataFrame({'전체환자수': dept_total, '만성질환환자수': dept_chronic}).fillna(0)
df_scatter['고가장비보유'] = df_scatter.index.isin(expensive_dept)

plt.figure(figsize=(7,5))
sns.scatterplot(data=df_scatter, x='전체환자수', y='만성질환환자수', hue='고가장비보유', style='고가장비보유', s=100)
plt.title('진료과별 전체환자수 vs 만성질환환자수 (고가장비 보유여부)')
plt.xlabel('전체환자수')
plt.ylabel('만성질환환자수')
plt.tight_layout()
plt.savefig('imgs/EDA_연관5/001/진료과별_전체vs만성질환환자수_산점도.png')
plt.close()

