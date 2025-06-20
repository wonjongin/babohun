import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import networkx as nx

# 한글 폰트 설정 (Mac 기준)
plt.rc('font', family='AppleGothic')

# 데이터 불러오기
df = pd.read_csv('final_merged_data/진료과정보.csv')
print(df.head())

print(df.columns)
print(df.info())
print(df['년도'].unique())

# 예시: 보유장비/진료내용이 쉼표로 구분된 텍스트라면
def extract_keywords(text):
    if pd.isnull(text):
        return []
    return [kw.strip() for kw in str(text).replace(';', ',').split(',') if kw.strip()]

df['보유장비_키워드'] = df['보유장비'].apply(extract_keywords)
df['진료내용_키워드'] = df['진료내용'].apply(extract_keywords)

dept_equip = df.groupby('진료과별')['보유장비_키워드'].sum().apply(lambda x: list(set(x)))
print(dept_equip)

# 예시: 'MRI' 장비 보유 현황
important_equip = 'MRI'
df['MRI_보유'] = df['보유장비_키워드'].apply(lambda x: important_equip in x)

# 진료과별
plt.figure(figsize=(10,4))
sns.countplot(data=df, x='진료과별', hue='MRI_보유')
plt.title('진료과별 MRI 보유 현황')
plt.xticks(rotation=90)
plt.savefig('imgs/EDA_진료정보/004/진료과별 MRI 보유 현황.png')
# plt.show()

# 지역별
plt.figure(figsize=(10,4))
sns.countplot(data=df, x='지역', hue='MRI_보유')
plt.title('지역별 MRI 보유 현황')
plt.xticks(rotation=45)
# plt.show()

# for dept in df['진료과별'].unique():
#     keywords = sum(df[df['진료과별']==dept]['진료내용_키워드'], [])
#     word_freq = Counter(keywords)
#     if not word_freq:  # 키워드가 없으면 건너뜀
#         print(f"{dept} : 진료내용 키워드 없음, 워드클라우드 생략")
#         continue
#     wc = WordCloud(font_path='/System/Library/Fonts/AppleGothic.ttf', background_color='white', width=800, height=400)
#     plt.figure(figsize=(8,4))
#     plt.imshow(wc.generate_from_frequencies(word_freq))
#     plt.axis('off')
#     plt.title(f'{dept} 주요 진료내용 워드클라우드')
#     plt.savefig(f'imgs/EDA_진료정보/004/{dept} 주요 진료내용 워드클라우드.png')
#     # plt.show()

# 예시: 'CT', 'MRI', '초음파' 등 필수 장비
essential_equip = ['CT', 'MRI', '초음파']
for equip in essential_equip:
    df[f'{equip}_보유'] = df['보유장비_키워드'].apply(lambda x: equip in x)

pivot = df.pivot_table(index='지역', values=[f'{e}_보유' for e in essential_equip], aggfunc='mean')
pivot.plot(kind='bar', figsize=(10,5))
plt.title('지역별 필수 장비 보유율')
plt.ylabel('보유율')
plt.savefig('imgs/EDA_진료정보/004/지역별 필수 장비 보유율.png')
# plt.show()

# 예시: '부속부서' 컬럼이 있다면
if '부속부서' in df.columns:
    edges = []
    for _, row in df.iterrows():
        main_dept = row['진료과별']
        sub_depts = extract_keywords(row['부속부서'])
        for sub in sub_depts:
            edges.append((main_dept, sub))
    G = nx.Graph()
    G.add_edges_from(edges)

    # 연결 많은 상위 15개 노드만 남기기
    N = 50  # 원하는 노드 수로 조정
    top_nodes = [n for n, d in sorted(G.degree, key=lambda x: x[1], reverse=True)[:N]]
    G_sub = G.subgraph(top_nodes)
    pos = nx.spring_layout(G_sub, k=2, seed=42)

    # 진료과/부속부서 구분
    node_colors = []
    node_sizes = []
    for node in G_sub.nodes():
        if node in df['진료과별'].unique():
            node_colors.append('skyblue')
            node_sizes.append(400)
        else:
            node_colors.append('lightgreen')
            node_sizes.append(200)

    nx.draw_networkx(
        G_sub, pos=pos, with_labels=True, font_family='AppleGothic',
        node_color=node_colors, node_size=node_sizes, font_size=12
    )
    plt.title(f'진료과-부속부서 네트워크 (상위 {N}개 노드)')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(f'imgs/EDA_진료정보/004/진료과-부속부서_상위{N}.png', bbox_inches='tight', pad_inches=0.2)
    # plt.show()

yearly = df.groupby(['년도', '진료과별'])['MRI_보유'].mean().unstack()
yearly.plot(figsize=(12,6))
plt.title('연도별 진료과별 MRI 보유율 변화')
plt.ylabel('보유율')
plt.savefig('imgs/EDA_진료정보/004/연도별 진료과별 MRI 보유율 변화.png')
# plt.show()
