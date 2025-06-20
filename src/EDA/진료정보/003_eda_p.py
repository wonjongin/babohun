# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from matplotlib import font_manager, rc
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
font_path = "fonts/Pretendard-Regular.ttf"
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)
rc('font', family=font_manager.FontProperties(fname=font_path).get_name())
plt.rcParams['axes.unicode_minus'] = False

# 워드클라우드 라이브러리 임포트 시도
try:
    from wordcloud import WordCloud
    wordcloud_available = True
except ImportError:
    wordcloud_available = False
    print("WordCloud 패키지가 설치되어 있지 않습니다. 워드클라우드 시각화는 건너뜁니다.")

# 데이터 로드
print("데이터 로드 중...")
df_medical_info = pd.read_csv('final_merged_data/진료과정보.csv')
df_age_reservation = pd.read_csv('final_merged_data/연령대별 진료과 예약건수_2023.csv')
df_monthly_discharge = pd.read_csv('final_merged_data/월별 진료과별 퇴원.csv')
print("데이터 로드 완료!")


# 진료과별 보유장비 수 분석
equipment_by_dept = df_medical_info.groupby('진료과별').size().reset_index(name='장비 수')
equipment_by_dept = equipment_by_dept.sort_values('장비 수', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='장비 수', y='진료과별', data=equipment_by_dept.head(15))
plt.title('진료과별 보유장비 수 (상위 15개)')
plt.tight_layout()
plt.savefig('imgs/EDA_진료정보/진료과별_보유장비_수.png')
plt.show()

# 워드클라우드 생성 함수
def generate_wordcloud_for_column(df, group_col, text_col, group_value):
    """특정 그룹에 대한 텍스트 열의 워드 클라우드 생성"""
    if not wordcloud_available:
        return
    
    text = ' '.join(df[df[group_col] == group_value][text_col].dropna())
    if not text.strip():
        print(f"'{group_value}'의 '{text_col}' 데이터가 없습니다.")
        return
    
    font_path = "fonts/Pretendard-Regular.ttf"
    # for font in ['Pretendard', 'Malgun Gothic', 'AppleGothic', 'NanumGothic']:
    #     try:
    #         font_path = font_manager.findfont(font_manager.FontProperties(family=font))
    #         break
    #     except:
    #         continue
    
    # 워드 클라우드 생성
    wordcloud = WordCloud(
        font_path=font_path,
        width=800, height=400,
        background_color='white',
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{group_value} - {text_col} 워드 클라우드')
    plt.tight_layout()
    plt.savefig(f'imgs/EDA_진료정보/{group_value}_{text_col}_워드클라우드.png')
    plt.show()

# 주요 진료과 워드클라우드 생성
top_depts = equipment_by_dept['진료과별'].head(5).tolist()
for dept in top_depts:
    generate_wordcloud_for_column(df_medical_info, '진료과별', '보유장비', dept)
    generate_wordcloud_for_column(df_medical_info, '진료과별', '진료내용', dept)

# 지역별 진료과 분포 히트맵
region_dept_count = df_medical_info.groupby(['지역', '진료과별']).size().unstack().fillna(0)
plt.figure(figsize=(14, 10))
sns.heatmap(region_dept_count, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('지역별 진료과 분포')
plt.tight_layout()
plt.savefig('imgs/EDA_진료정보/지역별_진료과_분포.png')
plt.show()

# 진료과별 부속부서 분석
dept_subdept = df_medical_info.groupby(['진료과별', '부속부서']).size().reset_index(name='장비 수')
dept_subdept_count = dept_subdept.groupby('진료과별').size().reset_index(name='부속부서 수')
dept_subdept_count = dept_subdept_count.sort_values('부속부서 수', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='부속부서 수', y='진료과별', data=dept_subdept_count.head(15))
plt.title('진료과별 부속부서 수 (상위 15개)')
plt.tight_layout()
plt.savefig('imgs/EDA_진료정보/진료과별_부속부서_수.png')
plt.show()
