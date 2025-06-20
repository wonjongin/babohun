import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rc('font', family='Pretendard')
os.makedirs('imgs/EDA_연관D/002', exist_ok=True)

# 파일 경로 리스트
data_dir = 'final_merged_data/'
file_list = glob.glob(data_dir + '*.csv')

# 데이터 전체 읽기
df_resv = pd.read_csv('final_merged_data/연령대별 진료과 예약건수_2023.csv')
df_diag = pd.read_csv('final_merged_data/외래 진료과별 상위20 주요상병.csv')

# 과거 진료 데이터 전처리
if '건수' in df_diag.columns:
    df_diag['건수'] = pd.to_numeric(df_diag['건수'], errors='coerce').fillna(0)


# 예약 데이터
# df_resv = dfs.get('연령대별 진료과 예약건수_2023.csv')
age_cols = [col for col in df_resv.columns if '대' in col]
df_resv['예약합계'] = df_resv[age_cols].replace(',', '', regex=True).astype(float).sum(axis=1)

# 과거 진료 데이터
# df_diag = dfs.get('외래 진료과별 상위20 주요상병.csv')

# 만성/급성 키워드
chronic_kw = ['당뇨','고혈압','만성','신부전','지질','골다공증','협심증','심부전','neoplasm','carcinoma','cancer','diabetes','hypertension','chronic','renal failure','cirrhosis','arthritis','stroke','myocardial infarction']
acute_kw = ['급성','감염','염','폐렴','감기','출혈','통증','외상','골절','acute','infection','pain','fracture','bleeding','trauma']

def is_chronic(x):
    return any(kw.lower() in str(x).lower() for kw in chronic_kw)
def is_acute(x):
    return any(kw.lower() in str(x).lower() for kw in acute_kw)

df_diag['만성'] = df_diag['상병명'].apply(is_chronic)
df_diag['급성'] = df_diag['상병명'].apply(is_acute)

# 최근 예약 상위 진료과(연령대별)
for age in age_cols:
    top_resv = df_resv[['진료과', age]].sort_values(age, ascending=False).head(5)
    plt.figure(figsize=(7,4))
    sns.barplot(x=top_resv[age], y=top_resv['진료과'], palette='Blues_r')
    plt.title(f'{age} 최근 예약 상위 진료과')
    plt.xlabel('예약건수')
    plt.ylabel('진료과')
    plt.tight_layout()
    plt.savefig(f'imgs/EDA_연관D/002/{age}_최근예약_상위진료과.png')
    plt.close()

# 과거 진료 데이터에서 연령대별 상위 진료과(가능한 경우)
if '연령대' in df_diag.columns:
    for age in df_diag['연령대'].unique():
        top_diag = df_diag[df_diag['연령대']==age].groupby('진료과')['건수'].sum().sort_values(ascending=False).head(5)
        plt.figure(figsize=(7,4))
        sns.barplot(x=top_diag.values, y=top_diag.index, palette='Greens_r')
        plt.title(f'{age} 과거 진료 상위 진료과')
        plt.xlabel('진료건수')
        plt.ylabel('진료과')
        plt.tight_layout()
        plt.savefig(f'imgs/EDA_연관D/002/{age}_과거진료_상위진료과.png')
        plt.close()

# 최근 예약 히트맵
plt.figure(figsize=(10,6))
sns.heatmap(df_resv.set_index('진료과')[age_cols], annot=True, fmt='.0f', cmap='Blues')
plt.title('진료과별 연령대별 최근 예약건수 히트맵')
plt.savefig('imgs/EDA_연관D/002/진료과별_연령대별_최근예약_히트맵.png')
plt.close()

# 과거 진료 히트맵(가능한 경우)
if '연령대' in df_diag.columns:
    pivot = df_diag.pivot_table(index='진료과', columns='연령대', values='건수', aggfunc='sum')
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Greens')
    plt.title('진료과별 연령대별 과거 진료건수 히트맵')
    plt.savefig('imgs/EDA_연관D/002/진료과별_연령대별_과거진료_히트맵.png')
    plt.close()

# 변화 감지(텍스트 출력)
for age in age_cols:
    recent_top = set(df_resv.sort_values(age, ascending=False)['진료과'].head(3))
    if '연령대' in df_diag.columns:
        past_top = set(df_diag[df_diag['연령대']==age].groupby('진료과')['건수'].sum().sort_values(ascending=False).head(3).index)
        changed = recent_top - past_top
        if changed:
            print(f"{age}에서 최근 예약 상위 진료과 변화 감지: {changed}")

# 예약 상위 진료과의 주요 상병 만성/급성 비율
top_depts = df_resv.groupby('진료과')['예약합계'].sum().sort_values(ascending=False).head(10).index
if df_diag is not None:
    for dept in top_depts:
        dept_diag = df_diag[df_diag['진료과']==dept]
        chronic_ratio = dept_diag['만성'].mean()
        acute_ratio = dept_diag['급성'].mean()
        print(f"{dept}: 만성질환 비율={chronic_ratio:.2f}, 급성질환 비율={acute_ratio:.2f}")

        # 주요 상병별 진료건수 barplot
        top_disease = dept_diag.groupby('상병명')['건수'].sum().sort_values(ascending=False).head(5)
        plt.figure(figsize=(7,4))
        sns.barplot(x=top_disease.values, y=top_disease.index, palette='Set2')
        plt.title(f'{dept} 주요 상병별 진료건수')
        plt.xlabel('진료건수')
        plt.ylabel('상병명')
        plt.tight_layout()
        plt.savefig(f'imgs/EDA_연관D/002/{dept}_주요상병_진료건수.png')
        plt.close()

