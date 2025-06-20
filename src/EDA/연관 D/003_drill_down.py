import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rc('font', family='AppleGothic')
os.makedirs('imgs/EDA_연관D/003', exist_ok=True)

# 데이터 불러오기
df_resv = pd.read_csv('final_merged_data/연령대별 진료과 예약건수_2023.csv')
df_diag = pd.read_csv('final_merged_data/외래 진료과별 상위20 주요상병.csv')
df_chronic = pd.read_csv('final_merged_data/만성질환 환자 연령별 현황.csv')

# 진료과/연령대 drill-down → 진료과 drill-down으로 변경
target_dept = '순환기내과'

# 연도별 진료건수(과거)
if '년도' in df_diag.columns:
    df_diag_dept = df_diag[df_diag['진료과']==target_dept]
    yearly = df_diag_dept.groupby('년도')['건수'].sum()
    if not yearly.empty:
        plt.figure(figsize=(8,4))
        sns.lineplot(x=yearly.index, y=yearly.values, marker='o')
        plt.title(f'{target_dept} 연도별 진료건수 추이')
        plt.ylabel('진료건수')
        plt.xlabel('년도')
        plt.tight_layout()
        plt.savefig(f'imgs/EDA_연관D/003/{target_dept}_연도별진료건수.png')
        plt.close()
    else:
        print(f"{target_dept}에 해당하는 연도별 진료 데이터가 없습니다.")

# 2023년 예약건수(연령대별 진료과 예약건수_2023.csv)
if target_dept in df_resv['진료과'].values:
    resv_row = df_resv[df_resv['진료과'] == target_dept]
    print(f"{target_dept} 2023년 예약건수(연령대별):")
    print(resv_row)
else:
    print(f"{target_dept}에 해당하는 예약 데이터가 없습니다.")

# 만성/급성 비율(진료과 단위)
chronic_kw = ['당뇨','고혈압','만성','신부전','지질','골다공증','협심증','심부전','neoplasm','carcinoma','cancer','diabetes','hypertension','chronic','renal failure','cirrhosis','arthritis','stroke','myocardial infarction']
acute_kw = ['급성','감염','염','폐렴','감기','출혈','통증','외상','골절','acute','infection','pain','fracture','bleeding','trauma']

def is_chronic(x):
    return any(kw.lower() in str(x).lower() for kw in chronic_kw)
def is_acute(x):
    return any(kw.lower() in str(x).lower() for kw in acute_kw)

df_diag['만성'] = df_diag['상병명'].apply(is_chronic)
df_diag['급성'] = df_diag['상병명'].apply(is_acute)

if '년도' in df_diag.columns:
    df_diag_dept = df_diag[df_diag['진료과']==target_dept]
    chronic_ratio = df_diag_dept.groupby('년도')['만성'].mean()
    acute_ratio = df_diag_dept.groupby('년도')['급성'].mean()
    if not chronic_ratio.empty:
        plt.figure(figsize=(8,4))
        plt.plot(chronic_ratio.index, chronic_ratio.values, marker='o', label='만성질환 비율')
        plt.plot(acute_ratio.index, acute_ratio.values, marker='o', label='급성질환 비율')
        plt.title(f'{target_dept} 연도별 만성/급성질환 비율')
        plt.ylabel('비율')
        plt.xlabel('년도')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'imgs/EDA_연관D/003/{target_dept}_연도별_만성_급성비율.png')
        plt.close()
    else:
        print(f"{target_dept}에 해당하는 만성/급성 비율 데이터가 없습니다.")

# 주요 상병별 진료건수 barplot
top_disease = df_diag_dept.groupby('상병명')['건수'].sum().sort_values(ascending=False).head(10)
if not top_disease.empty:
    plt.figure(figsize=(8,5))
    sns.barplot(x=top_disease.values, y=top_disease.index, palette='Set2')
    plt.title(f'{target_dept} 주요 상병별 진료건수')
    plt.xlabel('진료건수')
    plt.ylabel('상병명')
    plt.tight_layout()
    plt.savefig(f'imgs/EDA_연관D/003/{target_dept}_주요상병_진료건수.png')
    plt.close()
else:
    print(f"{target_dept}에 해당하는 상병별 진료 데이터가 없습니다.")
