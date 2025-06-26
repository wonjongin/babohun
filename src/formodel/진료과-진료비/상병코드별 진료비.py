import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def load_and_preprocess():
    df = pd.read_csv('src/formodel/상병코드-진료과/output_진료과_매핑.csv')

    # 진료비 원 단위 변환
    df['진료비_원'] = df['진료비(천원)'] * 1000

    # 진료과 리스트로 분리 (구분자 '+' 기준, 공백 포함 허용)
    df['대표진료과'] = df['진료과'].str.split(r'\s*\+\s*').str[0]

    # 진료비와 연인원 균등 분배
    def expand_and_equal_split(row):
        과들 = row['진료과']
        n = len(과들)
        진료비_나누기 = row['진료비_원'] / n
        연인원_나누기 = row['연인원'] / n
        return pd.DataFrame({
            '년도': [row['년도']] * n,
            '지역': [row['지역']] * n,
            '구분': [row['구분']] * n,
            '상병코드': [row['상병코드']] * n,
            '진료과': 과들,
            '진료비_원': [진료비_나누기] * n,
            '연인원': [연인원_나누기] * n,
        })

    df_expanded = pd.concat(df.apply(expand_and_equal_split, axis=1).tolist(), ignore_index=True)

    # 대표 진료과: 첫 번째 진료과만 사용
    df['대표진료과'] = df['진료과'].str.split(r'\s*\+\s*').str[0]

    return df, df_expanded

def group_and_anova(df_equal, df_represent):
    # 균등 분배 기준 그룹화
    grouped_equal = df_equal.groupby(['진료과']).agg(
        총진료비_원=('진료비_원', 'sum'),
        총연인원=('연인원', 'sum')
    ).reset_index()
    grouped_equal['1인당_평균진료비'] = grouped_equal['총진료비_원'] / grouped_equal['총연인원']

    # 대표 진료과 기준 그룹화
    grouped_represent = df_represent.groupby(['대표진료과']).agg(
        총진료비_원=('진료비_원', 'sum'),
        총연인원=('연인원', 'sum')
    ).reset_index()
    grouped_represent['1인당_평균진료비'] = grouped_represent['총진료비_원'] / grouped_represent['총연인원']

    # ANOVA - 균등 분배 기준
    model_equal = ols('Q("1인당_평균진료비") ~ C(진료과)', data=grouped_equal).fit()
    anova_equal = sm.stats.anova_lm(model_equal, typ=2)

    # ANOVA - 대표 진료과 기준
    model_represent = ols('Q("1인당_평균진료비") ~ C(대표진료과)', data=grouped_represent).fit()
    anova_represent = sm.stats.anova_lm(model_represent, typ=2)

    return grouped_equal, grouped_represent, anova_equal, anova_represent

if __name__ == '__main__':
    print("== 시작 ==")
    df_represent, df_equal = load_and_preprocess()
    
    print("원본 데이터 대표진료과 샘플:\n", df_represent.head())
    print("균등분배 확장 데이터 샘플:\n", df_equal.head())
    
    grouped_equal, grouped_represent, anova_equal, anova_represent = group_and_anova(df_equal, df_represent)
    
    print("그룹별 요약 (균등분배 기준):\n", grouped_equal.head())
    print("그룹별 요약 (대표진료과 기준):\n", grouped_represent.head())
    
    print("\n✅ [ANOVA - 균등분배 진료과 기준]\n", anova_equal.to_string())
    print("\n✅ [ANOVA - 대표진료과 기준]\n", anova_represent.to_string())
    print("== 종료 ==")
