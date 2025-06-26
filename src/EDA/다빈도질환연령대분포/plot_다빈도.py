import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("new_merged_data/다빈도 질환 환자 연령별 분포_순위추가_합계계산_값통일.csv")
df['진료비(천원)'] = pd.to_numeric(df['진료비(천원)'], errors='coerce').fillna(0)

df['실인원'] = pd.to_numeric(df['실인원'], errors='coerce')
df['연인원'] = pd.to_numeric(df['연인원'], errors='coerce')

## 기술통계:
# 3개년 전체 및 각 연도별 상병코드별 실인원/연인원/진료비 평균, 표준편차, 합계
# 단, 진료비는 NA 많아서 부정확할 가능성 높아보임
stats = df.groupby(['년도', '상병코드']).agg(
    실인원_합계 = ('실인원', 'sum'), 실인원_평균=('실인원', 'mean'), 실인원_표준편차=('실인원', 'std'),
    연인원_합계=('연인원', 'sum'), 연인원_평균=('연인원', 'mean'), 연인원_표준편차=('연인원', 'std'),
    진료비_합계 = ('진료비(천원)', 'sum'), 진료비_평균=('진료비(천원)', 'mean'), 진료비_표준편차=('진료비(천원)', 'std')
).reset_index()

total_stats = df.groupby('상병코드').agg(
    실인원_합계=('실인원', 'sum'), 실인원_평균=('실인원', 'mean'), 실인원_표준편차=('실인원', 'std'),
    연인원_합계=('연인원', 'sum'), 연인원_평균=('연인원', 'mean'), 연인원_표준편차=('연인원', 'std'),
    진료비_합계=('진료비(천원)', 'sum'), 진료비_평균=('진료비(천원)', 'mean'), 진료비_표준편차=('진료비(천원)', 'std')
).reset_index()
total_stats.insert(0, '년도', '전체')

combined_stats = pd.concat([stats, total_stats], ignore_index=True)
print(combined_stats)

# 연령대별 환자수 합계
age_cols = ['59이하', '60-64', '65-69', '70-79', '80-89', '90이상']
for col in age_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 연도별 + 연령대별
year_age = df.groupby('년도')[age_cols].sum().reset_index()
print("연도별 + 연령대별 환자 수 합계: \n", year_age)

# 병원별 + 연령대별
region_age = df.groupby('지역')[age_cols].sum().reset_index()
print("병원별 + 연령대별 환자 수 합계: \n", region_age)

# 연도별 + 병원별 + 연령대별
year_region_age = df.groupby(['년도', '지역'])[age_cols].sum().reset_index()
print("연도별 + 병원별 + 연령대별 환자 수 합계: \n", year_region_age)

## 시각화
# 상병코드별 실인원/연인원 추이 선 그래프 (연도별)
num_cols = ['실인원', '연인원', '59이하', '60-64', '65-69', '70-79', '80-89', '90이상']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

top_codes = df.groupby('상병코드')['연인원'].sum().nlargest(5).index
df_line = df[df['상병코드'].isin(top_codes)]

fig = px.line(df_line, x='년도', y='연인원', color='상병코드',
              title='상병코드별 연도별 연인원 추이',
              markers=True)
fig.show()

# 상병코드별 연령대별 환자 분포 히트맵
age_cols = ['59이하', '60-64', '65-69', '70-79', '80-89', '90이상']

# 상위 10개 상병코드 추출
top_codes = df.groupby('상병코드')[age_cols].sum().sum(axis=1).nlargest(10).index

# 히트맵 데이터 생성
heat_data = df[df['상병코드'].isin(top_codes)].groupby('상병코드')[age_cols].sum()

# 열(연령대)별 Z-score 정규화
heat_data_z = (heat_data - heat_data.mean()) / heat_data.std()

# 히트맵 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(heat_data_z, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('정규화된 상위 10개 상병코드별 연령대별 환자 분포 히트맵 (Z-score)')
plt.xlabel('연령대')
plt.ylabel('상병코드')
plt.tight_layout()
plt.show()

# 연령대별로 상위 10개 상병코드 추출 및 시각화
df_2023 = df[df['년도'] == 2023]

inpatient = df_2023[df_2023['구분'] == '입원(실인원)'].copy()
outpatient = df_2023[df_2023['구분'] == '외래'].copy()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('2023년 연령대별 상위 10개 상병코드 분포 (입원 실인원 + 외래)', fontsize=18)

for i, age in enumerate(age_cols):
    row, col = i // 3, i % 3

    # 연령대별 상병코드별 실인원 합계
    in_age_sum = inpatient.groupby('상병코드')[age].sum()
    out_age_sum = outpatient.groupby('상병코드')[age].sum()

    # 합산 후 상위 10개
    combined = (in_age_sum + out_age_sum).nlargest(10)

    # 시각화
    sns.barplot(x=combined.values, y=combined.index, ax=axes[row, col], palette='Set2')
    axes[row, col].set_title(f'{age} 연령대')
    axes[row, col].set_xlabel('환자 수')
    axes[row, col].set_ylabel('상병코드')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 65세 이상 노년층 다빈도 상위 질환
old_cols = ['65-69', '70-79', '80-89', '90이상']
inpatient['노년층환자수'] = inpatient[old_cols].sum(axis=1)
outpatient['노년층환자수'] = outpatient[old_cols].sum(axis=1)

combined_df = pd.concat([inpatient, outpatient])
top_n = 10
top_disease = combined_df.groupby('상병코드')['노년층환자수'].sum().nlargest(top_n).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=top_disease, x='노년층환자수', y='상병코드', palette='viridis')
plt.title('노년층(65세 이상) 다빈도 질환 Top 10 (입원 실인원 + 외래)')
plt.xlabel('환자 수')
plt.ylabel('상병 코드')
plt.tight_layout()
plt.show()

# 지역 버블차트
df_bubble = df_2023[df_2023['구분'].isin(['입원(실인원)', '외래'])]
top_codes_2023 = df_bubble.groupby('상병코드')['실인원'].sum().nlargest(10).index
df_bubble_2023 = df_bubble[df_bubble['상병코드'].isin(top_codes_2023)]
df_bubble_2023 = df_bubble_2023.groupby(['지역', '상병코드'], as_index=False)['실인원'].sum()

# 4. 버블 차트
import plotly.express as px
fig = px.scatter(
    df_bubble_2023,
    x='지역',
    y='상병코드',
    size='실인원',
    color='상병코드',
    title='[2023년] 지역별 주요 상병코드별 환자수 분포 (버블 차트)',
    size_max=60
)
fig.show()

# 지역별 상병코드별 실인원 합계
df_region = df_2023[df_2023['구분'].isin(['입원(실인원)', '외래'])]
df_grouped = df_region.groupby(['지역', '상병코드'])['실인원'].sum().reset_index()
regions = df_grouped['지역'].unique()

# subplot 구성 (3행 × 2열)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()

# 각 지역별 상위 10개 상병코드 시각화
for i, region in enumerate(regions):
    region_df = df_grouped[df_grouped['지역'] == region]
    top10 = region_df.sort_values('실인원', ascending=False).head(10)

    sns.barplot(data=top10, y='상병코드', x='실인원', palette='Set2', ax=axes[i], dodge=False, legend=False)
    axes[i].set_title(f"2023년 {region} 지역 상위 10개 상병코드")
    axes[i].set_xlabel("환자 수")
    axes[i].set_ylabel("상병코드")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 상병코드별 '연인원/실인원' 비율 계산 및 시각화 (막대 그래프 또는 산점도)
df['방문비율'] = df['연인원'] / df['실인원']
top20 = df.sort_values('방문비율', ascending=False).head(20)

chronic_codes = df['상병코드'].unique()
print(chronic_codes)
df['상병코드'] = df['상병코드'].str.strip()

chronic_disease_codes_estimated = [
    'I10', 'E11', 'F20', 'F06', 'I20', 'I25', 'I50', 'I65', 'I67', 'I63', 'I69', 'G81', 'G82', 
    'C16', 'C18', 'C22', 'C25', 'C34', 'C61', 'C67', 'D09', 'D41', 'N18' 
]

# 생성된 리스트의 길이와 내용 일부 출력 (확인용)
print(f"추정된 만성질환 코드 개수: {len(chronic_disease_codes_estimated)}")
print("일부 코드:", chronic_disease_codes_estimated[:10]) # 처음 10개 코드만 출력

df['만성질환여부'] = df['상병코드'].apply(
    lambda x: '만성' if any(x.strip().startswith(code) for code in chronic_disease_codes_estimated) else '기타'
)
print(df[['상병코드', '만성질환여부']].head())


plt.figure(figsize=(14, 6))
sns.barplot(data=df, x='상병코드', y='방문비율', hue='만성질환여부')  # 또는 y='진료비'
plt.xticks(rotation=90)
plt.title('상병코드별 방문비율')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='만성질환여부', y='방문비율', palette='Set2')
plt.title('만성질환 vs 기타질환 방문비율 비교')
plt.tight_layout()
plt.show()

# 진료비 사례연구
df_clean = df[df['진료비(천원)'].notna()].copy()
df_clean['진료비'] = df_clean['진료비(천원)'] * 1000

# 연도별 반복
for year in sorted(df_clean['년도'].unique()):
    df_year = df_clean[df_clean['년도'] == year]
    regions = df_year['지역'].unique()

    ### 1. 방문비율 vs 진료비 산점도 - 지역별 subplot
    if '방문비율' in df_year.columns and '만성질환여부' in df_year.columns:
        n = len(regions)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for i, region in enumerate(regions):
            data = df_year[df_year['지역'] == region]
            if data.empty:
                continue
            fixed_palette = {'만성': '#83B7DE', '기타': '#F7CFD8'}
            sns.scatterplot(data=data, x='방문비율', y='진료비(천원)', hue='만성질환여부',
                            ax=axes[i], alpha=0.7, palette=fixed_palette, legend=True)
            axes[i].set_title(f'{year}년 {region}')
            axes[i].set_xlabel('연인원 / 실인원')
            axes[i].set_ylabel('진료비 (천원)')

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f'{year}년 - 지역별 방문비율 vs 진료비', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    ### 2. 상병코드별 평균 진료비 (상위 20개) - 지역별 subplot
    if '상병코드' in df_year.columns:
        n = len(regions)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
        axes = axes.flatten()

        for i, region in enumerate(regions):
            data = df_year[df_year['지역'] == region]
            stats = data.groupby('상병코드')['진료비(천원)'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(20)

            axes[i].bar(stats.index, stats['mean'], yerr=stats['std'], capsize=5, color='skyblue')
            axes[i].set_title(f'{region}')
            axes[i].tick_params(axis='x', labelrotation=45)
            axes[i].set_ylabel('평균 진료비 (천원)')

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f'{year}년 - 지역별 상병코드별 평균 진료비 (상위 20개)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    ### 3. 연령대별 1인당 진료비 박스플롯 - 지역별 subplot
    if all(col in df_year.columns for col in age_cols):
        df_melted = df_year.melt(
            id_vars=['지역', '상병코드', '상병명', '진료비'],
            value_vars=age_cols,
            var_name='연령대',
            value_name='인원수'
        )
        df_melted = df_melted.dropna(subset=['인원수'])
        df_melted = df_melted[df_melted['인원수'] > 0]
        df_melted['1인당진료비'] = df_melted['진료비'] / df_melted['인원수']

        n = len(regions)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
        axes = axes.flatten()

        for i, region in enumerate(regions):
            region_data = df_melted[df_melted['지역'] == region]
            if region_data.empty:
                continue
            sns.boxplot(data=region_data, x='연령대', y='1인당진료비', ax=axes[i], palette='pastel')
            axes[i].set_title(f'{region}')
            axes[i].set_xlabel('연령대')
            axes[i].set_ylabel('1인당 진료비 (원)')
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f'{year}년 - 지역별 연령대별 1인당 진료비 분포', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# 연령대 melt 후 처리 (기존과 동일)
df_melted = df_year.melt(
    id_vars=['지역', '상병코드', '상병명', '진료비'],
    value_vars=age_cols,
    var_name='연령대',
    value_name='인원수'
)
df_melted = df_melted.dropna(subset=['인원수'])
df_melted = df_melted[df_melted['인원수'] > 0]
df_melted['1인당진료비'] = df_melted['진료비'] / df_melted['인원수']

# subplot 구성
n = len(regions)
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
axes = axes.flatten()

for i, region in enumerate(regions):
    region_data = df_melted[df_melted['지역'] == region]
    if region_data.empty:
        continue

    # ⭐ stripplot 또는 swarmplot 중 택일
    sns.stripplot(data=region_data, x='연령대', y='1인당진료비',
                  ax=axes[i], palette='pastel', jitter=True, alpha=0.5)
    # sns.swarmplot(data=region_data, x='연령대', y='1인당진료비',
    #               ax=axes[i], palette='pastel', alpha=0.7)

    axes[i].set_title(f'{region}')
    axes[i].set_xlabel('연령대')
    axes[i].set_ylabel('1인당 진료비 (원)')
    axes[i].tick_params(axis='x', rotation=45)

# 빈 subplot 비우기
for j in range(i+1, len(axes)):
    axes[j].axis('off')

fig.suptitle(f'{year}년 - 지역별 연령대별 1인당 진료비 분포 (stripplot)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

df_2023 = df[df['년도'] == 2023]

inpatient = df_2023[df_2023['구분'] == '입원(실인원)'].copy()
outpatient = df_2023[df_2023['구분'] == '외래'].copy()


# 연령대별 컬럼을 모두 합쳐서 전체 환자 수 구하기
# age_cols가 연령대 컬럼 리스트라고 가정
inpatient['전체환자수'] = inpatient[age_cols].sum(axis=1)
outpatient['전체환자수'] = outpatient[age_cols].sum(axis=1)

# 상병코드별 전체환자수 합산
in_sum = inpatient.groupby('상병코드')['전체환자수'].sum()
out_sum = outpatient.groupby('상병코드')['전체환자수'].sum()

# 입원 + 외래 합산 후 상위 10개 추출
total_sum = in_sum + out_sum
top10 = total_sum.nlargest(10)

# 시각화
plt.figure(figsize=(10,6))
sns.barplot(x=top10.values, y=top10.index, palette='Set2')
plt.title('2023년 전체 상병코드별 상위 10개 환자 수 분포 (입원 + 외래)')
plt.xlabel('환자 수')
plt.ylabel('상병코드')
plt.show()


'''
# 입원/외래 분석
# 컬럼명 지정
in_col = '입원(실인원)'
out_col = '외래'

# 데이터 집계
grouped = df.groupby(['상병코드', '구분'])['실인원'].sum().unstack(fill_value=0)
grouped['합계'] = grouped[in_col] + grouped[out_col]

# 상위 10개 인덱스
top10_in_idx = grouped.sort_values(by=in_col, ascending=False).head(10).index
top10_out_idx = grouped.sort_values(by=out_col, ascending=False).head(10).index
top10_total_idx = grouped.sort_values(by='합계', ascending=False).head(10).index

# 각각 추출
top10_in = grouped.loc[top10_in_idx]
top10_out = grouped.loc[top10_out_idx]
top10_total = grouped.loc[top10_total_idx]

# 시각화 함수
def plot_stacked(df, title, colors):
    ax = df[[in_col, out_col]].plot(kind='bar', stacked=True, color=colors, figsize=(12,6))
    ax.set_title(title)
    ax.set_ylabel('실인원')
    plt.xticks(rotation=45)
    ax.legend(title='구분')
    plt.tight_layout()
    plt.show()

# 페이지별 그래프 출력
plot_stacked(top10_in, '입원 기준 상위 10개 상병코드 (입원+외래 누적)', ['#f1948a', '#aed6f1'])
plot_stacked(top10_out, '외래 기준 상위 10개 상병코드 (외래+입원 누적)', ['#f1948a', '#aed6f1'])
plot_stacked(top10_total, '입원+외래 총합 기준 상위 10개 상병코드', ['#f1948a', '#aed6f1'])

# 만성/비만성 연령대별 막대그래프
df_melted = df.melt(
    id_vars=['상병코드', '상병명', '만성질환여부'],
    value_vars=age_cols,
    var_name='연령대',
    value_name='환자수'
)
grouped = df_melted.groupby(['연령대', '만성질환여부'])['환자수'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped, x='연령대', y='환자수', hue='만성질환여부')
plt.title('연령대별 만성/비만성 질환자 분포')
plt.ylabel('환자 수')
plt.tight_layout()
plt.show()

# 만성/준만성/비만성 연령대별 막대그래프
sub_chronic_disease = [
 'C61', 'D09', 'E14', 'F00', 'F03', 'G20', 'G62', 'G95', 'H25', 'H35', 'H42', 'H90',
 'I11', 'I48', 'I70', 'I73', 'I83', 'J44', 'J45',
 'K05', 'K21', 'K29', 'K59', 'K80',
 'L23', 'L24', 'L29', 'L30', 'L89',
 'M17', 'M19', 'M43', 'M48', 'M50', 'M51', 'M54', 'M75', 'M96',
 'N31', 'N40',
 'Z49', 'Z50', 'Z51'
]

def classify_disease(code):
    if code in chronic_codes:
        return '만성질환'
    elif code in sub_chronic_disease:
        return '준만성질환'
    else:
        return '비만성질환'
    
df['만성질환여부'] = df['상병코드'].apply(classify_disease)

df_melted = df.melt(
    id_vars=['상병코드', '상병명', '만성질환여부'],
    value_vars=age_cols,
    var_name='연령대',
    value_name='환자수'
)
grouped = df_melted.groupby(['연령대', '만성질환여부'])['환자수'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped, x='연령대', y='환자수', hue='만성질환여부')
plt.title('연령대별 만성/준만성/비만성 질환자 분포')
plt.ylabel('환자 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 만성질환의 연인원/실인원 비율 확인
chronic_df = df[df['상병코드'].isin(chronic_codes)].copy()
chronic_df['방문비율'] = chronic_df['연인원'] / chronic_df['실인원']
chronic_df_sorted = chronic_df.sort_values(by='방문비율', ascending=False)[['상병코드', '상병명', '방문비율']]
print(chronic_df_sorted.head(10))
chronic_df_sorted.to_csv("src/EDA/다빈도질환연령대분포/만성질환_방문비율_전체.csv", index=False, encoding='utf-8-sig')

# 연도별 Top N 질환 순위 변화
pivot = df.groupby(['년도', '상병코드'])['실인원'].sum().reset_index()
pivot['순위'] = pivot.groupby('년도')['실인원'].rank(method='first', ascending=False)
top_n = 10
top_codes = pivot[pivot['순위'] <= top_n]
pivot_chart = top_codes.pivot(index='상병코드', columns='년도', values='순위')

plt.figure(figsize=(10, 6))
for code in pivot_chart.index:
    plt.plot(pivot_chart.columns, pivot_chart.loc[code], marker='o', label=code)

plt.gca().invert_yaxis()
plt.title('연도별 질환 순위 변화 (Top 10)')
plt.xlabel('년도')
plt.ylabel('순위 (낮을수록 상위)')
plt.legend(title='상병코드', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
'''