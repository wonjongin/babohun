import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

# 저장 디렉토리 생성
save_dir = 'imgs/EDA_연도별진료인원/001/'
os.makedirs(save_dir, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv('final_merged_data/연도별 진료인원.csv')

# 실인원(자연인)과 연인원 각각 처리
def create_ratio_chart(data_type, year_data):
    result = []
    for region in year_data['지역'].unique():
        region_data = year_data[year_data['지역'] == region]
        total = region_data[data_type].sum()
        
        for _, row in region_data.iterrows():
            ratio = (row[data_type] / total) * 100
            result.append({
                '지역': region,
                '구분': row['구분'],
                '비율': ratio,
                '인원': row[data_type]
            })
    
    result_df = pd.DataFrame(result)
    pivot_df = result_df.pivot(index='지역', columns='구분', values='비율')
    pivot_df = pivot_df.fillna(0)
    
    return pivot_df

# 연도별 전체 병원 통합 비율 계산 함수
def create_total_ratio_trend(data_type):
    trend_data = []
    years = sorted(df['년도'].unique())
    
    for year in years:
        year_data = df[df['년도'] == year]
        total = year_data[data_type].sum()
        
        for _, row in year_data.iterrows():
            ratio = (row[data_type] / total) * 100
            trend_data.append({
                '년도': year,
                '구분': row['구분'],
                '비율': ratio
            })
    
    trend_df = pd.DataFrame(trend_data)
    pivot_trend = trend_df.pivot_table(index='년도', columns='구분', values='비율', aggfunc='sum')
    return pivot_trend

# 연도별로 그래프 생성
years = sorted(df['년도'].unique())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for year in years:
    # 해당 연도 데이터 필터링
    df_year = df[df['년도'] == year]
    
    # 실인원(자연인)과 연인원 각각 처리
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 실인원 차트
    pivot_natural = create_ratio_chart('자연인', df_year)
    pivot_natural.plot(kind='bar', stacked=True, color=colors, ax=ax1)
    
    ax1.set_title(f'{year}년 지역별 실인원(자연인) 구분 비율', fontsize=14, fontweight='bold')
    ax1.set_xlabel('지역', fontsize=12)
    ax1.set_ylabel('비율 (%)', fontsize=12)
    ax1.legend(title='구분', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 비율 표시
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%.1f%%', label_type='center')
    
    # 연인원 차트
    pivot_annual = create_ratio_chart('연인원', df_year)
    pivot_annual.plot(kind='bar', stacked=True, color=colors, ax=ax2)
    
    ax2.set_title(f'{year}년 지역별 연인원 구분 비율', fontsize=14, fontweight='bold')
    ax2.set_xlabel('지역', fontsize=12)
    ax2.set_ylabel('비율 (%)', fontsize=12)
    ax2.legend(title='구분', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 비율 표시
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%.1f%%', label_type='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}{year}년_지역별_진료인원_비율.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 데이터 테이블 출력
    print(f"\n{year}년 지역별 실인원(자연인) 구분 비율:")
    print(pivot_natural.round(1))
    print(f"\n{year}년 지역별 연인원 구분 비율:")
    print(pivot_annual.round(1))
    print("\n" + "="*50)

# 연도별 전체 병원 통합 비율 추이 그래프
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 실인원(자연인) 추이
trend_natural = create_total_ratio_trend('자연인')
trend_natural.plot(kind='line', marker='o', linewidth=2, markersize=8, ax=ax1, color=colors)

ax1.set_title('연도별 전체 병원 실인원(자연인) 구분 비율 추이', fontsize=14, fontweight='bold')
ax1.set_xlabel('연도', fontsize=12)
ax1.set_ylabel('비율 (%)', fontsize=12)
ax1.legend(title='구분', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# 실인원 비율 레이블 추가
for i, col in enumerate(trend_natural.columns):
    for j, (year, value) in enumerate(trend_natural[col].items()):
        ax1.annotate(f'{value:.1f}%', 
                    xy=(j, value), 
                    xytext=(0, 8), 
                    textcoords='offset points', 
                    ha='center', 
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

# 연인원 추이 (2010년 제외)
trend_annual = create_total_ratio_trend('연인원')
trend_annual_filtered = trend_annual[trend_annual.index != 2010]  # 2010년 제외
trend_annual_filtered.plot(kind='line', marker='o', linewidth=2, markersize=8, ax=ax2, color=colors)

ax2.set_title('연도별 전체 병원 연인원 구분 비율 추이 (2010년 제외)', fontsize=14, fontweight='bold')
ax2.set_xlabel('연도', fontsize=12)
ax2.set_ylabel('비율 (%)', fontsize=12)
ax2.legend(title='구분', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

# 연인원 비율 레이블 추가
for i, col in enumerate(trend_annual_filtered.columns):
    for j, (year, value) in enumerate(trend_annual_filtered[col].items()):
        ax2.annotate(f'{value:.1f}%', 
                    xy=(j, value), 
                    xytext=(0, 8), 
                    textcoords='offset points', 
                    ha='center', 
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig(f'{save_dir}연도별_전체병원_진료인원_비율_추이.png', dpi=300, bbox_inches='tight')
plt.close()

# 추이 데이터 출력
print("\n연도별 전체 병원 실인원(자연인) 구분 비율 추이:")
print(trend_natural.round(1))
print("\n연도별 전체 병원 연인원 구분 비율 추이 (2010년 제외):")
print(trend_annual_filtered.round(1))

print(f"\n그래프가 {save_dir} 폴더에 저장되었습니다.")
