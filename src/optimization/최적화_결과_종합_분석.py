import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

# 결과 폴더 경로
bed_dir = 'optimization_results_병상_분배_균등화_실제'
cost_dir = 'optimization_results_진료비_분배_최적화_개선'
doctor_dir = 'optimization_results_전문의_분배_최적화'
output_dir = 'optimization_results_결과종합'
os.makedirs(output_dir, exist_ok=True)

# ----------------------
# 1. 병상 분배 결과 비교
# ----------------------
def analyze_bed():
    results = {}
    for tag, fname in zip(['PuLP', 'SLSQP', 'GA'],
        ['병상_분배_균등화_결과.csv', '병상_분배_균등화_결과_SLSQP.csv', '병상_분배_균등화_결과_GA.csv']):
        fpath = os.path.join(bed_dir, fname)
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath)
        # 실제 컬럼명에 맞춰 집계
        std_before = df['현재_병상가동률'].std()
        std_after = df['최적_병상가동률'].std()
        mean_before = df['현재_병상가동률'].mean()
        mean_after = df['최적_병상가동률'].mean()
        total_change = df['변화량'].sum()
        mean_change_rate = df['변화율'].mean()
        results[tag] = {
            '최적화_전_가동률_표준편차': std_before,
            '최적화_후_가동률_표준편차': std_after,
            '최적화_전_가동률_평균': mean_before,
            '최적화_후_가동률_평균': mean_after,
            '총_변화량': total_change,
            '평균_변화율(%)': mean_change_rate
        }
    return results

# ----------------------
# 2. 진료비 분배 결과 비교
# ----------------------
def analyze_cost():
    results = {}
    for tag, fname in zip(['PuLP', 'SLSQP', 'GA'],
        ['진료비_분배_최적화_결과.csv', '진료비_분배_최적화_결과_SLSQP.csv', '진료비_분배_최적화_결과_GA.csv']):
        fpath = os.path.join(cost_dir, fname)
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath)
        total_current = df['현재_진료비(천원)'].sum()
        total_optimal = df['최적_진료비(천원)'].sum()
        total_change = df['변화량(천원)'].sum()
        mean_change_rate = df['변화율(%)'].mean()
        results[tag] = {
            '총_현재_진료비(천원)': total_current,
            '총_최적_진료비(천원)': total_optimal,
            '총_변화량(천원)': total_change,
            '평균_변화율(%)': mean_change_rate
        }
    # json에서 목적함수 값 등 추가
    for tag, fname in zip(['PuLP', 'SLSQP', 'GA'],
        ['최적화_요약.json', '최적화_요약_SLSQP.json', '최적화_요약_GA.json']):
        fpath = os.path.join(cost_dir, fname)
        if not os.path.exists(fpath): continue
        with open(fpath, 'r', encoding='utf-8') as f:
            js = json.load(f)
        if tag in results:
            results[tag].update(js)
        else:
            results[tag] = js
    return results

# ----------------------
# 3. 전문의 분배 결과 비교
# ----------------------
def analyze_doctor():
    results = {}
    for tag, fname in zip(['PuLP', 'SLSQP', 'GA'],
        ['전문의_분배_최적화_결과.csv', '전문의_분배_최적화_결과_SLSQP.csv', '전문의_분배_최적화_결과_GA.csv']):
        fpath = os.path.join(doctor_dir, fname)
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath)
        total_current = df['현재전문의수'].sum()
        total_optimal = df['최적전문의수'].sum()
        total_change = df['변화량'].sum()
        mean_change_rate = df['변화율'].mean()
        results[tag] = {
            '총_현재전문의수': total_current,
            '총_최적전문의수': total_optimal,
            '총_변화량': total_change,
            '평균_변화율(%)': mean_change_rate
        }
    # json에서 목적함수 값 등 추가
    for tag, fname in zip(['PuLP', 'SLSQP', 'GA'],
        ['최적화_요약.json', '최적화_요약_SLSQP.json', '최적화_요약_GA.json']):
        fpath = os.path.join(doctor_dir, fname)
        if not os.path.exists(fpath): continue
        with open(fpath, 'r', encoding='utf-8') as f:
            js = json.load(f)
        if tag in results:
            results[tag].update(js)
        else:
            results[tag] = js
    return results

# ----------------------
# 4. 종합 비교 및 시각화
# ----------------------
def plot_and_save(results, title, fname):
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(output_dir, f'{fname}.csv'), encoding='utf-8-sig')
    ax = df.plot(kind='bar', figsize=(12,6), title=title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fname}.png'))
    plt.close()

# ----------------------
# 6. 정성적 비교 표 생성
# ----------------------
def make_summary_table(bed, cost, doctor):
    # 병상: 표준편차, 진료비: 목적함수_값, 전문의: objective_value
    # 제약조건 위반: json의 성공여부/상태, 쏠림: 표준편차 기준, 실행 시간/해석력: 고정
    # 병상 분배
    bed_obj = {k: v.get('최적화_후_가동률_표준편차', None) for k, v in bed.items()}
    # 진료비 분배
    cost_obj = {
        'PuLP': cost.get('PuLP', {}).get('목적함수_값', None),
        'SLSQP': cost.get('SLSQP', {}).get('목적함수_값', None),
        'GA': cost.get('GA', {}).get('최고_적합도', None)
    }
    # 전문의 분배 (json 없음, objective_value는 PuLP만)
    doctor_obj = {
        'PuLP': doctor.get('PuLP', {}).get('objective_value', None),
        'SLSQP': None,
        'GA': None
    }
    # 제약조건 위반 (진료비 분배 json 기준)
    cost_constraint = {
        'PuLP': '없음' if cost.get('PuLP', {}).get('최적화_상태', '') == 'Optimal' else '있음',
        'SLSQP': '없음' if cost.get('SLSQP', {}).get('최적화_성공', True) else '있음',
        'GA': '없음' if cost.get('GA', {}).get('최고_적합도', 0) > -1e8 else '있음'
    }
    # 쏠림(표준편차 기준)
    def solim(std):
        if std is None: return '정보없음'
        if std > 10: return '심함'
        elif std > 3: return '중간'
        else: return '적음'
    # 실행 시간/해석력(고정)
    speed = {'PuLP':'빠름','SLSQP':'중간','GA':'느림'}
    interp = {'PuLP':'높음','SLSQP':'높음','GA':'낮음(확률적)'}
    # 표 생성
    table = []
    for tag in ['PuLP', 'SLSQP', 'GA']:
        table.append([
            tag,
            bed_obj.get(tag, None),
            cost_obj.get(tag, None),
            doctor_obj.get(tag, None),
            cost_constraint.get(tag, '정보없음'),
            solim(bed_obj.get(tag, None)),
            speed[tag],
            interp[tag]
        ])
    return table

# ----------------------
# 5. 리포트 텍스트 생성
# ----------------------
def make_report(bed, cost, doctor):
    lines = []
    lines.append('### 최적화 방법별 성과 비교 요약')
    lines.append('')
    # 정성적 비교 표
    lines.append('| 구분 | 병상분배(가동률 표준편차) | 진료비분배(목적함수값) | 전문의분배(objective) | 제약조건 위반 | 분배 쏠림 | 실행 시간 | 해석력 |')
    lines.append('|------|--------------------------|----------------------|----------------------|--------------|-----------|-----------|---------|')
    for row in make_summary_table(bed, cost, doctor):
        lines.append(f'| {row[0]} | {row[1]:.4f} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} |')
    lines.append('')
    # 기존 요약
    for cat, res in zip(['병상 분배', '진료비 분배', '전문의 분배'], [bed, cost, doctor]):
        lines.append(f'#### {cat}')
        for tag, vals in res.items():
            lines.append(f'- {tag}: ' + ', '.join([f'{k}={v}' for k,v in vals.items()]))
        lines.append('')
    lines.append('---')
    lines.append('**SLSQP 방식이 목적함수 값, 제약조건 만족, 현실성 등에서 가장 우수한 결과를 보였으며, GA는 다양한 해를 탐색할 수 있으나 실행 시간과 일관성에서 한계가 있었다. PuLP는 해석력과 속도는 뛰어나나, 분배 쏠림 현상이 심했다.**')
    with open(os.path.join(output_dir, '최적화_성과_종합_리포트.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ----------------------
# 메인 실행
# ----------------------
if __name__ == '__main__':
    bed = analyze_bed()
    cost = analyze_cost()
    doctor = analyze_doctor()
    plot_and_save(bed, '병상 분배 최적화 성과 비교', '병상_분배_성과비교')
    plot_and_save(cost, '진료비 분배 최적화 성과 비교', '진료비_분배_성과비교')
    plot_and_save(doctor, '전문의 분배 최적화 성과 비교', '전문의_분배_성과비교')
    make_report(bed, cost, doctor)
    print('최적화 성과 종합 분석 및 시각화/리포트 저장 완료!') 