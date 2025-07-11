import json
import pandas as pd

# 파일 경로
json_path = 'model_results_연령지역_진료과/performance/class_performance_metadata.json'

# 사용할 모델
target_models = ['LGB', 'GB', 'RF', 'LR', 'Stacking']

# JSON 로드
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 진료과명 리스트 추출 (모든 모델에 동일하다고 가정)
class_names = [c['class_name'] for c in data['LGB']['class_performance']]

rows = []
for class_idx, class_name in enumerate(class_names):
    first_row = True
    for model in target_models:
        perf = data[model]['class_performance'][class_idx]
        row = [class_name if first_row else '',
               model,
               perf['precision'],
               perf['recall'],
               perf['f1_score'],
               int(perf['support']),
               perf['accuracy']]
        rows.append(row)
        first_row = False

columns = ['클래스명', '모델', 'Precision', 'Recall', 'F1-score', 'Support', 'Accuracy']
df = pd.DataFrame(rows)
df.columns = columns

# 표 저장 (엑셀, CSV, 마크다운)
df.to_excel('모델1_성능평가표.xlsx', index=False)
df.to_csv('모델1_성능평가표.csv', index=False)
md_table = df.to_markdown(index=False)
if md_table is None:
    md_table = ''
with open('모델1_성능평가표.md', 'w', encoding='utf-8') as f:
    f.write(md_table)

print('표 저장 완료!')
