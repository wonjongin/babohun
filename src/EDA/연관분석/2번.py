import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정 
mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우용 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

# 데이터 불러오기
bed = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/병상정보.csv", encoding="utf-8-sig")
patientsfrequentdiseases = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/다빈도 질환 환자 연령별 분포.csv", encoding="utf-8-sig")
EMR = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/통EMR_부산보훈병원_2023.csv", encoding="utf-8-sig")
Medical = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/진료과정보.csv", encoding="utf-8-sig")
chronic = pd.read_csv("C:/Users/jenny/babohun/final_merged_data/만성질환 환자 연령별 현황.csv", encoding="utf-8-sig")

sorted(patientsfrequentdiseases[patientsfrequentdiseases["지역"] == "서울"]["상병명"].dropna().unique())
def map_seoul_disease_to_department(disease_name):
    name = disease_name.strip().lower()

    if any(x in name for x in ["hypertension", "hypertensive", "right heart failure", "angina", "atherosclerotic"]):
        return "순환기내과"
    elif any(x in name for x in ["diabetes", "non-insulin", "non0insulin"]):
        return "내분비내과"
    elif any(x in name for x in ["lung cancer", "bronchopneumonia", "pneumonia", "covid"]):
        return "호흡기내과"
    elif any(x in name for x in ["hepatocellular", "cholangiocarcinoma", "gallstone", "cardia of stomach"]):
        return "소화기내과"
    elif any(x in name for x in ["chronic kidney", "tubular necrosis"]):
        return "신장내과"
    elif any(x in name for x in ["cataract", "glaucoma", "retinopathy"]):
        return "안과"
    elif any(x in name for x in ["disc", "stenosis", "myelopathy", "spinal"]):
        return "신경외과"
    elif any(x in name for x in ["cerebral infarction", "concussion", "hemiplegia", "paraplegia", "occlusion", "dissection", "dizziness"]):
        return "신경과"
    elif any(x in name for x in ["caries", "pulpitis", "teeth", "tooth"]):
        return "치과"
    elif any(x in name for x in ["prostate", "bph", "benign prostate"]):
        return "비뇨의학과"
    elif any(x in name for x in ["arthritis", "periarthritis", "femur neck fracture", "thr", "osteoarthritis"]):
        return "정형외과"
    elif any(x in name for x in ["paranoid schizophrenia"]):
        return "정신건강의학과"
    elif any(x in name for x in ["dermatitis"]):
        return "피부과"
    elif any(x in name for x in ["hernia"]):
        return "외과"
    elif any(x in name for x in ["emergency use of u07", "work accident", "acute pain"]):
        return "응급의학과"
    elif any(x in name for x in ["therapeutic radiology", "radiation"]):
        return "방사선종양학과"
    else:
        return "기타 또는 미지정"

    


#####################
sorted(patientsfrequentdiseases[patientsfrequentdiseases["지역"] == "인천"]["상병명"].dropna().unique())
def map_disease_to_department(disease_name):
    manual_department_mapping = {
        "U07의 응급사용": "응급의학과",
        "감염성 및 상세불명 기원의 기타 위장염 및 대장염": "소화기내과",
        "급성 세기관지염": "호흡기내과",
        "기관지 및 폐의 악성 신생물": "호흡기내과",
        "기타 기능적 창자 장애": "소화기내과",
        "기타 단일 감염성 질환에 대한 예방접종의 필요": "감염내과",
        "기타 만성 폐쇄성 폐질환": "호흡기내과",
        "기타 척추병증": "정형외과",
        "기타 추간판장애": "신경외과",
        "기타관절증": "정형외과",
        "노인성 백내장": "안과",
        "뇌경색증": "신경과",
        "눈물기관의 장애": "안과",
        "달리 분류되지 않은 방광의 신경근육기능장애": "비뇨의학과",
        "달리 분류되지 않은 처치후 근육골격 장애": "정형외과",
        "달리 분류되지 않은 통증": "마취통증의학과",
        "담낭염": "소화기내과",
        "대퇴골의 골절": "정형외과",
        "목뼈원판 장애": "신경외과",
        "방광의 악성 신생물": "비뇨의학과",
        "배통": "정형외과",
        "본태성(일차성)고혈압": "순환기내과",
        "비뇨기관의 행동양식 불명 또는 미상의 신생물": "비뇨의학과",
        "상세불명 병원체의 폐렴, 제외: 보통간질(J84.1)": "호흡기내과",
        "상세불명의 당뇨병": "내분비내과",
        "양성 지방종성 신생물": "피부과",
        "어깨병변": "정형외과",
        "연기, 불 및 화염에 노출": "응급의학과",
        "위-식도역류병": "소화기내과",
        "위염 및 십이지장염": "소화기내과",
        "윤충증": "감염내과",
        "인슐린-비의존 당뇨병": "내분비내과",
        "자극성 접촉피부염": "피부과",
        "전립선의 악성 신생물": "비뇨의학과",
        "전립선증식증": "비뇨의학과",
        "전음성 및 감각신경성 청력소실": "이비인후과",
        "척수의 기타질환": "신경외과",
        "천식": "호흡기내과",
        "치수 및 치근단주위조직의 질환": "치과",
        "치아 및 지지구조의 기타 장애": "치과",
        "치아우식증": "치과",
        "치은염(잇몸염)및 치주 질환": "치과",
        "편마비": "신경과",
        "피부사상균증": "피부과",
        "헤르니아가 없는 마비성 장폐색증 및 창자폐쇄": "소화기내과",
        "기타 및 상세불명 부위의 제자리암종": "감염내과",
        "뇌내출혈": "신경외과",
        "늑골. 흉골 및 흉추골의 골절": "정형외과",
        "발목을 포함한 아래다리의 골절": "정형외과",
        "방광염": "비뇨의학과",
        "서혜헤르니아": "외과",
        "알레르기성 접촉피부염. 제외: 외이의 습진(H60.5) (Eczema of external ear)": "피부과",
        "알츠하이머병에서의 치매(G30.-+)": "신경과",
        "허리뼈 및 골반의 골절": "정형외과"
    }
    return manual_department_mapping.get(disease_name.strip(), "기타 또는 미지정")

sorted(patientsfrequentdiseases[patientsfrequentdiseases["지역"] == "광주"]["상병명"].dropna().unique())
def map_disease_gwangju(disease):
    if disease in ['감염성 및 상세불명 기원의 기타 위장염 및 결장염', '기타 단일 감염성 질환에 대한 예방접종의 필요', '상세불명 병원체의 폐렴', 'U07의 응급사용', '급성 코인두염 [감기]', '고체 및 액체에 의한 폐염']:
        return "호흡기내과"
    elif disease in ['기관지 및 폐의 악성 신생물', '간 및 간내담관의 악성 신생물', '결장의 악성 신생물', '위의 악성 신생물', '췌장의 악성신생물', '전립선의 악성 신생물']:
        return "혈액종양내과"
    elif disease in ['본태성(원발성) 고혈압', '심방세동 및 조동', '협심증']:
        return "내과"
    elif disease in ['만성 콩팥(신장)기능상실', '투석을 포함한 치료를 위하여 보건서비스와 접하고 있는 사람']:
        return "내과"
    elif disease in ['알츠하이머병에서의 치매(G30.-+)', '뇌경색증', '뇌내출혈', '파킨슨병', '편마비', '현기 및 어지러움', '인지기능 및 자각에 관한 기타 증상 및 징후']:
        return "신경과"
    elif disease in ['기타 척추병증', '기타 추간판장애', '어깨병변', '대퇴골의 골절', '허리뼈 및 골반의 골절']:
        return "정형외과"
    elif disease in ['노인성 백내장']:
        return "안과"
    elif disease in ['전립선증식증', '방광의 악성 신생물']:
        return "비뇨의학과"
    elif disease in ['담낭염', '담석증', '서혜헤르니아']:
        return "외과"
    elif disease in ['위-식도역류병']:
        return "소화기내과"
    elif disease in ['인슐린-비의존 당뇨병']:
        return "내과"
    elif disease in ['치은염(잇몸염) 및 치주질환', '치은염(잇몸염)및 치주 질환', '치아우식증']:
        return "치과"
    elif disease in ['피부사상균증']:
        return "피부과"
    elif disease in ['욕창성 및 압박부위 궤양']:
        return "재활의학과"
    elif disease in ['윤충증']:
        return "감염내과"
    elif disease in ['혈관운동성 및 앨러지성 비염']:
        return "이비인후과"
    elif disease in ['달리 분류되지 않은 사춘기의 장애']:
        return "정신건강의학과"
    elif disease in ['음식 및 수액섭취에 관계된 증상 및 징후']:
        return "가정의학과"
    else:
        return "기타 또는 미지정"

sorted(patientsfrequentdiseases[patientsfrequentdiseases["지역"] == "대전"]["상병명"].dropna().unique())
def map_disease_daejeon(disease):
    if any(x in disease for x in ["고혈압", "협심증", "심방세동", "심부전"]):
        return "내과"
    elif any(x in disease for x in ["당뇨", "인슐린", "2형"]):
        return "내과"
    elif any(x in disease for x in ["신장", "콩팥", "투석"]):
        return "신장내과"
    elif any(x in disease for x in ["폐렴", "기관지", "호흡장애"]):
        return "내과"
    elif any(x in disease for x in ["위염", "위", "장", "소화"]):
        return "내과"
    elif any(x in disease for x in ["뇌경색", "뇌내출혈", "뇌혈관", "어지럼", "하반신마비", "사지마비"]):
        return "신경과"
    elif any(x in disease for x in ["치매"]):
        return "정신건강의학과"
    elif any(x in disease for x in ["골절", "무릎", "척추", "관절", "등통증", "요추", "변형성", "추간판", "어깨"]):
        return "정형외과"
    elif any(x in disease for x in ["백내장"]):
        return "안과"
    elif any(x in disease for x in ["전립선"]):
        return "비뇨의학과"
    elif any(x in disease for x in ["치아", "치주", "치은", "우식"]):
        return "치과"
    elif any(x in disease for x in ["피부", "가려움"]):
        return "피부과"
    elif any(x in disease for x in ["담석", "담낭", "췌장", "결장", "위의 악성", "결장의악성신생물"]):
        return "외과"
    elif any(x in disease for x in ["감염", "신종질환", "예방접종", "기생충"]):
        return "가정의학과"
    elif any(x in disease for x in ["사타구니탈장", "서혜헤르니아"]):
        return "외과"
    elif any(x in disease for x in ["재활", "욕창", "근골격", "하반신마비"]):
        return "재활의학과"
    else:
        return "기타 또는 미지정"

sorted(patientsfrequentdiseases[patientsfrequentdiseases["지역"] == "대구"]["상병명"].dropna().unique())
def map_disease_daegu(disease):
    if any(x in disease for x in ["고혈압", "협심증", "심방세동", "심부전", "죽상경화"]):
        return "순환기내과"
    elif any(x in disease for x in ["당뇨", "인슐린", "2형"]):
        return "내분비내과"
    elif any(x in disease for x in ["신장", "콩팥", "투석"]):
        return "신장내과"
    elif any(x in disease for x in ["폐렴", "코로나", "감염", "기생충"]):
        return "감염내과"
    elif any(x in disease for x in ["위염", "위", "장", "소화"]):
        return "소화기내과"
    elif any(x in disease for x in ["뇌경색", "뇌출혈", "뇌혈관", "어지럼", "다발신경병", "인지", "치매"]):
        return "신경과"
    elif any(x in disease for x in ["골절", "무릎", "척추", "관절", "요추", "어깨", "통증"]):
        return "정형외과"
    elif any(x in disease for x in ["백내장"]):
        return "안과"
    elif any(x in disease for x in ["전립선"]):
        return "비뇨의학과"
    elif any(x in disease for x in ["치아", "치주", "치은", "우식", "근단", "지지구조"]):
        return "치과"
    elif any(x in disease for x in ["피부", "백선", "사상균", "가려움"]):
        return "피부과"
    elif any(x in disease for x in ["담석", "담낭", "위의 악성", "결장", "위-식도역류", "사타구니탈장"]):
        return "외과"
    elif any(x in disease for x in ["재활", "욕창", "사지마비", "처치후 근골격"]):
        return "재활의학과"
    elif any(x in disease for x in ["알츠하이머", "치매", "정신", "인지"]):
        return "정신건강의학과"
    elif any(x in disease for x in ["의학적 관찰", "보건서비스", "예방접종"]):
        return "가정의학과"
    else:
        return "기타 또는 미지정"

sorted(patientsfrequentdiseases[patientsfrequentdiseases["지역"] == "부산"]["상병명"].dropna().unique())
def map_busan_disease_to_department(disease):
    if any(x in disease for x in ["당뇨", "인슐린"]):
        return "내과"
    elif any(x in disease for x in ["간", "담관", "췌장", "위염", "위", "장", "소화"]):
        return "외과"
    elif any(x in disease for x in ["기관지", "폐", "폐렴", "코로나", "감염", "기생충"]):
        return "내과"
    elif any(x in disease for x in ["요추", "척추", "등통증", "무릎", "골절", "관절", "어깨"]):
        return "정형외과"
    elif any(x in disease for x in ["뇌", "편마비", "마비", "경색", "출혈"]):
        return "신경과"
    elif any(x in disease for x in ["치은", "치아", "우식", "치주"]):
        return "치과"
    elif any(x in disease for x in ["백내장", "망막", "눈물", "안과"]):
        return "안과"
    elif any(x in disease for x in ["피부염", "사상균", "가려움", "피부"]):
        return "피부과"
    elif any(x in disease for x in ["전립선", "방광", "비뇨"]):
        return "비뇨의학과"
    elif any(x in disease for x in ["재활", "사지마비", "하반신마비"]):
        return "재활의학과"
    elif any(x in disease for x in ["정신", "인지", "피로", "우울", "불안"]):
        return "정신건강의학과"
    elif any(x in disease for x in ["심부전", "협심증", "심방세동", "심장"]):
        return "내과"
    elif any(x in disease for x in ["코", "이비인후", "호흡"]):
        return "이비인후과"
    elif "검사" in disease:
        return "진단검사의학과"
    else:
        return "가정의학과"
#합쳐서 새로운 컬럼 추가
region_mapper = {
    "서울": map_seoul_disease_to_department,
    "부산": map_busan_disease_to_department,
    "대전": map_disease_daegu,
    "광주": map_disease_gwangju,
    "대구": map_disease_daegu,
    "인천": map_disease_to_department,
}
def apply_department(row):
    region = row["지역"]
    disease = row["상병명"]
    mapper = region_mapper.get(region)
    if mapper:
        return mapper(disease.strip())
    else:
        return "미지정"
patientsfrequentdiseases["진료과"] = patientsfrequentdiseases.apply(apply_department, axis=1)

sorted(chronic[chronic["지역"] == "인천"]["상병명"].dropna().unique())
def map_chronic_incheon_disease_to_department(disease_name):
    mapping = {
        "간의 질환": "소화기내과",           # 간염, 간경변 등
        "갑상선의 장애": "가정의학과",       # 내분비외래 없을 시 대체
        "고혈압": "순환기내과",
        "당뇨병": "가정의학과",              # 내분비내과 없어서 대체
        "대뇌혈관질환": "신경과",             # 뇌졸중 등
        "만성신부전증": "가정의학과",         # 신장내과 없음 → 대체
        "신경계질환": "신경과",
        "심장질환": "순환기내과",
        "악성신생물": "외과",                # 위암, 대장암, 간암 등
        "정신 및 행동장애": "가정의학과",     # 정신건강의학과 없음
        "호흡기결핵": "호흡기내과"
    }
    return mapping.get(disease_name.strip(), "기타 또는 미지정")

sorted(chronic[chronic["지역"] == "서울"]["상병명"].dropna().unique())
def map_chronic_seoul_disease_to_department(disease_name):
    mapping = {
        "간의 질환": "소화기내과",           # 간염, 간경변 등
        "갑상선의 장애": "내분비내과",
        "고혈압": "순환기내과",
        "당뇨병": "내분비내과",
        "대뇌혈관질환": "신경과",            # 뇌졸중 등
        "만성신부전증": "신장내과",
        "신경계질환": "신경과",
        "심장질환": "순환기내과",
        "악성신생물": "혈액종양내과",        # 암센터/종양내과 존재
        "정신 및 행동장애": "정신건강의학과",
        "호흡기결핵": "호흡기내과"
    }
    return mapping.get(disease_name.strip(), "기타 또는 미지정")

sorted(chronic[chronic["지역"] == "광주"]["상병명"].dropna().unique())
def map_chronic_gwangju_disease_to_department(disease_name):
    mapping = {
        "간의 질환": "내과",              # 간염, 간경변 등 → 내과
        "갑상선의 장애": "가정의학과",      # 내분비내과 없음 → 가정의학과
        "고혈압": "내과",                # 순환기계 질환 포함
        "당뇨병": "가정의학과",           # 내분비계 → 가정의학과로 대체
        "대뇌혈관질환": "신경과",          # 뇌졸중 등
        "만성신부전증": "내과",           # 신장내과 없음
        "신경계질환": "신경과",
        "심장질환": "내과",
        "악성신생물": "혈액종양내과",       # 암 전문과 존재
        "정신 및 행동장애": "정신건강의학과",
        "호흡기결핵": "내과"              # 호흡기내과 없음
    }
    return mapping.get(disease_name.strip(), "기타 또는 미지정")

sorted(chronic[chronic["지역"] == "대전"]["상병명"].dropna().unique())
def map_chronic_daejeon_disease_to_department(disease_name):
    mapping = {
        "간의 질환": "내과",              # 간염, 간경변 등
        "갑상선의 장애": "가정의학과",      # 내분비내과 없음 → 가정의학과
        "고혈압": "내과",                # 순환기 질환
        "당뇨병": "가정의학과",           # 내분비계 → 가정의학과
        "대뇌혈관질환": "신경과",          # 뇌졸중 등
        "만성신부전증": "신장내과",        # 신장내과 존재
        "신경계질환": "신경과",
        "심장질환": "내과",
        "악성신생물": "외과",             # 혈액종양내과 없음
        "정신 및 행동장애": "정신건강의학과",
        "호흡기결핵": "내과"              # 호흡기내과 없음
    }
    return mapping.get(disease_name.strip(), "기타 또는 미지정")

sorted(chronic[chronic["지역"] == "대구"]["상병명"].dropna().unique())
def map_chronic_daegu_disease_to_department(disease_name):
    mapping = {
        "간의 질환": "소화기내과",         # 간염, 간경변 등
        "갑상선의 장애": "내분비내과",
        "고혈압": "순환기내과",
        "당뇨병": "내분비내과",
        "대뇌혈관질환": "신경과",
        "만성신부전증": "신장내과",
        "신경계질환": "신경과",
        "심장질환": "순환기내과",
        "악성신생물": "내과",             # 혈액종양내과 없음
        "정신 및 행동장애": "정신건강의학과",
        "호흡기결핵": "감염내과"          # 감염내과 존재
    }
    return mapping.get(disease_name.strip(), "기타 또는 미지정")

sorted(chronic[chronic["지역"] == "부산"]["상병명"].dropna().unique())
def map_chronic_busan_disease_to_department(disease_name):
    mapping = {
        "간의 질환": "내과",              # 간염, 간경변
        "갑상선의 장애": "내과",          # 내분비 대체
        "고혈압": "내과",                # 순환기 질환
        "당뇨병": "내과",                # 내분비 대체
        "대뇌혈관질환": "신경과",
        "만성신부전증": "내과",           # 신장 대체
        "신경계질환": "신경과",
        "심장질환": "내과",
        "악성신생물": "외과",             # 종양 치료 목적
        "정신 및 행동장애": "정신건강의학과",
        "호흡기결핵": "내과"              # 감염/호흡기내과 없음
    }
    return mapping.get(disease_name.strip(), "기타 또는 미지정")

#합쳐서 새로운 컬럼 추가
region_mapper = {
    "서울": map_chronic_seoul_disease_to_department,
    "부산": map_chronic_busan_disease_to_department,
    "대전": map_chronic_daejeon_disease_to_department,
    "광주": map_chronic_gwangju_disease_to_department,
    "대구": map_chronic_daegu_disease_to_department,
    "인천": map_chronic_incheon_disease_to_department,
}
def apply_department(row):
    region = row["지역"]
    disease = row["상병명"]
    mapper = region_mapper.get(region)
    if mapper:
        return mapper(disease.strip())
    else:
        return "미지정"
chronic["진료과"] = chronic.apply(apply_department, axis=1)

# 다빈도 질환: 지역 + 년도 + 상병명 + 진료과 기준 실인원 합계
frequent_grouped = (
    patientsfrequentdiseases
    .groupby(["지역", "년도", "상병명", "진료과"])["실인원"]
    .sum()
    .reset_index()
)
frequent_top = frequent_grouped.sort_values(
    ["지역", "년도", "실인원", "진료과"], ascending=[True, True, False, True]
)

# 만성질환: 지역 + 년도 + 상병명 + 진료과 기준 연령별_합계
chronic_grouped = (
    chronic
    .groupby(["지역", "년도", "상병명", "진료과"])["연령별_합계"]
    .sum()
    .reset_index()
)
chronic_top = chronic_grouped.sort_values(
    ["지역", "년도", "연령별_합계", "진료과"], ascending=[True, True, False, True]
)