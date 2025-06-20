import pandas as pd

beds_list = []

# 광주 시설
gwangju_beds = {
    '병원명': '광주',
    '병상종류': [
        '일반입원실_상급', '일반입원실_일반', '중환자실_성인', '중환자실_소아', '중환자실_신생아',
        '정신과개방_상급', '정신과개방_일반', '정신과폐쇄_상급', '정신과폐쇄_일반',
        '격리병실', '무균치료실', '분만실', '수술실', '응급실', '물리치료실',
        '신생아실', '회복실', '인공신장실', '강내치료실', '방사선옥소'
    ],
    '병상수': [10, 502, 26, 0, 0, 0, 0, 0, 0, 2, 0, 2, 8, 16, 36, 5, 7, 31, 0, 0]
}
gwangju_doctors = {
    '병원명': '광주',
    '진료과목': ['내과', '신경과', '정신건강의학과', '외과', '정형외과', '신경외과', '마취통증의학과', '산부인과', '소아청소년과', '안과', 
              '이비인후과', '피부과', '비뇨의학과', '영상의학과', '병리과', '진단검사의학과', '재활의학과', '핵의학과', '가정의학과', '응급의학과', '치과', '한방내과'],
    '전문의수': [11, 3, 1, 2, 2, 2, 3, 2, 2, 2, 2, 1, 4, 4, 1, 1, 4, 1, 3, 3, 0, 0]
}
gwangju_staff = {
    '병원명': '광주',
    '직종': ['의사', '간호사', '사회복지사'],
    '인원수': [3, 17, 2]
}
gwangju_other_staff = {
    '병원명': '광주',
    '직종': ['약사', '한약사', '사회복지사', '물리치료사', '작업치료사'],
    '인원수': [23, 0, 5, 23, 7]
}
gwangju_facilities = {
    '병원명': '광주',
    '시설명': ['가족실', '임종실', '간호사실', '화장실', '목욕실', '상담실', '처치실'],
    '개수': [1, 2, 1, 2, 0, 1, 1]
}
df_beds_gj = pd.DataFrame(gwangju_beds)
df_doctors_gj = pd.DataFrame(gwangju_doctors)
df_staff_gj = pd.DataFrame(gwangju_staff)
df_other_staff_gj = pd.DataFrame(gwangju_other_staff)
df_facilities_gj = pd.DataFrame(gwangju_facilities)

# 서울 시설
seoul_beds = {
    '병원명': '서울',
    '병상종류': [
        '일반입원실_상급', '일반입원실_일반', '중환자실_성인', '중환자실_소아', '중환자실_신생아',
        '정신과개방_상급', '정신과개방_일반', '정신과폐쇄_상급', '정신과폐쇄_일반',
        '격리병실', '무균치료실', '분만실', '수술실', '응급실', '물리치료실',
        '신생아실', '회복실', '인공신장실', '강내치료실', '방사선옥소'
    ],
    '병상수': [5, 830, 54, 0, 0, 0, 6, 0, 60, 28, 2, 1, 15, 25, 25, 6, 10, 25, 0, 0]
}
seoul_doctors = {
    '병원명': '서울',
    '진료과목': [
        '내과', '신경과', '정신건강의학과', '외과', '정형외과', '신경외과', '심장혈관흉부외과', '성형외과', '마취통증의학과', 
        '산부인과', '소아청소년과', '안과', '이비인후과', '피부과', '비뇨의학과', '영상의학과', '방사선종양학과', '병리과', 
        '진단검사의학과', '재활의학과', '핵의학과', '가정의학과', '응급의학과', '구강악안면외과', '치과보철과', '치주과', 
        '치과보존과', '구강내과', '통합치의학과', '한방재활의학과'
    ],
    '전문의수': [46, 8, 3, 7, 8, 4, 3, 1, 7, 2, 2, 6, 5, 3, 7, 7, 2, 2, 3, 6, 1, 6, 11, 3, 9, 5, 4, 1, 2, 1]
}
seoul_staff = {
    '병원명': '서울',
    '직종': ['의사', '간호사', '사회복지사'],
    '인원수': [5, 17, 3]
}
seoul_other_staff = {
    '병원명': '서울',
    '직종': ['약사', '한약사', '사회복지사', '물리치료사', '작업치료사'],
    '인원수': [57, 0, 13, 48, 14]
}
seoul_facilities = {
    '병원명': '서울',
    '시설명': ['가족실', '임종실', '간호사실', '화장실', '목욕실', '상담실', '처치실'],
    '개수': [1, 2, 1, 2, 0, 1, 1]
}
df_beds_seoul = pd.DataFrame(seoul_beds)
df_doctors_seoul = pd.DataFrame(seoul_doctors)
df_staff_seoul = pd.DataFrame(seoul_staff)
df_other_staff_seoul = pd.DataFrame(seoul_other_staff)
df_facilities_seoul = pd.DataFrame(seoul_facilities)

# 대전 시설
daejeon_beds = {
    '병원명': '대전',
    '병상종류': [
        '일반입원실_상급', '일반입원실_일반', '중환자실_성인', '중환자실_소아', '중환자실_신생아',
        '정신과개방_상급', '정신과개방_일반', '정신과폐쇄_상급', '정신과폐쇄_일반',
        '격리병실', '무균치료실', '분만실', '수술실', '응급실', '물리치료실',
        '신생아실', '회복실', '인공신장실', '강내치료실', '방사선옥소'
    ],
    '병상수': [4, 355, 19, 0, 0, 0, 0, 0, 0, 9, 0, 8, 5, 9, 58, 15, 1, 13, 0, 0]
}
daejeon_doctors = {
    '병원명': '대전',
    '진료과목': [
        '내과', '신경과', '정신건강의학과', '외과', '정형외과', '신경외과', '마취통증의학과', '산부인과',
        '소아청소년과', '안과', '이비인후과', '피부과', '비뇨의학과', '영상의학과', '진단검사의학과',
        '재활의학과', '가정의학과', '응급의학과', '구강악안면외과', '한방내과'
    ],
    '전문의수': [6, 3, 1, 2, 3, 2, 3, 1, 1, 1, 1, 1, 3, 2, 1, 1, 3, 0, 0, 0]
}
daejeon_staff = {
    '병원명': '대전',
    '직종': ['의사', '간호사', '사회복지사'],
    '인원수': [2, 12, 3]
}
daejeon_other_staff = {
    '병원명': '대전',
    '직종': ['약사', '한약사', '사회복지사', '물리치료사', '작업치료사'],
    '인원수': [13, 0, 5, 14, 4]
}
daejeon_facilities = {
    '병원명': '대전',
    '시설명': ['가족실', '임종실', '간호사실', '화장실', '목욕실', '상담실', '처치실'],
    '개수': [1, 2, 1, 6, 0, 1, 1]
}
df_beds_dj = pd.DataFrame(daejeon_beds)
df_doctors_dj = pd.DataFrame(daejeon_doctors)
df_staff_dj = pd.DataFrame(daejeon_staff)
df_other_staff_dj = pd.DataFrame(daejeon_other_staff)
df_facilities_dj = pd.DataFrame(daejeon_facilities)

# 부산 시설
busan_beds = {
    '병원명': '부산',
    '병상종류': [
        '일반입원실_상급', '일반입원실_일반', '중환자실_성인', '중환자실_소아', '중환자실_신생아',
        '정신과개방_상급', '정신과개방_일반', '정신과폐쇄_상급', '정신과폐쇄_일반',
        '격리병실', '무균치료실', '분만실', '수술실', '응급실', '물리치료실',
        '신생아실', '회복실', '인공신장실', '강내치료실', '방사선옥소'
    ],
    '병상수': [7, 426, 24, 0, 0, 0, 0, 0, 0, 19, 0, 1, 7, 13, 33, 0, 4, 26, 0, 0]
}
busan_doctors = {
    '병원명': '부산',
    '진료과목': [
        '내과', '신경과', '정신건강의학과', '외과', '정형외과', '신경외과', '심장혈관흉부외과', '마취통증의학과', '산부인과',
        '소아청소년과', '안과', '이비인후과', '피부과', '비뇨의학과', '영상의학과', '병리과', '진단검사의학과',
        '재활의학과', '가정의학과', '치주과', '통합치의학과', '치과보철과', '한방내과'
    ],
    '전문의수': [14, 2, 2, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 5, 3, 1, 1, 2, 5, 1, 0, 0]
}
while len(busan_doctors['전문의수']) < len(busan_doctors['진료과목']):
    busan_doctors['전문의수'].append(0)
busan_staff = {
    '병원명': '부산',
    '직종': ['의사', '간호사', '사회복지사'],
    '인원수': [1, 17, 1]
}
busan_other_staff = {
    '병원명': '부산',
    '직종': ['약사', '한약사', '사회복지사', '물리치료사', '작업치료사'],
    '인원수': [19, 0, 4, 22, 4]
}
busan_facilities = {
    '병원명': '부산',
    '시설명': ['가족실', '임종실', '간호사실', '화장실', '목욕실', '상담실', '처치실'],
    '개수': [2, 1, 1, 2, 0, 1, 1]
}
df_beds_bs = pd.DataFrame(busan_beds)
df_doctors_bs = pd.DataFrame(busan_doctors)
df_staff_bs = pd.DataFrame(busan_staff)
df_other_staff_bs = pd.DataFrame(busan_other_staff)
df_facilities_bs = pd.DataFrame(busan_facilities)

# 인천 시설
incheon_beds = {
    '병원명': '인천',
    '병상종류': [
        '일반입원실_상급', '일반입원실_일반', '중환자실_성인', '중환자실_소아', '중환자실_신생아',
        '정신과개방_상급', '정신과개방_일반', '정신과폐쇄_상급', '정신과폐쇄_일반',
        '격리병실', '무균치료실', '분만실', '수술실', '응급실', '물리치료실',
        '신생아실', '회복실', '인공신장실', '강내치료실', '방사선옥소'
    ],
    '병상수': [6, 124, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 13, 0, 1, 0, 0, 0]
}
incheon_doctors = {
    '병원명': '인천',
    '진료과목': [
        '내과', '신경과', '외과', '정형외과', '신경외과', '마취통증의학과', '안과', '이비인후과',
        '피부과', '비뇨의학과', '영상의학과', '진단검사의학과', '재활의학과', '가정의학과',
        '구강악안면외과', '치과보철과', '치주과', '치과보존과', '예방치과'
    ],
    '전문의수': [3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
}
incheon_staff = {
    '병원명': '인천',
    '직종': ['의사', '간호사', '사회복지사'],
    '인원수': [0, 0, 0]  # 인력현황 별도 없으니 0 또는 필요시 조정 가능
}
incheon_other_staff = {
    '병원명': '인천',
    '직종': ['약사', '한약사', '사회복지사', '물리치료사', '작업치료사'],
    '인원수': [5, 0, 2, 10, 3]
}
incheon_facilities = {
    '병원명': '인천',
    '시설명': [],  # 제공된 시설현황 데이터 없음
    '개수': []
}
df_beds_ic = pd.DataFrame(incheon_beds)
df_doctors_ic = pd.DataFrame(incheon_doctors)
df_staff_ic = pd.DataFrame(incheon_staff)
df_other_staff_ic = pd.DataFrame(incheon_other_staff)
df_facilities_ic = pd.DataFrame(incheon_facilities)

# 대구 시설
daegu_beds = {
    '병원명': '대구',
    '병상종류': [
        '일반입원실_상급', '일반입원실_일반', '중환자실_성인', '중환자실_소아', '중환자실_신생아',
        '정신과개방_상급', '정신과개방_일반', '정신과폐쇄_상급', '정신과폐쇄_일반',
        '격리병실', '무균치료실', '분만실', '수술실', '응급실', '물리치료실',
        '신생아실', '회복실', '인공신장실', '강내치료실', '방사선옥소'
    ],
    '병상수': [14, 416, 23, 0, 0, 0, 0, 0, 0, 3, 0, 1, 6, 14, 39, 0, 5, 28, 0, 0]
}
daegu_doctors = {
    '병원명': '대구',
    '진료과목': [
        '내과', '신경과', '정신건강의학과', '외과', '정형외과', '신경외과', '심장혈관흉부외과', '마취통증의학과',
        '산부인과', '소아청소년과', '안과', '이비인후과', '피부과', '비뇨의학과', '영상의학과',
        '진단검사의학과', '재활의학과', '핵의학과', '가정의학과', '응급의학과', '치과보철과',
        '치주과', '치과보존과', '구강내과', '영상치의학과', '통합치의학과'
    ],
    '전문의수': [9, 2, 1, 3, 4, 3, 1, 2, 1, 1, 2, 3, 2, 1, 3, 1, 2, 0, 4, 0, 0, 0, 1, 0, 2]
}
while len(daegu_doctors['전문의수']) < len(daegu_doctors['진료과목']):
    daegu_doctors['전문의수'].append(0)
daegu_staff = {
    '병원명': '대구',
    '직종': ['의사', '간호사', '사회복지사'],
    '인원수': [2, 12, 2]
}
daegu_other_staff = {
    '병원명': '대구',
    '직종': ['약사', '한약사', '사회복지사', '물리치료사', '작업치료사'],
    '인원수': [13, 0, 5, 20, 5]
}
daegu_facilities = {
    '병원명': '대구',
    '시설명': ['가족실', '임종실', '간호사실', '화장실', '목욕실', '상담실', '처치실'],
    '개수': [1, 1, 1, 2, 0, 1, 1]
}
df_beds_dg = pd.DataFrame(daegu_beds)
df_doctors_dg = pd.DataFrame(daegu_doctors)
df_staff_dg = pd.DataFrame(daegu_staff)
df_other_staff_dg = pd.DataFrame(daegu_other_staff)
df_facilities_dg = pd.DataFrame(daegu_facilities)

regions = ['gj', 'seoul', 'bs', 'dg', 'dj', 'ic']

total_list = []
for region in regions:
    try:
        # 병상
        df_beds = globals()[f'df_beds_{region}']
        beds_pivot = df_beds.pivot_table(index='병원명', columns='병상종류', values='병상수', aggfunc='sum')

        # 진료과
        df_doctors = globals()[f'df_doctors_{region}']
        doctors_pivot = df_doctors.pivot_table(index='병원명', columns='진료과목', values='전문의수', aggfunc='sum')
        doctors_pivot.columns = [f"{col}_전문의수" for col in doctors_pivot.columns]

        # 인력
        df_staff = globals().get(f'df_staff_{region}', pd.DataFrame())
        df_other_staff = globals().get(f'df_other_staff_{region}', pd.DataFrame())
        staff_all = pd.concat([df_staff, df_other_staff], ignore_index=True)
        staff_pivot = staff_all.pivot_table(index='병원명', columns='직종', values='인원수', aggfunc='sum')
        staff_pivot.columns = [f"{col}_인원수" for col in staff_pivot.columns]

        # 시설
        df_facilities = globals()[f'df_facilities_{region}']
        facilities_pivot = df_facilities.pivot_table(index='병원명', columns='시설명', values='개수', aggfunc='sum')

        # 병합
        merged = beds_pivot \
            .join(doctors_pivot, how='outer') \
            .join(staff_pivot, how='outer') \
            .join(facilities_pivot, how='outer')
        
        merged.fillna(0, inplace=True)
        total_list.append(merged)

    except KeyError as e:
        print(f"{region} 지역 데이터 누락: {e}")
    except Exception as e:
        print(f"{region} 처리 중 오류 발생: {e}")

if total_list:
    total_df = pd.concat(total_list)
    total_df.reset_index(inplace=True)
    total_df.to_csv("new_merged_data/병원_통합_데이터.csv", index=False, encoding='utf-8-sig')
    print("CSV 파일이 생성되었습니다: 병원_통합_데이터.csv")
else:
    print("데이터가 없습니다.")