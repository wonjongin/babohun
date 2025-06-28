import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import unicodedata

def clean_number_string(value):
    """숫자 문자열에서 쉼표 제거 및 정수 변환"""
    if pd.isna(value) or value == '-' or value == '':
        return 0
    
    # 문자열로 변환
    value_str = str(value).strip()
    
    # 쉼표 제거
    value_str = value_str.replace(',', '')
    
    # 숫자가 아닌 문자 제거 (공백, 특수문자 등)
    value_str = re.sub(r'[^\d]', '', value_str)
    
    if value_str == '':
        return 0
    
    return int(value_str)

def extract_region_from_filename(filename):
    """파일명에서 지역명 추출"""
    # 파일명에서 지역명 패턴 찾기
    region_patterns = {
        '인천': '인천',
        '대전': '대전', 
        '대구': '대구',
        '광주': '광주',
        '부산': '부산',
        '중앙': '중앙'
    }
    
    # 파일명을 NFC 정규화 (macOS 한글 파일명 문제 해결)
    normalized_filename = unicodedata.normalize('NFC', filename)
    
    for pattern, region in region_patterns.items():
        if pattern in normalized_filename:
            return region
    
    # 디버깅을 위해 파일명 출력
    print(f"    파일명: {filename}")
    print(f"    정규화된 파일명: {normalized_filename}")
    
    return '기타'

def read_hospital_data(file_path, region):
    """개별 병원 데이터 파일 읽기"""
    
    try:
        # 파일을 먼저 텍스트로 읽어서 구조 확인
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 헤더 구조 확인
        if len(lines) >= 2:
            header1 = lines[0].strip().split(',')
            header2 = lines[1].strip().split(',')
            
            print(f"  헤더1: {header1}")
            print(f"  헤더2: {header2}")
            
            # 실제 데이터는 3번째 줄부터 시작
            data_lines = lines[2:]
            
            # 데이터 파싱
            parsed_data = []
            for line in data_lines:
                if line.strip():  # 빈 줄 제외
                    # 쉼표로 분리하되, 따옴표 안의 쉼표는 무시
                    values = []
                    current_value = ""
                    in_quotes = False
                    
                    for char in line.strip():
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            values.append(current_value.strip())
                            current_value = ""
                        else:
                            current_value += char
                    
                    # 마지막 값 추가
                    values.append(current_value.strip())
                    
                    if len(values) >= 10:  # 최소 10개 컬럼 필요
                        parsed_data.append(values[:10])  # 처음 10개만 사용
            
            # DataFrame 생성
            df = pd.DataFrame(parsed_data, columns=[
                '구분', '국비_남', '국비_여', '국비_계', 
                '사비_남', '사비_여', '사비_계', 
                '합계_남', '합계_여', '합계_계'
            ])
            
            # 숫자 데이터 정리
            numeric_columns = ['국비_남', '국비_여', '국비_계', '사비_남', '사비_여', '사비_계', '합계_남', '합계_여', '합계_계']
            
            for col in numeric_columns:
                df[col] = df[col].apply(clean_number_string)
            
            # 지역 정보 추가
            df['지역'] = region
            
            # '계' 행 제거 (마지막 행)
            df = df[df['구분'] != '계'].copy()
            
            # 연령대 정규화
            df['연령대'] = df['구분'].apply(normalize_age_group)
            
            return df
            
    except Exception as e:
        print(f"파일 읽기 오류 ({file_path}): {e}")
        return None

def normalize_age_group(age_str):
    """연령대 문자열 정규화"""
    if pd.isna(age_str):
        return '기타'
    
    age_str = str(age_str).strip()
    
    # 연령대 매핑
    age_mapping = {
        '0~9': '0대',
        '10~19': '10대', 
        '20~29': '20대',
        '30~39': '30대',
        '40~49': '40대',
        '50~59': '50대',
        '60~69': '60대',
        '70~79': '70대',
        '80~89': '80대',
        '90세 이상': '90대'
    }
    
    return age_mapping.get(age_str, age_str)

def merge_all_hospital_data():
    """모든 병원 데이터 통합"""
    
    # 데이터 폴더 경로
    data_folder = Path('data/성별연령대별 보훈병원이용자')
    
    # 결과를 저장할 리스트
    all_data = []
    
    # 각 파일 처리
    for file_path in data_folder.glob('*.csv'):
        print(f"처리 중: {file_path.name}")
        
        # 파일명에서 지역 추출
        region = extract_region_from_filename(file_path.name)
        
        # 데이터 읽기
        df = read_hospital_data(file_path, region)
        
        if df is not None:
            all_data.append(df)
            print(f"  - {region}: {len(df)}개 연령대")
        else:
            print(f"  - {region}: 읽기 실패")
    
    # 모든 데이터 통합
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 컬럼 순서 정리
        column_order = ['지역', '연령대', '구분', '국비_남', '국비_여', '국비_계', 
                       '사비_남', '사비_여', '사비_계', '합계_남', '합계_여', '합계_계']
        
        # 존재하는 컬럼만 선택
        existing_columns = [col for col in column_order if col in merged_df.columns]
        merged_df = merged_df[existing_columns]
        
        return merged_df
    else:
        return None

def create_summary_statistics(merged_df):
    """통합 데이터 요약 통계 생성"""
    
    # 지역별 총 이용자 수
    region_summary = merged_df.groupby('지역').agg({
        '합계_계': 'sum'
    }).reset_index()
    region_summary.columns = ['지역', '총이용자수']
    
    # 연령대별 총 이용자 수
    age_summary = merged_df.groupby('연령대').agg({
        '합계_계': 'sum'
    }).reset_index()
    age_summary.columns = ['연령대', '총이용자수']
    
    # 지역별 연령대별 이용자 수
    region_age_summary = merged_df.groupby(['지역', '연령대']).agg({
        '합계_계': 'sum'
    }).reset_index()
    region_age_summary.columns = ['지역', '연령대', '이용자수']
    
    return region_summary, age_summary, region_age_summary

def create_gender_statistics(merged_df):
    """성별 분포 통계 생성"""
    # 전체 성별 합계
    total_gender = pd.DataFrame({
        '성별': ['남', '여'],
        '이용자수': [merged_df['합계_남'].sum(), merged_df['합계_여'].sum()]
    })
    # 연령대별 성별 합계
    age_gender = merged_df.groupby('연령대').agg({'합계_남': 'sum', '합계_여': 'sum'}).reset_index()
    age_gender = age_gender.melt(id_vars='연령대', var_name='성별', value_name='이용자수')
    age_gender['성별'] = age_gender['성별'].map({'합계_남': '남', '합계_여': '여'})
    # 지역별 성별 합계
    region_gender = merged_df.groupby('지역').agg({'합계_남': 'sum', '합계_여': 'sum'}).reset_index()
    region_gender = region_gender.melt(id_vars='지역', var_name='성별', value_name='이용자수')
    region_gender['성별'] = region_gender['성별'].map({'합계_남': '남', '합계_여': '여'})
    # 지역·연령대별 성별 합계
    region_age_gender = merged_df.groupby(['지역', '연령대']).agg({'합계_남': 'sum', '합계_여': 'sum'}).reset_index()
    region_age_gender = region_age_gender.melt(id_vars=['지역', '연령대'], var_name='성별', value_name='이용자수')
    region_age_gender['성별'] = region_age_gender['성별'].map({'합계_남': '남', '합계_여': '여'})
    return total_gender, age_gender, region_gender, region_age_gender

def main():
    """메인 실행 함수"""
    print("=== 성별연령대별 보훈병원 이용자 데이터 통합 ===\n")
    # 1. 모든 데이터 통합
    print("1. 데이터 통합 중...")
    merged_df = merge_all_hospital_data()
    if merged_df is None:
        print("데이터 통합 실패!")
        return None, None, None, None
    print(f"   통합 완료: {len(merged_df)}개 행")
    print(f"   지역 수: {merged_df['지역'].nunique()}개")
    print(f"   연령대 수: {merged_df['연령대'].nunique()}개")
    # 2. 데이터 확인
    print("\n2. 통합된 데이터 미리보기:")
    print(merged_df.head())
    print("\n3. 지역별 데이터 수:")
    print(merged_df['지역'].value_counts())
    print("\n4. 연령대별 데이터 수:")
    print(merged_df['연령대'].value_counts())
    # 3. 요약 통계 생성
    print("\n5. 요약 통계 생성 중...")
    region_summary, age_summary, region_age_summary = create_summary_statistics(merged_df)
    print("\n   지역별 총 이용자 수:")
    print(region_summary)
    print("\n   연령대별 총 이용자 수:")
    print(age_summary)
    # 4. 성별 통계 생성
    print("\n6. 성별 통계 생성 중...")
    total_gender, age_gender, region_gender, region_age_gender = create_gender_statistics(merged_df)
    print("   전체 성별 이용자수:")
    print(total_gender)
    print("   연령대별 성별 이용자수:")
    print(age_gender.head())
    print("   지역별 성별 이용자수:")
    print(region_gender.head())
    print("   지역·연령대별 성별 이용자수:")
    print(region_age_gender.head())
    # 5. 파일 저장
    print("\n7. 파일 저장 중...")
    output_folder = Path('new_merged_data')
    output_folder.mkdir(exist_ok=True)
    merged_file = output_folder / '성별연령대별_보훈병원이용자_통합.csv'
    merged_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
    print(f"   통합 데이터 저장: {merged_file}")
    region_summary_file = output_folder / '지역별_총이용자수.csv'
    region_summary.to_csv(region_summary_file, index=False, encoding='utf-8-sig')
    print(f"   지역별 요약 저장: {region_summary_file}")
    age_summary_file = output_folder / '연령대별_총이용자수.csv'
    age_summary.to_csv(age_summary_file, index=False, encoding='utf-8-sig')
    print(f"   연령대별 요약 저장: {age_summary_file}")
    region_age_summary_file = output_folder / '지역별연령대별_이용자수.csv'
    region_age_summary.to_csv(region_age_summary_file, index=False, encoding='utf-8-sig')
    print(f"   지역별연령대별 요약 저장: {region_age_summary_file}")
    # 성별 통계 저장
    total_gender_file = output_folder / '전체_성별_이용자수.csv'
    total_gender.to_csv(total_gender_file, index=False, encoding='utf-8-sig')
    print(f"   전체 성별 저장: {total_gender_file}")
    age_gender_file = output_folder / '연령대별_성별_이용자수.csv'
    age_gender.to_csv(age_gender_file, index=False, encoding='utf-8-sig')
    print(f"   연령대별 성별 저장: {age_gender_file}")
    region_gender_file = output_folder / '지역별_성별_이용자수.csv'
    region_gender.to_csv(region_gender_file, index=False, encoding='utf-8-sig')
    print(f"   지역별 성별 저장: {region_gender_file}")
    region_age_gender_file = output_folder / '지역별연령대별_성별_이용자수.csv'
    region_age_gender.to_csv(region_age_gender_file, index=False, encoding='utf-8-sig')
    print(f"   지역별연령대별 성별 저장: {region_age_gender_file}")
    # 6. 최종 확인
    print("\n=== 통합 완료 ===")
    print(f"총 {len(merged_df)}개 행의 데이터가 통합되었습니다.")
    print(f"지역: {', '.join(merged_df['지역'].unique())}")
    print(f"연령대: {', '.join(sorted(merged_df['연령대'].unique()))}")
    return merged_df, region_summary, age_summary, region_age_summary

if __name__ == "__main__":
    merged_data, region_sum, age_sum, region_age_sum = main()
