import os

def convert_euckr_to_utf8_in_folder(folder_path):
    """
    지정된 폴더 내의 모든 파일을 EUC-KR에서 UTF-8로 변환합니다.
    주의: 원본 파일을 직접 덮어씁니다. 실행 전 반드시 백업하세요.
    """
    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}'는 유효한 폴더가 아닙니다.")
        return

    print(f"주의: '{folder_path}' 폴더의 파일들을 EUC-KR에서 UTF-8로 변환합니다.")
    print("이 작업은 원본 파일을 직접 수정합니다. 계속하기 전에 반드시 백업하세요!")
    confirm = input("계속하시겠습니까? (y/n): ").lower()
    if confirm != 'y':
        print("작업이 취소되었습니다.")
        return

    converted_files = 0
    failed_files = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 파일인 경우에만 처리 (하위 폴더는 무시)
        if os.path.isfile(file_path):
            print(f"처리 중: {file_path}")
            try:
                # 1. EUC-KR로 파일 읽기
                with open(file_path, 'r', encoding='euc-kr') as f:
                    content = f.read()

                # 2. UTF-8로 파일 쓰기 (같은 파일에 덮어쓰기)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"성공: '{filename}' 변환 완료 (EUC-KR -> UTF-8)")
                converted_files += 1

            except UnicodeDecodeError:
                print(f"오류: '{filename}'은(는) EUC-KR로 인코딩된 파일이 아니거나 손상되었습니다. 건너뜁니다.")
                failed_files += 1
            except Exception as e:
                print(f"오류: '{filename}' 처리 중 예외 발생: {e}. 건너뜁니다.")
                failed_files += 1
        # else:
        #     print(f"건너뜀 (폴더): {filename}") # 하위 폴더는 건너뜀을 명시

    print("\n--- 작업 완료 ---")
    print(f"총 {converted_files}개 파일 변환 성공.")
    print(f"총 {failed_files}개 파일 변환 실패 또는 건너뜀.")

if __name__ == "__main__":
    # 여기에 변환할 폴더 경로를 입력하세요.
    # 예: target_folder = r"C:\Users\YourName\Documents\MyFolder"
    target_folder = "data/비급여정보 UTF-8"
    # target_folder_path = input("EUC-KR 파일을 UTF-8로 변환할 폴더 경로를 입력하세요: ")
    
    convert_euckr_to_utf8_in_folder(target_folder)
