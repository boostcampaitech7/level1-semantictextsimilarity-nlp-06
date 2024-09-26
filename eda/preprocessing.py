#--------------------- py-hanspell 설치 ---------------------#
# 유지보수 안되고 있는 패키지라서 pip install도 안되고, 실행 파일도 직접 수정해줘야 하는 번거로움이 있습니다
# 그래도 현재 이용 가능한 맞춤법 검사 라이브러리 중 가장 성능이 좋고 많이 쓰이는 라이브러리라서,
# 본 코드에서는 번거로움을 감수하고.. 해당 라이브러리를 이용합니다.
# 원활한 실행을 위해 우선 다음을 실행한 뒤, 본 파이썬 파일을 실행해주세요.

#### 1. git-clone해 py-hanspell 설치
# git clone https://github.com/ssut/py-hanspell.git
# cd py-hanspell
# python setup.py install
# mv hanspell /opt/conda/lib/python3.10/site-packages
# cd ..
# rm -r py-hanspell

#### 2. `./hanspell/spell_checker.py` 파일에서 payload에 "passportKey" 추가
# cd /opt/conda/lib/python3.10/site-packages/hanspell
# vi spell_checker.py
# (payload 부분 다음과 같이 수정: passportKey 추가)
# payload = {
#     'passportKey': '',
#     'color_blindness': '0',
#     'q': text
# }
#-----------------------------------------------------------#

import re
import requests
import pandas as pd
import emoji
from soynlp.normalizer import repeat_normalize
from hanspell import spell_checker
from tqdm import tqdm


# 원본 데이터 Load
data_train = pd.read_csv("../../data/train.csv")
data_dev = pd.read_csv("../../data/dev.csv")
data_test = pd.read_csv("../../data/test.csv")

# data cleansing
def clean(x): 
    x = emoji.replace_emoji(x, replace='')  # 이모지 제거
    x = repeat_normalize(x, num_repeats=2)  # 4글자 이상 반복되는 글자 2번으로 압축 (ㅋㅋㅋㅋ, ㅠㅠㅠㅠ, 와하하하하핫 -> ㅋㅋ, ㅠㅠ, 와하하핫)
    x = re.sub(r'([~!?.\-_;:@#$%&*()=+])\1{2,}', r'\1\1', x)  # 3번 이상 반복되는 특수문자 2번으로 압축
    x = x.strip() # 문자열 양쪽의 공백 제거
    x = re.sub(r'\s{2,}', ' ', x)  # 두 번 이상 반복되는 공백 ' '로 압축
    return x

# 네이버에서 '네이버 맞춤법 검사기' 페이지에서 passportKey를 획득
def get_passport_key():
    url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=네이버+맞춤법+검사기"
    res = requests.get(url)
    html_text = res.text
    match = re.search(r'passportKey=([^&"}]+)', html_text)
    if match:
        passport_key = match.group(1)
        return passport_key
    else:
        return False

# 획득한 passportkey를 spell_checker.py파일에 적용
def fix_spell_checker_py_code(file_path, passportKey):
    pattern = r"'passportKey': '.*'"
    with open(file_path, 'r', encoding='utf-8') as input_file:
        content = input_file.read()
        modified_content = re.sub(pattern, f"'passportKey': '{passportKey}'", content)
    with open(file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(modified_content)
    return 

# 맞춤법 교정 + 띄어쓰기
def han_checker(text):
    try:
        result = spell_checker.check(text)
        return result.checked
    except Exception as e:  # '&' 처리 못함 (에러 발생)
        return text  # 오류 발생 시 원본 텍스트 반환

# 데이터프레임에 전처리 함수 적용
def process_sentences(df):
    for col in ['sentence_1', 'sentence_2']:
        tqdm.pandas(desc=f"Processing {col}")
        df[col] = df[col].progress_apply(clean).progress_apply(han_checker)
    return df


# hanspell 밑작업 (매 번 Passport key 입력해줘야 에러 안남)
spell_checker_file_path = '/opt/conda/lib/python3.10/site-packages/hanspell/spell_checker.py'
passport_key = get_passport_key()
fix_spell_checker_py_code(spell_checker_file_path, passport_key)


# train, dev 데이터셋 전처리 및 저장

# training set
print("Processing train dataset:")
data_train = process_sentences(data_train)
print("\nSaving processed data...")
data_train.to_csv("../../data/train_preprocess_v1.csv", index=False)
print("Data saving completed.\n")

# dev set
print("\nProcessing dev dataset:")
data_dev = process_sentences(data_dev)
print("\nSaving processed data...")
data_dev.to_csv("../../data/dev_preprocess_v1.csv", index=False)
print("Data saving completed.\n")

# test set
print("\nProcessing test dataset:")
data_test = process_sentences(data_test)
print("\nSaving processed data...")
data_test.to_csv("../../data/test_preprocess_v1.csv", index=False)
print("Data saving completed.\n")


