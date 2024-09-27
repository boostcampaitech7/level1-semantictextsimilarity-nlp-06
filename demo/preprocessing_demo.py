import re
import requests
import pandas as pd
import emoji
from soynlp.normalizer import repeat_normalize
from hanspell import spell_checker
from tqdm import tqdm


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

def preprocessing_texts(text):
    text_clean = clean(text)
    text_spell_check = han_checker(text_clean)
    return text_spell_check


# hanspell 밑작업 (매 번 Passport key 입력해줘야 에러 안남)
spell_checker_file_path = '/opt/conda/lib/python3.10/site-packages/hanspell/spell_checker.py'
passport_key = get_passport_key()
fix_spell_checker_py_code(spell_checker_file_path, passport_key)
