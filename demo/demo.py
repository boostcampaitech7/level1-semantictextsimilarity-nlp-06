import os
import time
import yaml
from box import Box
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
from bertviz import head_view

import streamlit as st
import streamlit.components.v1 as components

import preprocessing_demo as preprocessing



st.set_page_config(layout="wide", page_title="문장 간 유사도 측정 데모")

# 함수 정의 --------------------------------------------------------------------------------------------------
def load_config(config_file):
    # Load Config.yaml
    with open(config_file) as file:
        config = yaml.safe_load(file) # Dictionary
        config = Box(config)
    return config

def create_hyperparameter_dataframe(config):
    # Extract values from config
    data = {
        'Parameter': ['Batch Size', 'Max Epoch', 'Learning Rate', 'Loss', 'Optimizer', 'Weight Decay', 'Scheduler'],
        'Value': [
            config.training.batch_size,
            config.training.max_epoch,
            '{:.0e}'.format(float(config.training.learning_rate)),
            config.training.loss if 'loss' in config.training else config.training.criterion,
            config.training.optimization.name if 'optimization' in config.training \
                else config.training.optimizer.name,
            '{:.0e}'.format(float(config.training.optimization.weight_decay)) if 'optimization' in config.training \
                else '{:.0e}'.format(float(config.training.optimizer.weight_decay)),
            config.training.scheduler.name
        ]
    }
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def configurer(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
        config = Box(config)

    return config

def simulate(config_paths, sentence_1, sentence_2, is_voting):
    sequence = '[SEP]'.join([sentence_1, sentence_2])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_prediction = None
    for config_path in config_paths:
        config = configurer(config_path)

        print(f"Now Using {config.model_name}")
        
        pt_name = config.model_name.split('/')[1] + '.pt'
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, max_length=160)
        model = torch.load(os.path.join('../huggingface_code/models', pt_name))

        tokens = tokenizer(sequence, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True)

        model.to(device)
        tokens.to(device)

        model.eval()
        with torch.no_grad():
            pred = model(**tokens).logits.cpu().numpy()
        
        if base_prediction is None:
            base_prediction = pred
        else:
            base_prediction = np.concatenate((base_prediction, pred), axis=1)
        # del model
        # torch.cuda.empty_cache()
    
    label = None
    if is_voting:
        label = np.mean(base_prediction, axis=1)
    else:
        classifier = LinearRegression()
        label = classifier.predict(base_prediction)
    label = np.round(label, 1)
    return base_prediction, label


def show_head_view(model, tokenizer, sentence_a, sentence_b):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True).to('cuda')
    # 입력 ids
    input_ids = inputs['input_ids']
    token_type_ids = inputs.get('token_type_ids', None)
    uses_token_type_ids = torch.any(token_type_ids != 0)

    # 모델이 token_type_ids를 사용하는지 확인 (Roberta 모델은 사용 안함 -> 다 0으로 처리됨)
    if uses_token_type_ids:
        token_type_ids = inputs['token_type_ids']
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    else:
        token_type_ids = None
        attention = model(input_ids)[-1]
    
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    # sentence_b의 시작 인덱스 결정
    if uses_token_type_ids:
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        # RoBERTa는 token_type_ids를 사용하지 않으므로 SEP 토큰의 위치를 이용하여 구분
        sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        sentence_b_start = input_id_list.index(sep_token_id, 1) + 1  # 첫 SEP 이후의 인덱스
    
    figure = head_view(attention, tokens, sentence_b_start=sentence_b_start, html_action='return')
    return figure


# 데이터 준비 -------------------------------------------------------------------------------------------------

# 테스트 데이터 가져오기 (랜덤 문장 쌍 생성 위함)
@st.cache_data
def load_test_data():
    return pd.read_csv("../../data/test.csv")

@st.cache_data
def load_dev_data():
    return pd.read_csv("../../data/dev.csv")

@st.cache_resource
def load_models_and_tokenizers():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer_deberta = AutoTokenizer.from_pretrained(deberta_config.model_name, max_length=160)
    tokenizer_roberta = AutoTokenizer.from_pretrained(roberta_config.model_name, max_length=160)
    tokenizer_electra_kr = AutoTokenizer.from_pretrained(electra_kr_config.model_name, max_length=160)
    tokenizer_electra_kor = AutoTokenizer.from_pretrained(electra_kor_config.model_name, max_length=160)
    
    model_deberta = torch.load(deberta_path + ".pt").to(device)
    model_roberta = torch.load(roberta_path + ".pt").to(device)
    model_electra_kr = torch.load(electra_kr_path + ".pt").to(device)
    model_electra_kor = torch.load(electra_kor_path + ".pt").to(device)
    
    model_deberta.config.output_attentions = True
    model_roberta.config.output_attentions = True
    model_electra_kr.config.output_attentions = True
    model_electra_kor.config.output_attentions = True
    
    return (tokenizer_deberta, tokenizer_roberta, tokenizer_electra_kr, tokenizer_electra_kor,
            model_deberta, model_roberta, model_electra_kr, model_electra_kor)

# 테스트 데이터 불러오기
# test_data = load_test_data()
dev_data = load_dev_data()

# 모델 저장 경로 지정
deberta_path = "../huggingface_code/models/kf-deberta-base-cross-sts"
roberta_path = "../huggingface_code/models/roberta-large"
electra_kr_path = "../huggingface_code/models/KR-ELECTRA-discriminator"
electra_kor_path = "../huggingface_code/models/electra-kor-base"

# config 파일 불러오기
deberta_config = load_config(deberta_path + ".yaml")
roberta_config = load_config(roberta_path + ".yaml")
electra_kr_config = load_config(electra_kr_path + ".yaml")
electra_kor_config = load_config(electra_kor_path + ".yaml")
config_paths = [roberta_path+'.yaml', deberta_path+'.yaml', electra_kr_path+'.yaml', electra_kor_path+'.yaml']

# 모델 및 토크나이저 로드
(tokenizer_deberta, tokenizer_roberta, tokenizer_electra_kr, tokenizer_electra_kor,\
 model_deberta, model_roberta, model_electra_kr, model_electra_kor) = load_models_and_tokenizers()

# 하이퍼파라미터 표 생성
hyperparameter_deberta = create_hyperparameter_dataframe(deberta_config)
hyperparameter_roberta = create_hyperparameter_dataframe(roberta_config)
hyperparameter_electra_kr = create_hyperparameter_dataframe(electra_kr_config)
hyperparameter_electra_kor = create_hyperparameter_dataframe(electra_kor_config)





# 페이지 디자인 -------------------------------------------------------------------------------------------------

# 사이드바 설정
st.sidebar.title("Project 1")
st.sidebar.subheader("문장 간 유사도 측정")

# 사이드바의 옵션
page = st.sidebar.radio("", ["모델 구조", "Label 예측하기"])

# 페이지 내용 설정
if page == "모델 구조":
    st.markdown("<p style='font-size: 20px;'><strong>[NLP-06]</strong> Project 1 - 문장 간 유사도 측정</p>", unsafe_allow_html=True)
    st.title('Semantic Textual Similarity 데모')
    st.markdown("<p style='font-size: 24px; font-weight: bold;'>모델 구조 및 성능 소개</p>", unsafe_allow_html=True)

    st.markdown(""); st.markdown("")
    st.subheader("모델 구조")
    image_col1, image_col2, image_col3 = st.columns([0.5, 8, 1.5])
    with image_col2:
        st.image("./image/flowchart.png")

    st.subheader("")
    st.subheader("하이퍼파라미터")
    hyper_col1, hyper_col2, hyper_col3, hyper_col4 = st.columns(4)

    # 가운데 정렬을 위한 CSS 스타일 추가
    center_style = '<div style="text-align: center;">'

    with hyper_col1:
        st.markdown("<center><p style='font-size: 21px; font-weight: bold;'>DeBERTa</p></center>", unsafe_allow_html=True)
        st.dataframe(hyperparameter_deberta.set_index(hyperparameter_deberta.columns[0]), width=1000)

    with hyper_col2:
        st.markdown("<center><p style='font-size: 21px; font-weight: bold;'>RoBERTa</p></center>", unsafe_allow_html=True)
        st.dataframe(hyperparameter_roberta.set_index(hyperparameter_roberta.columns[0]), width=1000)

    with hyper_col3:
        st.markdown("<center><p style='font-size: 21px; font-weight: bold;'>KR-ELECTRA</p></center>", unsafe_allow_html=True)
        st.dataframe(hyperparameter_electra_kr.set_index(hyperparameter_electra_kr.columns[0]), width=1000)

    with hyper_col4:
        st.markdown("<center><p style='font-size: 21px; font-weight: bold;'>ELECTRA-KOR</p></center>", unsafe_allow_html=True)
        st.dataframe(hyperparameter_electra_kor.set_index(hyperparameter_electra_kor.columns[0]), width=1000)


    st.subheader("")
    st.subheader("성능: 피어슨 상관계수")
    st.markdown("")
    
    st.markdown("최종 리더보드 결과입니다. (최종 순위 `10/16`)")
    st.markdown("baseline으로 제시되었던 성능으로부터 0.1760을 개선했습니다. (피어슨 상관계수: `0.7618` -> `0.9378`)")
    st.image("./image/leaderboard_final.png")

    st.markdown("");st.markdown("")
    st.markdown("한편, private 리더보드에서보다 public 리더보드에서 더 높은 성능이 확인되었습니다.")
    st.markdown("이를 통해 우리 모델은 새로운 데이터에 대해 잘 일반화될 수 있는 견고한 모델임을 알 수 있습니다.")
    st.image("./image/leaderboard_final2.png")
    



elif page == "Label 예측하기":

    st.markdown("""
        <style>
            .centered {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }
            .left-aligned {
                text-align: left;
            }
            .label {
                font-size: 28px;
                font-weight: bold;
            }
            .predicted-number {
                font-size: 36px;
                font-weight: bold;
            }
            .real-number {
                font-size: 36px;
                font-weight: normal;
            }
        </style>
        """, unsafe_allow_html=True)
        

    # 변수 초기화
    if 'sentence_1' not in st.session_state:
        st.session_state.sentence_1 = ""
    if 'sentence_2' not in st.session_state:
        st.session_state.sentence_2 = ""
    if 'prediction_sentence_1' not in st.session_state:
        st.session_state.prediction_sentence_1 = ""
    if 'prediction_sentence_2' not in st.session_state:
        st.session_state.prediction_sentence_2 = ""
    
    if 'true_label' not in st.session_state:
        st.session_state.true_label = None
    if 'predicted_label' not in st.session_state:
        st.session_state.predicted_label = None
    if 'predicted_label_deberta' not in st.session_state:
        st.session_state.predicted_label_deberta = None
    if 'predicted_label_roberta' not in st.session_state:
        st.session_state.predicted_label_roberta = None
    if 'predicted_label_electra_kr' not in st.session_state:
        st.session_state.predicted_label_electra_kr = None
    if 'predicted_label_electra_kor' not in st.session_state:
        st.session_state.predicted_label_electra_kor = None

    if 'true_label_color' not in st.session_state:
        st.session_state.true_label_color = ''
    if 'label_color' not in st.session_state:
        st.session_state.label_color = ''
    if 'label_color_deberta' not in st.session_state:
        st.session_state.label_color_deberta = ''
    if 'label_color_roberta' not in st.session_state:
        st.session_state.label_color_roberta = ''
    if 'label_color_electra_kr' not in st.session_state:
        st.session_state.label_color_electra_kr = ''
    if 'label_color_electra_kor' not in st.session_state:
        st.session_state.label_color_electra_kor = ''

    st.markdown("<p style='font-size: 20px;'><strong>[NLP-06]</strong> Project 1 - 문장 간 유사도 측정</p>", unsafe_allow_html=True)
    st.title('Semantic Textual Similarity 데모')
    st.markdown("<p style='font-size: 24px; font-weight: bold;'>두 문장의 유사도 계산 예시</p>", unsafe_allow_html=True)


    st.subheader("")
    st.subheader("한국어 문장 쌍의 Label 예측")

    if st.button("문장 쌍 랜덤으로 가져오기 (`dev.csv`)", key="random_button"):
        random_row = dev_data.sample(n=1).iloc[0]
        st.session_state.sentence_1 = random_row['sentence_1']
        st.session_state.sentence_2 = random_row['sentence_2']
        st.session_state.true_label = random_row['label']
        st.session_state.true_label_color = 'green' if float(st.session_state.true_label) >= 2.5 else 'red'

    sentence1 = st.text_input("문장1", value=st.session_state.get('sentence_1', ''), key='sentence1_input', placeholder="첫번째 문장을 입력해주세요")
    sentence2 = st.text_input("문장2", value=st.session_state.get('sentence_2', ''), key='sentence2_input', placeholder="두번째 문장을 입력해주세요")
        

    st.text('')
    if st.button("Label 값 예측하기", key="prediction_button", use_container_width=True):
        with st.spinner('Prediction 중...'):  
            # 입력된 문장
            st.session_state.sentence_1 = sentence1
            st.session_state.sentence_2 = sentence2

            # 텍스트 전처리
            st.session_state.prediction_sentence_1 = preprocessing.preprocessing_texts(st.session_state.sentence_1)
            st.session_state.prediction_sentence_2 = preprocessing.preprocessing_texts(st.session_state.sentence_2)
            
            # 예측 수행
            model_results, final_results = simulate(config_paths, st.session_state.prediction_sentence_1, st.session_state.prediction_sentence_2, is_voting=True)
            st.session_state.predicted_label_deberta = f"{round(model_results[0][0], 1):.1f}"
            st.session_state.predicted_label_roberta = f"{round(model_results[0][1], 1):.1f}"
            st.session_state.predicted_label_electra_kr = f"{round(model_results[0][2], 1):.1f}"
            st.session_state.predicted_label_electra_kor = f"{round(model_results[0][3], 1):.1f}"
            st.session_state.predicted_label = f"{round(final_results[0], 1):.1f}"

        if 'predicted_label' in st.session_state and st.session_state.predicted_label is not None:
            # label 별 색상 지정
            st.session_state.label_color = 'green' if float(st.session_state.predicted_label) >= 2.5 else 'red'
            st.session_state.label_color_deberta = 'green' if float(st.session_state.predicted_label_deberta) >= 2.5 else 'red'
            st.session_state.label_color_roberta = 'green' if float(st.session_state.predicted_label_roberta) >= 2.5 else 'red'
            st.session_state.label_color_electra_kr = 'green' if float(st.session_state.predicted_label_electra_kr) >= 2.5 else 'red'
            st.session_state.label_color_electra_kor = 'green' if float(st.session_state.predicted_label_electra_kor) >= 2.5 else 'red'

            st.markdown(f"""
            <style>
                .centered {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                }}
                .left-aligned {{
                    text-align: left;
                    margin-top: 20px; /* Adjust this value to increase the space between labels and sentences */
                }}
                .label {{
                    font-size: 28px;
                    font-weight: bold;
                }}
                .predicted-number {{
                    font-size: 36px;
                    font-weight: bold;
                }}
                .real-label {{
                    font-size: 36px;
                    font-weight: normal;
                }}
            </style>

            <div class='centered'>
                <p class='label'>
                    <span style='color: black;'>예측된 Label : </span>
                    <span class='predicted-number' style='color: {st.session_state.label_color};'>{st.session_state.predicted_label}</span>
                </p>
                <p class='label' style='font-weight: normal; margin-top: -10px;'> <!-- Decrease margin-top to bring labels closer -->
                    <span style='color: black;'>실제 Label : </span>
                    <span class='real-label' style='color: {st.session_state.true_label_color};'>{st.session_state.true_label}</span>
                </p>
                <div class='left-aligned'>
                    <p><strong>문장 1</strong>: {st.session_state.sentence_1}</p>
                    <p><strong>문장 2</strong>: {st.session_state.sentence_2}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 점수 시각화 막대 그래프 생성
            fig, ax = plt.subplots(figsize=(8, 0.3))  # 바 높이를 얇게 설정
            ax.barh([0], [5], color='#e9ecef', height=0.2)  # 회색 배경색 지정: 가능한 점수 범위 시각적 전달 위함
            ax.barh([0], [float(st.session_state.predicted_label)], color=st.session_state.label_color, height=0.2)
            ax.set_xlim(0, 5)
            ax.set_yticks([])
            ax.set_xticks([0, 2.5, 5])
            ax.set_xticklabels(['0.0', '2.5', '5.0'])

            plt.axvline(x=2.5, color='gray', linewidth=1, linestyle='-')  # 중앙선

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(False)
            ax.yaxis.set_ticks_position('none')

            plot_col1, plot_col2, plot_col3 = st.columns([0.5, 9, 0.5])
            with plot_col1:
                st.markdown("")
                st.markdown("<center><p style='font-size: 12px; font-weight: bold; color: red;'>다른 의미</p></center>", unsafe_allow_html=True)
            with plot_col2:
                st.markdown("<div class='centered'>", unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
            with plot_col3:
                st.markdown("")
                st.markdown("<center><p style='font-size: 12px; font-weight: bold; color: green;'>같은 의미</p></center>", unsafe_allow_html=True)

            st.info("""##### Label 값 의미
- **0** : 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음
- **1** : 두 문자의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음
- **2** : 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함
- **3** : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음
- **4** : 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음
- **5** : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함""")

        st.header("\n")
        st.subheader("최종 Label 예측 과정 시각화")
        predict_col1, predict_col2, predict_col3, predict_col4 = st.columns(4)

        # 가운데 정렬을 위한 CSS 스타일 추가
        center_style = '<div style="text-align: center;">'


        # 4개 모델에 대한 head_view 생성
        with predict_col1:
            st.markdown("### <center>DeBERTa</center>", unsafe_allow_html=True)
            html_output_deberta = show_head_view(model_deberta, tokenizer_deberta, \
                        st.session_state.prediction_sentence_1, st.session_state.prediction_sentence_2)
            components.html(html_output_deberta.data, height=450, scrolling=True)
            st.markdown("<center>predicted label</center>", unsafe_allow_html=True)
            st.markdown(f"<center><p style='font-size: 30px; font-weight: bold; color: {st.session_state.label_color_deberta};'>{st.session_state.predicted_label_deberta}</p></center>", unsafe_allow_html=True)

        with predict_col2:
            st.markdown("### <center>RoBERTa</center>", unsafe_allow_html=True)
            html_output_roberta = show_head_view(model_roberta, tokenizer_roberta, \
                        st.session_state.prediction_sentence_1, st.session_state.prediction_sentence_2)
            components.html(html_output_roberta.data, height=450, scrolling=True)
            st.markdown("<center>predicted label</center>", unsafe_allow_html=True)
            st.markdown(f"<center><p style='font-size: 30px; font-weight: bold; color: {st.session_state.label_color_roberta};'>{st.session_state.predicted_label_roberta}</p></center>", unsafe_allow_html=True)
        
        with predict_col3:
            st.markdown("### <center>KR-ELECTRA</center>", unsafe_allow_html=True)
            html_output_electra_kr = show_head_view(model_electra_kr, tokenizer_electra_kr, \
                        st.session_state.prediction_sentence_1, st.session_state.prediction_sentence_2)
            components.html(html_output_electra_kr.data, height=450, scrolling=True)
            st.markdown("<center>predicted label</center>", unsafe_allow_html=True)
            st.markdown(f"<center><p style='font-size: 30px; font-weight: bold; color: {st.session_state.label_color_electra_kr};'>{st.session_state.predicted_label_electra_kr}</p></center>", unsafe_allow_html=True)
        
        with predict_col4:
            st.markdown("### <center>ELECTRA-KOR</center>", unsafe_allow_html=True)
            html_output_electra_kor = show_head_view(model_electra_kor, tokenizer_electra_kor, \
                        st.session_state.prediction_sentence_1, st.session_state.prediction_sentence_2)
            components.html(html_output_electra_kor.data, height=450, scrolling=True)
            st.markdown("<center>predicted label</center>", unsafe_allow_html=True)
            st.markdown(f"<center><p style='font-size: 30px; font-weight: bold; color: {st.session_state.label_color_electra_kor};'>{st.session_state.predicted_label_electra_kor}</p></center>", unsafe_allow_html=True)

        # 화살표 입력
        st.markdown("")
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image('./image/arrow_softvoting.png', use_column_width=True)
        st.markdown("")

        # 앙상블 통해 도출된 최종 label
        st.markdown("<center>final predicted label (true label)</center>", unsafe_allow_html=True)
        st.markdown(f"""
                <div style='text-align: center; white-space: nowrap;'>
                    <span style='font-size: 48px; font-weight: bold; color: {st.session_state.label_color};'>{st.session_state.predicted_label}</span>
                    <span style='font-size: 48px; color: black;'> (</span>
                    <span style='font-size: 48px; color: {st.session_state.true_label_color};'>{st.session_state.true_label}</span>
                    <span style='font-size: 48px; color: black;'>)</span>
                </div>
            """, unsafe_allow_html=True)
        