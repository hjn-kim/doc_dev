# -*- coding: utf-8 -*-
"""메신저 검색 성능 평가 - Streamlit 리포트

표와 예시는 산출물에서 뽑아 하드코딩했다. 파일 없이도 그대로 열린다.
값을 갱신하려면 파이프라인을 다시 돌린 뒤 DETAIL 상수만 바꾸면 된다.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from shared_utils import apply_common_styles

apply_common_styles()

st.markdown('<h1 class="main-title">메신저 검색 성능 평가</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">BAAI/bge-m3 (dense + sparse), nlpai-lab/KURE-v1 (dense)</p>',
            unsafe_allow_html=True)

SOURCE_URL = ("https://github.com/NorthwaveSecurity/"
              "complete_translation_leaked_chats_conti_ransomware/tree/main")

METRICS = ["H@5", "H@10", "R@5", "R@10", "MRR@10", "nDCG@10"]
CONFIGS = ["bb-dense", "bb-hybrid", "bk", "kb", "kk"]
QTYPES = ["1 의미기반", "2 식별자+의미기반", "3 화자기반", "4 날짜기반"]

# result(2).txt 의 구성별 표. {구성: {행 이름: METRICS 순서의 값}}
DETAIL = {
    "bb-dense": {
        "jabber_en": [0.650, 0.720, 0.537, 0.600, 0.508, 0.559],
        "jabber_ru": [0.660, 0.710, 0.523, 0.575, 0.564, 0.600],
        "1 의미기반": [0.460, 0.520, 0.352, 0.392, 0.361, 0.400],
        "2 식별자+의미기반": [0.740, 0.760, 0.610, 0.657, 0.664, 0.688],
        "3 화자기반": [0.820, 0.880, 0.685, 0.769, 0.638, 0.697],
        "4 날짜기반": [0.600, 0.700, 0.474, 0.532, 0.481, 0.533],
        "ALL": [0.655, 0.715, 0.530, 0.587, 0.536, 0.580],
    },
    "bb-hybrid": {
        "jabber_en": [0.700, 0.770, 0.603, 0.677, 0.596, 0.638],
        "jabber_ru": [0.700, 0.770, 0.572, 0.638, 0.601, 0.642],
        "1 의미기반": [0.480, 0.520, 0.362, 0.412, 0.407, 0.435],
        "2 식별자+의미기반": [0.840, 0.880, 0.742, 0.786, 0.718, 0.758],
        "3 화자기반": [0.800, 0.940, 0.701, 0.839, 0.674, 0.737],
        "4 날짜기반": [0.680, 0.740, 0.545, 0.595, 0.595, 0.630],
        "ALL": [0.700, 0.770, 0.587, 0.658, 0.598, 0.640],
    },
    "bk": {
        "jabber_en": [0.630, 0.740, 0.538, 0.629, 0.546, 0.592],
        "jabber_ru": [0.650, 0.720, 0.517, 0.602, 0.542, 0.584],
        "1 의미기반": [0.460, 0.560, 0.365, 0.445, 0.335, 0.389],
        "2 식별자+의미기반": [0.740, 0.780, 0.643, 0.723, 0.683, 0.707],
        "3 화자기반": [0.780, 0.900, 0.631, 0.756, 0.647, 0.708],
        "4 날짜기반": [0.580, 0.680, 0.471, 0.539, 0.510, 0.550],
        "ALL": [0.640, 0.730, 0.528, 0.616, 0.544, 0.588],
    },
    "kb": {
        "jabber_en": [0.630, 0.710, 0.524, 0.598, 0.498, 0.549],
        "jabber_ru": [0.700, 0.740, 0.553, 0.609, 0.580, 0.619],
        "1 의미기반": [0.460, 0.520, 0.340, 0.390, 0.350, 0.391],
        "2 식별자+의미기반": [0.740, 0.760, 0.637, 0.663, 0.644, 0.673],
        "3 화자기반": [0.820, 0.900, 0.678, 0.811, 0.670, 0.726],
        "4 날짜기반": [0.640, 0.720, 0.501, 0.549, 0.493, 0.549],
        "ALL": [0.665, 0.725, 0.539, 0.603, 0.539, 0.584],
    },
    "kk": {
        "jabber_en": [0.710, 0.750, 0.590, 0.641, 0.587, 0.627],
        "jabber_ru": [0.690, 0.740, 0.547, 0.633, 0.570, 0.611],
        "1 의미기반": [0.540, 0.580, 0.420, 0.450, 0.422, 0.461],
        "2 식별자+의미기반": [0.760, 0.780, 0.670, 0.717, 0.680, 0.704],
        "3 화자기반": [0.860, 0.920, 0.699, 0.829, 0.673, 0.733],
        "4 날짜기반": [0.640, 0.700, 0.485, 0.553, 0.539, 0.578],
        "ALL": [0.700, 0.745, 0.569, 0.637, 0.579, 0.619],
    },
}
NDCG = METRICS.index("nDCG@10")

TYPE_EX = [
    ("유형 1 — 의미 기반", "고유명사를 모두 빼고 범죄 정황만으로 검색",
     "피해 조직의 서버와 컴퓨터 현황을 파악하고 내부 문서를 탈취한 정황에 대한 증거를 찾아주세요.", 25),
    ("유형 2 — 식별자+의미기반", "파일명·해시·탐지명 등 고유 식별자가 질문에 들어간다",
     "68.224.217.72와 CALAHANLAW가 등장하는 기록에서 피해 조직의 서버와 컴퓨터 현황을 파악하고 내부 문서를 탈취한 정황에 대한 증거를 찾아주세요.", 25),
    ("유형 3 — 화자 + 의미 기반", "대화 참여자 두 명을 지정한다",
      "ahtyng와 alarm 사이의 대화에서 피해 조직의 서버와 컴퓨터 현황을 파악하고 내부 문서를 탈취한 정황에 대한 증거를 찾아주세요.", 25),
    ("유형 4 — 날짜 + 의미 기반", "대화가 오간 날짜를 지정한다",
     "2020-09-29에 피해 조직의 서버와 컴퓨터 현황을 파악하고 내부 문서를 탈취한 정황에 대한 증거를 찾아주세요.", 25),
]


def frame(rows, columns):
    st.dataframe(pd.DataFrame(rows, columns=columns),
                 hide_index=True, use_container_width=True)


def table(rows: list[str], index_name: str, metric: str = "nDCG@10"):
    """행=비교 대상, 열=구성. 맨 뒤에 어느 구성이 가장 높았는지 이름으로 적는다."""
    i = METRICS.index(metric)
    df = pd.DataFrame({c: [DETAIL[c][r][i] for r in rows] for c in CONFIGS},
                      index=rows)
    df.index.name = index_name
    df["가장 높은 구성"] = df[CONFIGS].idxmax(axis=1)
    return df


def show(df: pd.DataFrame, numeric: list[str] | None = None):
    cols = numeric or CONFIGS
    st.dataframe(df.style.format({c: "{:.3f}" for c in cols}),
                 use_container_width=True)


tab_result, tab_data, tab_prep, tab_chunking, tab_eval, tab_model, tab_metadata = st.tabs(
    ["1. 결과", "2. 데이터셋", "3. 전처리", "4. 청킹",
     "5. 평가 방식", "6. 모델", "7. 메타데이터"])


with tab_result:
    st.markdown("### 1. 전체")
    overall = pd.DataFrame({c: DETAIL[c]["ALL"] for c in CONFIGS}, index=METRICS)
    overall.index.name = "지표"
    overall["가장 높은 구성"] = overall[CONFIGS].idxmax(axis=1)
    show(overall)

    st.markdown("### 2. 한국어")
    frame([["한국어 코퍼스 없음"] + ["—"] * len(CONFIGS)], ["코퍼스"] + CONFIGS)

    st.markdown("### 3. 외국어")
    show(table(["jabber_ru", "jabber_en"], "코퍼스")
         .rename(index={"jabber_ru": "원문 (러시아어)", "jabber_en": "번역본 (영어)"}))

    st.markdown("### 4. 질의 유형별")
    show(table(QTYPES, "질의 유형"))

    with st.expander("구성별 전체 지표 보기"):
        for cfg in CONFIGS:
            st.markdown(f"**{cfg}**")
            d = pd.DataFrame(DETAIL[cfg], index=METRICS).T
            d.index.name = "구분"
            show(d, numeric=METRICS)


with tab_data:

    st.markdown("### 데이터셋")

    st.markdown(
    f"**`jabberchat2020.csv`** — 실제 Conti 랜섬웨어 팀이 범행을 협의하며 약 1년간 "
    f"사용한 1:1 메신저 기록이다. 원문은 러시아어, 번역본은 영어다.\n\n"
    f"출처: {SOURCE_URL}")
    
    c = st.columns(3)
    c[0].metric("메시지", "107,967")
    c[1].metric("계정", "295")
    c[2].metric("대화 창", "1,113")

    st.markdown("### preview")
    frame([
        ["42498", "00:08:51", "target → barmen",
         "give decryption"],
        ["42499", "00:08:53", "target → barmen",
         "server: 400\\150  computer: 4100\\300+  memory: 60TB  Reveneu: 150 Million"],
        ["42503", "00:09:11", "target → barmen",
         "Nas: - Backup-80 TB"],
        ["42507", "00:22:33", "target → barmen",
         "Mueller Inc Foursquare Healthcare find out that with these offices there "
         "you have them on contact they have a panic there they are ready to pay"]
    ], ["행번호", "시각", "발신 → 수신", "body_en"])



with tab_eval:
    
    st.markdown("### 평가 방식")
    st.markdown("질문을 주고 해당 질문의 근거가 담긴 청크 번호를 찾아내는지 평가 ")

    
    a, b = st.columns([3, 2])
    with a:
        st.code("질문  : azot와 bentley 사이의 대화에서 바이러스와 백도어를\n"
                "        테스트하고 이를 '디지털 무기 작업'에 비유하며 정상\n"
                "        동작하는 소프트웨어를 파트너에게 제공하는 업무에\n"
                "        대한 증거를 찾아주세요.\n\n"
                "정답  : 청크 313, 314", language=None)
        st.code("[대화] azot | bentley | 2020-09-25 11:40~12:09 (1/2)\n"
                "11:40 bentley: I have a few questions for you\n"
                "11:50 bentley: I'm interested in the time zone where you live\n"
                "11:52 bentley: How much time you have ...", language=None)
    with b:
        frame([
            ["질의", "200개 (100문항 × 원문·번역본 2벌)"],
            ["후보 청크", "코퍼스당 15,522개"],
            ["검색 범위", "각 질의를 자기 코퍼스 안에서만"],
            ["정답 청크", "문항당 평균 1.93개"],
        ], ["항목", "값"])

    st.markdown("### 지표")
    frame([
        ["H@5", "정답 청크 중 하나라도 상위 5위 안에 들면 1점",
         "들면 1.00 / 못 들면 0"],
        ["H@10", "정답 청크 중 하나라도 상위 10위 안에 들면 1점",
         "들면 1.00 / 못 들면 0"],
        ["R@5", "상위 5위가 덮은 정답 청크 수 / min(정답 수, 5)",
         "정답 2개 중 1개 = 0.50"],
        ["R@10", "상위 10위가 덮은 정답 청크 수 / min(정답 수, 10)",
         "정답 2개 중 2개 = 1.00"],
        ["MRR@10", "첫 적중 순위의 역수", "1위 = 1.00, 4위 = 0.25"],
        ["nDCG@10", "1 / log2(첫 적중 순위 + 1)", "1위 = 1.00, 4위 = 0.43"],
    ], ["지표", "계산 방식", "예시 점수"])

    st.markdown("### 질문 유형")

    st.dataframe(pd.DataFrame(
        [[name, desc, direct, n] for name, desc, direct, n in TYPE_EX],
        columns=["질의 유형", "설명", "question", "개수"]
    ), hide_index=True, use_container_width=True, row_height=60)


with tab_model:

    st.markdown("### 모델")

    frame([
        ["BAAI/bge-m3", "1,024", "8,192", "의미 + 어휘",
         "다국어 임베딩. 어휘 매칭용 sparse 헤드가 함께 배포된다"],
        ["nlpai-lab/KURE-v1", "1,024", "8,192", "의미",
         "BGE-M3 를 한국어로 파인튜닝. 구조가 같아 형식이 호환된다"],
    ], ["모델", "벡터 차원", "최대 토큰", "만드는 벡터", "특징"])

    st.markdown("### 비교한 5가지 구성")
    frame([
        ["bb-dense", "bge", "bge", "의미 유사도만", "기준선"],
        ["bb-hybrid", "bge", "bge", "의미 + 어휘 매칭", "어휘 가중 0.3"],
        ["bk", "bge", "kure", "의미 유사도만", "질문만 bge"],
        ["kb", "kure", "bge", "의미 유사도만", "문서만 bge"],
        ["kk", "kure", "kure", "의미 유사도만", "한국어 특화 모델 단독"],
    ], ["구성", "질문 인코더", "문서 인코더", "점수 계산", "비고"])


with tab_prep:

    st.markdown("### 전처리")

    st.markdown("원본 **107,967행 → 106,566행**. 실제로 확인된 결함만 제거했다.")

    st.markdown("### ① HTML 엔티티 복원 — 10,521행")
    frame([
        ["I&#39;ll ask Morse", "I'll ask Morse"],
        ["I&#39;ll clarify by group", "I'll clarify by group"],
        ["Didn&#39;t specify how much", "Didn't specify how much"],
    ], ["처리 전", "처리 후"])

    st.markdown("### ② 시스템 메시지 줄 제거 — 6행")
    a, b = st.columns(2)
    with a:
        st.code("Didn't you come? 17:13:10]<steller> gitlab:\n"
                "https://179.43.147.243/steller/backdoor.js\n"
                "[17:22:15] *** Message was not sent. Either end\n"
                "the private conversation or restart it.\n"
                "[17:22:15]<steller> Check out http.js. ...", language=None)
    with b:
        st.code("Didn't you come? 17:13:10]<steller> gitlab:\n"
                "https://179.43.147.243/steller/backdoor.js\n"
                "[17:22:15]<steller> Check out http.js. ...", language=None)

    st.markdown("### ③ 양방향 중복 제거 — 270행")
    frame([
        ["51491", "2020-10-14 12:17:30.608", "azot", "professor",
         "Проф тут запары с админкой", "유지"],
        ["51492", "2020-10-14 12:17:30.624", "professor", "azot",
         "Проф тут запары с админкой", "제거"],
    ], ["행번호", "시각", "발신", "수신", "본문", "처리"])

    st.markdown("### ④ 브로드캐스트 병합 — 1,103행 → 공지 31건")
    a, b = st.columns(2)
    with a:
        frame([
            ["38205", "12:36:58", "defender", "song"],
            ["38206", "12:37:26", "defender", "kerasid"],
            ["38207", "12:37:27", "defender", "steller"],
            ["...", "...", "...", "... (197명)"],
        ], ["행번호", "시각", "발신", "수신"])
    with b:
        st.code("[공지] defender → 197명 | 2020-06-24 12:36\n"
                "수신: 0x00lord, 8383, airbnb1, alaska, alert,\n"
                "      ali, aloxa, andy, atlant, axel, ...\n\n"
                "백업 자바 계정을 안 보낸 사람 전원, 지금 보내라.\n"
                "이 계정이 막히면 연락이 끊긴다.", language=None)


with tab_chunking:
    st.markdown("### 청킹")

    frame([
        ["① 대화창 분리", "dyad (발화자-수신자 쌍)", "1,112개"],
        ["② 시간 gap 분리", "같은 대화창에서 1시간 이상 끊기면 분리", "12,478 세션"],
        ["512토큰 초과 세션 (약 2,300자)", "세션 토큰 중앙 46, p99 2,095, 최대 15,294",
         "886개 (7.10%)"],
        ["③ 512/128토큰 분할", "886개 세션만 발화 경계에서 분할",
         "대화 15,522 청크"],
    ], ["단계", "기준", "결과"])

    st.markdown("### 512 토큰 근거")
    frame([
        ["BGE-M3 개발자 권장",
         '"a chunk size of 512 is enough. And splitting the entire document '
         'into multiple chunks often can improve the retrieval performance."',
         "Hugging Face BAAI/bge-m3 Discussion #59, Shitao (BAAI), 2024-05-28",
         "huggingface.co/BAAI/bge-m3/discussions/59"],
        ["모델 카드",
         "최대 8,192 지원. "
         '"If you dont need such a long length, you can set a smaller value '
         'to speed up the encoding process."',
         "BAAI/bge-m3 모델 카드",
         "huggingface.co/BAAI/bge-m3"],
        ["청크 크기 연구",
         "사실 단답형 정답은 64~128 토큰, 넓은 맥락이 필요하면 512~1,024 토큰이 "
         "유리. 모델별로 최적 크기가 갈린다",
         "Rethinking Chunk Size For Long-Document Retrieval (2025)",
         "arxiv.org/abs/2505.21700"],
        ["데이터 분포",
         "세션 토큰 중앙 46. 12,478 세션 중 92.9%가 480 이하라 512에서 "
         "잘리는 세션은 7.10%뿐",
         "jabberchat2020process.csv 세션 집계 (bge-m3 토크나이저)",
         "data/chunking.py"],
        ["상한 무의미",
         "8,192를 넘는 세션은 2개(0.02%). 8192/1024로 잡으면 12,480 청크로 "
         "세션 수(12,478)와 같아져 사실상 분할이 없는 것과 동일",
         "동일 세션 집계",
         "data/chunking.py"]
    ], ["근거", "내용", "출처", "링크"])


with tab_metadata:
    st.markdown("### 메타 데이터")

    st.code("사용자 질문\n"
            "  -> LLM이 메타데이터, 키워드 추출\n"
            "  -> 메타데이터, 키워드로 검색 대상 축소\n"
            "  -> 벡터 검색\n"
            "  -> 청크 선정\n"
            "  -> 답변", language=None)

    st.markdown("### 메타데이터 리스트")
    frame([
        ["파일명"],
        ["발화자"],
        ["수신자"],
        ["대화창 (발화자-수신자 쌍)"],
        ["증거 소유자"],
        ["시작 날짜"],
        ["종료 날짜"],
    ], ["메타데이터"])

    st.markdown("### 키워드")
    st.markdown("해당 키워드가 존재하는 청크들 + 앞뒤 N개로 범위 좁히기")
