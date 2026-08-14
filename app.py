# -*- coding: utf-8 -*-
"""메신저 검색 성능 평가 - Streamlit 리포트 (단일 페이지)

실행:
    streamlit run app.py

표와 예시는 산출물에서 뽑아 하드코딩했다. 파일 없이도 그대로 열린다.
값을 갱신하려면 파이프라인을 다시 돌린 뒤 DETAIL 상수만 바꾸면 된다.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="메신저 검색 성능 평가", layout="wide")

SOURCE_URL = ("https://github.com/NorthwaveSecurity/"
              "complete_translation_leaked_chats_conti_ransomware/tree/main")

METRICS = ["R@5", "R@10", "MRR@10", "nDCG@10"]
CONFIGS = ["bb-dense", "bb-hybrid", "bk", "kb", "kk"]
QTYPES = ["1 의미기반", "2 식별자+의미기반", "3 화자기반", "4 날짜기반"]

# result.txt 의 구성별 표. {구성: {행 이름: [R@5, R@10, MRR@10, nDCG@10]}}
DETAIL = {
    "bb-dense": {
        "jabber_en": [0.480, 0.575, 0.389, 0.433],
        "jabber_ru": [0.515, 0.570, 0.416, 0.454],
        "1 의미기반": [0.300, 0.350, 0.227, 0.257],
        "2 식별자+의미기반": [0.750, 0.780, 0.666, 0.694],
        "3 화자기반": [0.590, 0.720, 0.442, 0.508],
        "4 날짜기반": [0.350, 0.440, 0.275, 0.314],
        "직접포함": [0.645, 0.700, 0.525, 0.568],
        "의역": [0.350, 0.445, 0.280, 0.319],
        "ALL": [0.497, 0.573, 0.403, 0.443],
    },
    "bb-hybrid": {
        "jabber_en": [0.520, 0.625, 0.435, 0.480],
        "jabber_ru": [0.560, 0.610, 0.445, 0.485],
        "1 의미기반": [0.300, 0.350, 0.238, 0.265],
        "2 식별자+의미기반": [0.860, 0.890, 0.717, 0.760],
        "3 화자기반": [0.600, 0.750, 0.465, 0.532],
        "4 날짜기반": [0.400, 0.480, 0.341, 0.374],
        "직접포함": [0.685, 0.755, 0.582, 0.623],
        "의역": [0.395, 0.480, 0.298, 0.342],
        "ALL": [0.540, 0.618, 0.440, 0.482],
    },
    "bk": {
        "jabber_en": [0.470, 0.540, 0.390, 0.426],
        "jabber_ru": [0.470, 0.580, 0.402, 0.443],
        "1 의미기반": [0.290, 0.380, 0.215, 0.254],
        "2 식별자+의미기반": [0.760, 0.810, 0.695, 0.723],
        "3 화자기반": [0.530, 0.680, 0.410, 0.474],
        "4 날짜기반": [0.300, 0.370, 0.263, 0.288],
        "직접포함": [0.630, 0.720, 0.537, 0.581],
        "의역": [0.310, 0.400, 0.254, 0.288],
        "ALL": [0.470, 0.560, 0.396, 0.435],
    },
    "kb": {
        "jabber_en": [0.480, 0.555, 0.400, 0.437],
        "jabber_ru": [0.545, 0.595, 0.458, 0.491],
        "1 의미기반": [0.280, 0.350, 0.238, 0.265],
        "2 식별자+의미기반": [0.740, 0.760, 0.655, 0.681],
        "3 화자기반": [0.640, 0.750, 0.523, 0.577],
        "4 날짜기반": [0.390, 0.440, 0.301, 0.335],
        "직접포함": [0.645, 0.715, 0.534, 0.578],
        "의역": [0.380, 0.435, 0.324, 0.351],
        "ALL": [0.512, 0.575, 0.429, 0.464],
    },
    "kk": {
        "jabber_en": [0.520, 0.585, 0.430, 0.467],
        "jabber_ru": [0.540, 0.615, 0.444, 0.484],
        "1 의미기반": [0.330, 0.390, 0.247, 0.281],
        "2 식별자+의미기반": [0.760, 0.810, 0.697, 0.724],
        "3 화자기반": [0.650, 0.790, 0.491, 0.561],
        "4 날짜기반": [0.380, 0.410, 0.312, 0.336],
        "직접포함": [0.680, 0.740, 0.561, 0.604],
        "의역": [0.380, 0.460, 0.313, 0.348],
        "ALL": [0.530, 0.600, 0.437, 0.476],
    },
}
NDCG = METRICS.index("nDCG@10")


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


# ==========================================================================
st.title("메신저 검색 성능 평가")
st.markdown("#### BAAI/bge-m3 (dense + sparse) · nlpai-lab/KURE-v1 (dense)")
st.markdown(
    " 범행 증거를 찾아내는"
    "성능을 비교한다. 질의 400개, 후보 청크 15,522개.")
st.divider()


# ==========================================================================
st.header("1. 결과")

st.markdown("### 1. 전체")
overall = pd.DataFrame({c: DETAIL[c]["ALL"] for c in CONFIGS}, index=METRICS)
overall.index.name = "지표"
overall["가장 높은 구성"] = overall[CONFIGS].idxmax(axis=1)
show(overall)

st.markdown("### 2. 한국어")
st.dataframe(pd.DataFrame([["한국어 코퍼스 없음"] + ["—"] * len(CONFIGS)],
                          columns=["코퍼스"] + CONFIGS),
             hide_index=True, use_container_width=True)

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

st.divider()


# ==========================================================================
st.header("2. 데이터셋")

st.markdown(
    f"**`jabberchat2020.csv`** — 실제 Conti 랜섬웨어 팀이 범행을 협의하며 약 1년간 "
    f"사용한 1:1 메신저 기록이다. 원문은 러시아어, 번역본은 영어다.\n\n"
    f"출처: {SOURCE_URL}")

c = st.columns(3)
c[0].metric("메시지", "107,967")
c[1].metric("계정", "295")
c[2].metric("대화 상대 쌍", "1,113")

st.markdown("### 실제 대화 — 피해 기업에 몸값을 요구하는 장면")
st.markdown(
    "2020-09-20 새벽, `target` 이 부하에게 침입한 기업의 자산 규모를 넘기며 몸값 "
    "우선순위를 논의하고, 14분 뒤 **피해 기업 두 곳이 지불할 준비가 되었는지 "
    "확인하라고 지시한다.**")
st.dataframe(pd.DataFrame([
    ["42498", "00:08:51", "target → barmen",
     "дай расшифровку", "복호화 대가 내역을 요구"],
    ["42499", "00:08:53", "target → barmen",
     "server: 400\\150  computer: 4100\\300+  memory: 60TB  Reveneu: 150 Million",
     "장악한 서버·PC 수와 피해 기업 연매출 1억 5천만 달러"],
    ["42503", "00:09:11", "target → barmen",
     "Nas: - Backup-80 TB", "백업 저장장치 80TB — 복구를 막을 대상"],
    ["42507", "00:22:33", "target → barmen",
     "Mueller Inc, Foursquare Healthcare — 그쪽은 네가 연락 담당이다. "
     "패닉 상태이고 지불할 준비가 되어 있다",
     "피해 기업 실명을 대며 지불 의사를 확인하라고 지시"],
    ["42508", "00:22:38", "target → troy",
     "(같은 지시를 다른 조직원에게 반복 발송)",
     "1:1 구조라 같은 지시가 여러 행으로 남는다"],
], columns=["행번호", "시각", "발신 → 수신", "대화 내용", "무엇을 뜻하는가"]),
    hide_index=True, use_container_width=True)

st.markdown("### 데이터 구조")
st.dataframe(pd.DataFrame([
    ["ts", "타임스탬프 (마이크로초). 전 행이 유일해 정렬 기준으로 쓴다"],
    ["from / to", "발신·수신 계정. 1:1 대화라 항상 개인 계정이다"],
    ["body", "원문 (러시아어 중심)"],
    ["body_en", "영어 번역"],
    ["body_language", "감지된 언어. 짧은 문장이 자주 오탐되어 쓰지 않는다"],
], columns=["컬럼", "설명"]), hide_index=True, use_container_width=True)

st.info("메시지 1건은 중앙값 20자로 매우 짧다. 그대로 임베딩하면 `Ok` 수천 개가 같은 "
        "벡터가 되므로, **대화 상대 쌍 + 1시간 공백** 으로 세션을 묶어 "
        "**15,522개 청크**로 만들었다. 분할 기준은 영문이라 모든 언어가 같은 경계로 "
        "잘리고 청크 번호가 일치한다.")

st.divider()


# ==========================================================================
st.header("3. 평가 방식")

st.markdown("한국어 질문을 주고 **그 근거가 담긴 청크를 찾아내는지** 본다. "
            "정답은 청크 번호다.")

a, b = st.columns([3, 2])
with a:
    st.markdown("**질문 → 정답 청크**")
    st.code("질문  : azot와 bentley 사이의 대화에서 바이러스와 백도어를\n"
            "        테스트하고 이를 '디지털 무기 작업'에 비유하며 정상\n"
            "        동작하는 소프트웨어를 파트너에게 제공하는 업무에\n"
            "        대한 증거를 찾아주세요.\n\n"
            "정답  : 청크 313, 314", language=None)
    st.markdown("**청크 313 본문**")
    st.code("[대화] azot | bentley | 2020-09-25 11:40~12:09 (1/2)\n"
            "11:40 bentley: I have a few questions for you\n"
            "11:50 bentley: I'm interested in the time zone where you live\n"
            "11:52 bentley: How much time you have ...", language=None)
with b:
    st.markdown("**평가 규모**")
    st.dataframe(pd.DataFrame([
        ["질의", "400개 (200문항 × 원문·번역본 2벌)"],
        ["후보 청크", "코퍼스당 15,522개"],
        ["검색 범위", "각 질의를 자기 코퍼스 안에서만"],
        ["정답 청크", "문항당 평균 1.96개"],
    ], columns=["항목", "값"]), hide_index=True, use_container_width=True)
    st.markdown("정답이 여러 개인 것은 하나의 정황이 청크 겹침(128토큰) 때문에 두 "
                "청크에 걸친 결과이지, 별개 정답이 아니다.")

st.markdown("### 지표")
st.dataframe(pd.DataFrame([
    ["R@5", "정답 청크가 상위 5위 안에 있으면 1점", "들면 1.00 / 못 들면 0"],
    ["R@10", "정답 청크가 상위 10위 안에 있으면 1점", "들면 1.00 / 못 들면 0"],
    ["MRR@10", "첫 적중 순위의 역수", "1위 = 1.00, 4위 = 0.25"],
    ["nDCG@10", "순위를 로그로 할인. 상위권을 더 크게 평가", "1위 = 1.00, 4위 = 0.43"],
], columns=["지표", "계산 방식", "예시 점수"]),
    hide_index=True, use_container_width=True)

st.divider()


# ==========================================================================
st.header("4. 사용한 모델")

st.dataframe(pd.DataFrame([
    ["BAAI/bge-m3", "1,024", "8,192", "의미 + 어휘",
     "다국어 임베딩. 어휘 매칭용 sparse 헤드가 함께 배포된다"],
    ["nlpai-lab/KURE-v1", "1,024", "8,192", "의미",
     "BGE-M3 를 한국어로 파인튜닝. 구조가 같아 형식이 호환된다"],
], columns=["모델", "벡터 차원", "최대 토큰", "만드는 벡터", "특징"]),
    hide_index=True, use_container_width=True)

st.markdown("### 비교한 5가지 구성")
st.markdown("질문과 문서를 서로 다른 모델로 인코딩해 교차 비교한다. 이름의 앞 글자가 "
            "질문, 뒤 글자가 문서 인코더다 (b = bge, k = kure).")
st.dataframe(pd.DataFrame([
    ["bb-dense", "bge", "bge", "의미 유사도만", "기준선"],
    ["bb-hybrid", "bge", "bge", "의미 + 어휘 매칭", "어휘 가중 0.3"],
    ["bk", "bge", "kure", "의미 유사도만", "질문만 bge"],
    ["kb", "kure", "bge", "의미 유사도만", "문서만 bge"],
    ["kk", "kure", "kure", "의미 유사도만", "한국어 특화 모델 단독"],
], columns=["구성", "질문 인코더", "문서 인코더", "점수 계산", "비고"]),
    hide_index=True, use_container_width=True)
st.markdown("어휘 매칭(sparse)은 BGE-M3 에만 있어 **bb-hybrid** 에서만 쓴다.")

st.divider()


# ==========================================================================
st.header("부록 1. 전처리")
st.markdown("원본 **107,967행 → 106,566행**. 실제로 확인된 결함만 제거했다.")

st.markdown("### ① HTML 엔티티 복원 — 10,521행")
st.markdown("번역 컬럼에 엔티티가 문자 그대로 남아 있다.")
st.dataframe(pd.DataFrame([
    ["I&#39;ll ask Morse", "I'll ask Morse"],
    ["I&#39;ll clarify by group", "I'll clarify by group"],
    ["Didn&#39;t specify how much", "Didn't specify how much"],
], columns=["처리 전", "처리 후"]), hide_index=True, use_container_width=True)

st.markdown("### ② 시스템 메시지 줄 제거 — 6행")
st.markdown(
    "메신저가 남긴 전송 실패 안내가 실제 대화 중간에 끼어 있다. **행째로 지우면 "
    "같은 행의 증거(악성코드 배포 URL, IP)까지 사라진다.** 줄 단위로 지우고, 그 "
    "결과 본문이 비면 그때 행을 제거한다.")
a, b = st.columns(2)
with a:
    st.markdown("**처리 전** (행 27667)")
    st.code("Didn't you come? 17:13:10]<steller> gitlab:\n"
            "https://179.43.147.243/steller/backdoor.js\n"
            "[17:22:15] *** Message was not sent. Either end\n"
            "the private conversation or restart it.\n"
            "[17:22:15]<steller> Check out http.js. ...", language=None)
with b:
    st.markdown("**처리 후** — 안내 줄만 삭제")
    st.code("Didn't you come? 17:13:10]<steller> gitlab:\n"
            "https://179.43.147.243/steller/backdoor.js\n"
            "[17:22:15]<steller> Check out http.js. ...", language=None)

st.markdown("### ③ 양방향 중복 제거 — 270행")
st.markdown("발신·수신만 뒤바뀐 같은 메시지가 **0.016초 간격**으로 두 번 기록돼 있다. "
            "로그 재구성 아티팩트이므로 먼저 온 행만 남긴다.")
st.dataframe(pd.DataFrame([
    ["51491", "2020-10-14 12:17:30.608", "azot", "professor",
     "Проф тут запары с админкой", "유지"],
    ["51492", "2020-10-14 12:17:30.624", "professor", "azot",
     "Проф тут запары с админкой", "제거"],
], columns=["행번호", "시각", "발신", "수신", "본문", "처리"]),
    hide_index=True, use_container_width=True)

st.markdown("### ④ 브로드캐스트 병합 — 1,103행 → 공지 31건")
st.markdown("1:1 구조라 조직 공지도 개별 메시지로 뿌려진다. `defender` 가 백업 "
            "계정을 내놓으라는 공지 하나를 **197명에게 211행으로** 보냈다.")
c = st.columns(3)
c[0].metric("공지 건수", "31건")
c[1].metric("병합된 행", "1,103행")
c[2].metric("최대 수신자", "197명")
a, b = st.columns(2)
with a:
    st.markdown("**처리 전** — 같은 내용이 197행")
    st.dataframe(pd.DataFrame([
        ["38205", "12:36:58", "defender", "song"],
        ["38206", "12:37:26", "defender", "kerasid"],
        ["38207", "12:37:27", "defender", "steller"],
        ["...", "...", "...", "... (197명)"],
    ], columns=["행번호", "시각", "발신", "수신"]),
        hide_index=True, use_container_width=True)
with b:
    st.markdown("**처리 후** — 1행 + 수신자 목록")
    st.code("[공지] defender → 197명 | 2020-06-24 12:36\n"
            "수신: 0x00lord, 8383, airbnb1, alaska, alert,\n"
            "      ali, aloxa, andy, atlant, axel, ...\n\n"
            "백업 자바 계정을 안 보낸 사람 전원, 지금 보내라.\n"
            "이 계정이 막히면 연락이 끊긴다.", language=None)
st.divider()


# ==========================================================================
st.header("부록 2. 질문 유형")
st.markdown("유형 4종 × 표현 2종 × 25문항 = **200문항**. 같은 정답 청크에 대해 "
            "직접포함·의역이 1:1 로 짝지어져 있어 표현을 바꾼 효과만 따로 볼 수 있다.")

TYPE_EX = [
    ("1 의미기반", "고유명사를 모두 빼고 범죄 정황만으로 검색",
     "바이러스와 백도어를 테스트하고 이를 '디지털 무기 작업'에 비유하며 정상 "
     "동작하는 소프트웨어를 파트너에게 제공하는 업무에 대한 증거를 찾아주세요.",
     "악성 프로그램의 작동 여부를 시험하는 일을 무기와 유사한 업무라고 설명하고, "
     "검증을 마친 프로그램을 협력자에게 넘기는 업무 정황을 찾아주세요."),
    ("2 식별자+의미기반", "IP·비트코인 주소·파일명·회사명이 질문에 들어간다",
     "IP 68.224.217.72와 CALAHANLAW가 관련된 기록에서 피해 조직의 문서를 탈취한 "
     "정황에 대한 증거를 찾아주세요.",
     "IP 68.224.217.72와 CALAHANLAW에 관한 대화에서 외부로 빼낸 피해 업체 자료가 "
     "언급된 증거를 찾아주세요."),
    ("3 화자기반", "대화 참여자 두 명을 지정한다",
     "azot와 bentley 사이의 대화에서 바이러스와 백도어를 테스트하고 이를 '디지털 "
     "무기 작업'에 비유하며 정상 동작하는 소프트웨어를 파트너에게 제공하는 업무에 "
     "대한 증거를 찾아주세요.",
     "azot와 bentley 사이의 대화에서 악성 프로그램의 작동 여부를 시험하는 일을 "
     "무기와 비슷한 업무라고 설명하고 검증된 프로그램을 협력자에게 넘기는 내용을 "
     "찾아주세요."),
    ("4 날짜기반", "대화가 오간 날짜를 지정한다",
     "2020-09-25에 바이러스와 백도어를 테스트하고 이를 '디지털 무기 작업'에 "
     "비유하며 정상 동작하는 소프트웨어를 파트너에게 제공하는 업무에 대해 논의한 "
     "증거를 찾아주세요.",
     "2020-09-25에 악성 프로그램의 작동 여부를 시험하는 일을 무기와 비슷한 "
     "업무라고 설명하고 검증된 프로그램을 협력자에게 넘기는 내용을 찾아주세요."),
]

st.markdown("### 유형별 성능 요약 (bb-hybrid 기준)")
st.dataframe(pd.DataFrame(
    [[name, desc, DETAIL["bb-hybrid"][name][NDCG]]
     for name, desc, _, _ in TYPE_EX],
    columns=["질의 유형", "설명", "nDCG@10"]
).style.format({"nDCG@10": "{:.3f}"}),
    hide_index=True, use_container_width=True)

for name, desc, direct, para in TYPE_EX:
    st.markdown(f"### {name} — {desc}"  )
    a, b = st.columns(2)
    with a:
        st.markdown("직역")
        st.info(direct)
    with b:
        st.markdown("의역")
        st.warning(para)

