# -*- coding: utf-8 -*-
"""합성증거 20사건 문서용 QA 생성기 (문서당 100문항, 질문은 한국어).

문서 7종(ch/en/ko/pil/rs/uz/vn)은 20사건 x 증거목록 20블록(A-001~T-020)으로
구조가 완전히 평행하다. 한국어판에서 사건별 인물·법인·장소를 뽑아 한국어 질의를
만들고, 각 언어판에서는 같은 증거블록이 차지하는 청크를 정답으로 삼는다.

질의 문형
    "<그 블록에만 나오는 대상> <증거 유형>에 해당하는 증거를 찾아주세요."
    예: "김도윤과 박서진 명의 가상계좌의 입출금 거래 내역에 해당하는 증거를 찾아주세요."

  사전 실험 결과 짧고 초점이 좁은 질의가 가장 잘 검색된다(중앙 순위 5위).
  사건명·범죄사실·금액·장소를 덧붙이면 같은 사건의 다른 블록과도 매칭되어
  오히려 순위가 떨어졌다(중앙 16~116위). 그래서 대상 + 증거유형만 남긴다.

정답 청크
    블록의 문자 구간을 토크나이저로 토큰 구간으로 바꾼 뒤, npz 의 token_start/
    token_end 와 겹치는 청크를 모두 정답으로 삼는다. overlap=128 이라 한 블록이
    여러 청크에 걸치는 것이 정상이다.

사용:
    python src/build_qa_synthetic.py                 # 전체 7문서
    python src/build_qa_synthetic.py --doc ko
    python src/build_qa_synthetic.py --per-case 5 --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import string
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT_DIR = os.path.join(ROOT, "data", "txt")
EMB_DIR = os.path.join(ROOT, "data", "emb")
QA_DIR = os.path.join(ROOT, "data", "qa")

LETTERS = string.ascii_uppercase[:20]
KO_TXT = os.path.join(TXT_DIR, "ko_합성증거_20사건.txt")
MAX_GOLD = 8                      # 정답이 이보다 많은 청크에 걸치면 변별력이 없다


# --------------------------------------------------------------------------
# 언어판별 사건 프로파일 추출
# --------------------------------------------------------------------------
# 질의 안의 인물·법인·장소는 그 문서의 언어 표기를 그대로 쓴다.
# (예: 중국어판 질의에는 金道允, 러시아어판에는 Ким До Юн)
# 헤더에 없는 필드는 None 이고, 아래 fallback 순서로 대체한다.
LABELS = {
    "ko":  {"people": "주요 관계인", "orgs": "관련 법인·기관", "places": "주요 장소",
            "keywords": None},
    "ch":  {"people": "主要相关人员", "orgs": None, "places": None,
            "keywords": "主要检索词"},
    "en":  {"people": "Principal Persons", "orgs": "Relevant Organizations",
            "places": "Principal Locations", "keywords": None},
    "vn":  {"people": "Những người chính", "orgs": "Tổ chức liên quan",
            "places": "Địa điểm chính", "keywords": None},
    "uz":  {"people": "Asosiy shaxslar", "orgs": None, "places": None,
            "keywords": "Asosiy qidiruv terminlari"},
    "rs":  {"people": "Основные лица", "orgs": None, "places": None,
            "keywords": "Основные поисковые термины"},
    "pil": {"people": "Mga pangunahing taong sangkot", "orgs": None, "places": None,
            "keywords": "Mga pangunahing search term"},
}
SPLIT_RE = r"\s*[,、·]\s*"


def _field(head: str, label: str | None) -> list[str]:
    if not label:
        return []
    m = re.search(re.escape(label) + r"\s*[:：]\s*([^\n]+)", head)
    if not m:
        return []
    return [x.strip() for x in re.split(SPLIT_RE, m.group(1)) if x.strip()]


def profiles_for(tag: str, text: str, spans: dict[str, tuple[int, int]]) -> dict[int, dict]:
    """사건별 프로파일. 헤더는 그 사건 첫 증거블록(X-001) 바로 앞에 있다."""
    labels = LABELS.get(tag, LABELS["en"])
    out = {}
    for i, letter in enumerate(LETTERS, start=1):
        bid = f"{letter}-001"
        if bid not in spans:
            continue
        start = spans[bid][0]
        head = text[max(0, start - 1600):start]
        people = _field(head, labels["people"])
        orgs = _field(head, labels["orgs"])
        places = _field(head, labels["places"])
        keywords = _field(head, labels["keywords"])
        # 헤더에 법인/장소가 없는 언어판(ch·uz·rs·pil)은 검색어에서 대체어를 얻는다.
        if not orgs:
            orgs = keywords[:3] or people
        if not places:
            places = keywords[:4] or people
        out[i] = {"people": people, "orgs": orgs, "places": places}
    return out


# --------------------------------------------------------------------------
# 문서 파싱: 사건 -> 증거블록 -> 문자 구간
# --------------------------------------------------------------------------
def block_spans(text: str) -> dict[str, tuple[int, int]]:
    """증거ID -> (시작 문자 offset, 끝 문자 offset). 첫 등장을 블록 시작으로 본다."""
    first: dict[str, int] = {}
    for m in re.finditer(r"\b([A-T])-(\d{3})\b", text):
        first.setdefault(m.group(0), m.start())
    ordered = sorted(first.items(), key=lambda kv: kv[1])
    out = {}
    for i, (bid, st) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        out[bid] = (st, end)
    return out


def char_to_token_offsets(text: str, cuts: list[int], tokenizer) -> list[int]:
    """문자 offset 목록을 토큰 offset 으로 바꾼다.

    구간별로 나눠 토크나이즈한 뒤 개수를 누적한다. 경계에서 1~2 토큰 오차가
    생길 수 있으나 청크가 512 토큰이고 128 겹치므로 정답 청크 판정에 영향이 없다.
    """
    cuts = sorted(set([0] + cuts + [len(text)]))
    counts, acc = {}, 0
    for i in range(len(cuts) - 1):
        seg = text[cuts[i]:cuts[i + 1]]
        n = len(tokenizer.encode(seg, add_special_tokens=False)) if seg else 0
        counts[cuts[i]] = acc
        acc += n
    counts[cuts[-1]] = acc
    return counts


def gold_chunks(tok_start: int, tok_end: int,
                ts: np.ndarray, te: np.ndarray) -> list[int]:
    """[tok_start, tok_end) 와 겹치는 청크 인덱스."""
    return [i for i in range(len(ts)) if ts[i] < tok_end and te[i] > tok_start]


# --------------------------------------------------------------------------
# 질의 템플릿: 대상 + 증거유형 (짧게)
# --------------------------------------------------------------------------
SUFFIX = "에 해당하는 증거를 찾아주세요."


def templates(p: dict) -> list[tuple[int, str]]:
    """(증거블록 번호, 질의 본문) 목록.

    고유명사는 그 문서 언어의 표기를 그대로 쓰므로, 외국어 이름 뒤에 한국어 조사가
    붙어 어색해지지 않도록 조사를 최소화한 명사구로 쓴다.
    """
    pe, org, pl = p["people"], p["orgs"], p["places"]

    def person(i):
        return pe[i % len(pe)] if pe else "관계인"

    def org_(i):
        return org[i % len(org)] if org else "관련 법인"

    def place(i):
        return pl[i % len(pl)] if pl else "현장"

    return [
        (1,  f"{person(0)}·{person(1)} 참여 업무방 메신저 대화 원문"),
        (2,  f"{person(0)} → {person(1)} 삭제 메신저 복구 레코드"),
        (3,  f"{person(0)} → {person(1)} 이메일 송수신 및 첨부파일 기록"),
        (4,  f"{person(0)}·{person(1)} 명의 전화번호 통화 수발신 상세내역"),
        (5,  f"{place(0)} 부근 기지국·GPS·Wi-Fi 위치기록"),
        (6,  f"{place(1)} 출입 차량 GPS·번호판 인식 및 주차기록"),
        (7,  f"{place(0)} CCTV 영상 메타데이터 및 관찰기록"),
        (8,  f"{person(0)}·{person(2)} 명의 가상계좌 입출금 거래 내역"),
        (9,  f"{org_(0)} 법인카드·간편결제·영수증 사용 기록"),
        (10, f"{person(0)} 웹브라우저 검색어 및 다운로드 기록"),
        (11, f"{org_(0)} 업무서버 계정별 접속 감사로그"),
        (12, f"{person(0)} 클라우드 업로드·삭제파일 복구 기록"),
        (13, f"{org_(0)} 문서 버전관리·ERP·전자결재 기록"),
        (14, f"{person(1)} SNS 게시물 및 비공개 메시지"),
        (15, f"{place(0)} 압수수색 현장 압수물 목록 및 감정자료"),
        (16, f"{person(0)} 일정·캘린더 및 출입통제 기록"),
        (17, f"{person(0)}·{person(1)} 관련 교차증거 타임라인"),
        (19, f"{person(0)} 기기 포렌식 이미지 해시 등 보충 수사보고용 원시자료"),
        (20, f"{person(0)}·{person(1)}·{person(2)} 사건 증거 연결 인덱스 및 핵심 검색어"),
    ]


# --------------------------------------------------------------------------
def build_for(tag: str, txt_path: str, npz_path: str, per_case: int,
              tokenizer) -> tuple[dict, dict]:
    text = open(txt_path, encoding="utf-8", errors="replace").read()
    npz = np.load(npz_path, allow_pickle=True)
    ts, te, tc = npz["token_start"], npz["token_end"], npz["token_count"]
    info = json.loads(str(npz["info"]))

    spans = block_spans(text)
    profiles = profiles_for(tag, text, spans)
    cuts = sorted({c for se in spans.values() for c in se})
    tokmap = char_to_token_offsets(text, cuts, tokenizer)

    pairs, stats = [], {"missing_block": 0, "too_many": 0, "short": []}
    seq = 0

    for letter in LETTERS:
        no = LETTERS.index(letter) + 1
        prof = profiles.get(no)
        if not prof:
            continue
        cands = templates(prof)
        off = (no - 1) % len(cands)
        cands = cands[off:] + cands[:off]

        made = 0
        for bnum, body in cands:
            if made >= per_case:
                break
            bid = f"{letter}-{bnum:03d}"
            if bid not in spans:
                stats["missing_block"] += 1
                continue
            cs, ce = spans[bid]
            gold = gold_chunks(tokmap[cs], tokmap[ce], ts, te)
            if not gold:
                stats["missing_block"] += 1
                continue
            if len(gold) > MAX_GOLD:
                stats["too_many"] += 1
                continue
            seq += 1
            made += 1
            pairs.append({
                "id": f"{tag}-{seq:03d}",
                "case_no": no,
                "evidence_id": bid,
                "question": body + SUFFIX,
                "answer_chunk_indices": gold,
                "answer_chunk_spans": [
                    {"chunk": g, "token_start": int(ts[g]), "token_end": int(te[g]),
                     "token_count": int(tc[g])} for g in gold
                ],
            })
        if made < per_case:
            stats["short"].append((no, made))

    doc = {
        "source": os.path.basename(txt_path),
        "embedding_file": os.path.basename(npz_path),
        "chunking": {"model": info["model"], "dim": info["dim"],
                     "chunk_size": info["chunk_size"], "overlap": info["overlap"],
                     "n_chunks": info["n_chunks"]},
        "question_language": "ko",
        "n_qa": len(pairs),
        "note": ("질의는 '<대상> <증거유형>에 해당하는 증거를 찾아주세요' 형태의 한국어 문장이다. "
                 "정답은 해당 증거블록이 걸쳐 있는 청크 전부(answer_chunk_indices, 0-based)이며 "
                 "overlap=128 때문에 여러 개인 것이 정상이다. "
                 "질문은 한국어이고 본문은 문서 언어이므로 교차언어 검색 평가가 된다."),
        "qa_pairs": pairs,
    }
    return doc, stats


def main():
    ap = argparse.ArgumentParser(description="합성증거 문서용 한국어 QA 생성")
    ap.add_argument("--per-case", type=int, default=5)
    ap.add_argument("--doc", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    targets = []
    for txt in sorted(glob.glob(os.path.join(TXT_DIR, "*합성증거*.txt"))):
        tag = os.path.basename(txt).split("_합성증거")[0]
        if args.doc and tag != args.doc:
            continue
        npz = os.path.join(EMB_DIR, f"{tag}_합성증거_20사건_embeddings.npz")
        if os.path.exists(npz):
            targets.append((tag, txt, npz))
        else:
            print(f"[!] 임베딩 없음: {os.path.basename(npz)}", file=sys.stderr)

    if not targets:
        raise SystemExit("대상 문서가 없습니다.")

    os.makedirs(QA_DIR, exist_ok=True)
    print(f"{'doc':5} {'문항':>5} {'청크':>6} {'gold=1':>7} {'gold>=2':>8} {'평균gold':>8} "
          f"{'제외(블록)':>10} {'제외(과다)':>10}")
    print("-" * 74)
    for tag, txt, npz in targets:
        doc, st = build_for(tag, txt, npz, args.per_case, tokenizer)
        n = len(doc["qa_pairs"])
        g = [len(p["answer_chunk_indices"]) for p in doc["qa_pairs"]]
        one = sum(1 for x in g if x == 1)
        print(f"{tag:5} {n:>5} {doc['chunking']['n_chunks']:>6} {one:>7} {n - one:>8} "
              f"{np.mean(g) if g else 0:>8.2f} {st['missing_block']:>10} {st['too_many']:>10}")
        if st["short"]:
            print(f"      [!] 문항 부족 사건: {st['short']}")
        if not args.dry_run:
            with open(os.path.join(QA_DIR, f"{tag}_합성증거_20사건_qa.json"),
                      "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

    print("\nDRY-RUN: 저장하지 않았습니다." if args.dry_run else f"\n저장 위치: {QA_DIR}")


if __name__ == "__main__":
    main()
