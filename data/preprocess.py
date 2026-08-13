# -*- coding: utf-8 -*-
"""jabberchat2020.csv -> 검색용 청크 (원문/영문 2벌).

Conti 조직의 Jabber 1:1 대화 로그를 RAG 검색 평가용 청크로 가공한다.
문서가 아니라 대화 로그라 청킹 단위가 다르다. 메시지 1건은 중앙값 20자라
그대로 임베딩하면 'Ok' 3,118개가 동일 벡터가 되므로, 대화 세션으로 묶는다.

    ① 정제      원본의 실제 결함만 제거 (아래 7종)
    ② 세션 분리  dyad(대화 상대 쌍) + 시간 gap
    ③ 청킹      세션 내부에서만 512/128 토큰, 메시지 경계 스냅

원본에서 실제로 확인된 결함 (107,967행 기준):

    시간순 아님        149개 날짜 블록이 섞인 순서, 역행 174회
    HTML 엔티티        body_en 10,521행 (body 는 1행)
    앞뒤 공백          body 1,657행 / body_en 598행
    시스템 메시지      'Your message was not sent...' 6행에 혼입
    conference 도메인  22행 (그룹채팅방, dyad 개념 불성립)
    화자 중복 오류     273행 (a->b 와 b->a 가 같은 본문·같은 시각)
    발화 중복 나열     1,103행 (공지 1건을 최대 197명에게 개별 발송)

원문/영문 청크 경계는 기본적으로 일치시킨다(--split-basis max). 메시지별
토큰 수를 두 언어의 큰 쪽으로 잡아 분할 지점을 한 번만 계산하므로, 두 벌의
청크 인덱스가 같아져 QA 정답(청크 위치)을 1벌만 만들면 된다. 따로 계산하면
청크 수가 14,717 vs 14,693 으로 어긋나 정답을 2벌 만들어야 한다.

사용:
    python data/preprocess.py
    python data/preprocess.py --gap-hours 2 --chunk-size 1024 --overlap 256
    python data/preprocess.py --split-basis each     # 원문/영문 따로 분할
    python data/preprocess.py --stats-only           # 저장 없이 통계만
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "jabberchat2020.csv")
DEFAULT_OUT = os.path.join(ROOT, "data", "chunks")

TOKENIZER = "BAAI/bge-m3"

# 조직 내부 도메인. '@' 가 빠진 주소(10행)에서 접미사로 떼어내야 한다.
#   twinq3mcco35auwcstmt.onion  -> twin
#   vjud.q3mcco35auwcstmt.onion -> vjud
DOMAIN_SUFFIX = re.compile(r"\.?q?3?mcco35auwcstmt\.(?:onion|oinon)$", re.I)

# OTR 시스템 메시지. 순수 시스템 행(2행)도 있지만 실제 대화에 섞인 행(4행)이
# 더 많아서 행째로 지우면 backdoor.js 배포 URL 같은 증거까지 사라진다. 줄만 뺀다.
SYSTEM_LINE = re.compile(
    r"(?im)^[ \t]*(?:\[\d{2}:\d{2}:\d{2}\][ \t]*)?\*\*\*.*?"
    r"(?:message was not sent|сообщение не было отправлено).*?$\n?")

MIRROR_BUCKET = "10s"     # 화자 중복 오류 판정 시각 해상도
BCAST_GAP = 600           # 같은 공지의 연속 발송 간격 상한(초)
BCAST_MIN_RECIPIENTS = 5  # 이 인원 이상이면 브로드캐스트로 본다


# --------------------------------------------------------------------------
# ① 정제
# --------------------------------------------------------------------------
def norm_text(s) -> str:
    """HTML 엔티티 디코딩 + 공백 정규화. 줄바꿈은 공백 1개로 접는다.

    Defender 이벤트 로그처럼 여러 줄짜리 증거가 7,842행 있으나, 청크 텍스트는
    한 줄 = 한 발화 형식이라 내부 줄바꿈을 남기면 화자 귀속이 흐트러진다.
    """
    return re.sub(r"\s+", " ", html.unescape(str(s))).strip()


def strip_domain(addr: str) -> str:
    """주소에서 닉네임만 남긴다.

    대부분은 '@' 앞부분이면 되지만, '@' 가 아예 없는 행이 10개 있다.
    그냥 split('@')[0] 하면 'twinq3mcco35auwcstmt.onion' 이 통째로 닉네임이
    되어 가짜 계정이 생긴다. 도메인 오타(oinon, 후행 공백, '>')는 '@' 뒤쪽에
    있으므로 자동으로 버려진다.
    """
    a = str(addr).strip().rstrip(">").strip()
    if "@" in a:
        return a.split("@", 1)[0].strip()
    return DOMAIN_SUFFIX.sub("", a).strip() or a


def clean(csv_path: str, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """원본 CSV -> 정제된 대화 DataFrame + 브로드캐스트 목록."""
    df = pd.read_csv(csv_path, index_col=0)
    df = df.rename_axis("src_row").reset_index()
    stat = {"원본": len(df)}

    # 1) 시간축 복원. 파일은 149개 날짜 블록이 뒤섞인 순서라 이걸 먼저 하지
    #    않으면 이후 모든 시간 윈도우 연산(중복 탐지·세션 gap)이 무너진다.
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)

    # 2) 엔티티 디코딩 + 공백 정리 + 시스템 메시지 줄 제거
    for src, dst in (("body", "ru"), ("body_en", "en")):
        txt = df[src].astype(str).map(html.unescape)
        txt = txt.str.replace(SYSTEM_LINE, "", regex=True)
        df[dst] = txt.map(norm_text)

    # 3) 닉네임 추출
    df["nick_f"] = df["from"].map(strip_domain)
    df["nick_t"] = df["to"].map(strip_domain)

    # 4) conference(그룹채팅방) 제외 - 1:1 이 아니라 dyad 가 성립하지 않는다
    is_conf = (df["from"].astype(str).str.contains("conference", case=False, na=False)
               | df["to"].astype(str).str.contains("conference", case=False, na=False))
    df = df[~is_conf]
    stat["conference 제외"] = int(is_conf.sum())

    # 5) 빈 본문 제거 (원래 빈 1행 + 시스템 메시지만 있던 2행)
    empty = (df.ru.str.len() == 0) | (df.en.str.len() == 0)
    df = df[~empty]
    stat["빈 본문 제거"] = int(empty.sum())

    df["dyad"] = np.where(df.nick_f < df.nick_t,
                          df.nick_f + " | " + df.nick_t,
                          df.nick_t + " | " + df.nick_f)

    # 6) 화자 중복 오류: a->b 와 b->a 가 같은 본문·같은 시각으로 둘 다 기록된
    #    로그 재구성 아티팩트. 먼저 온 행만 남긴다.
    key = (df.dyad + "\x00" + df.ru + "\x00"
           + df.ts.dt.floor(MIRROR_BUCKET).astype(str))
    mirror = key.duplicated(keep="first")
    df = df[~mirror]
    stat["화자 중복 오류 제거"] = int(mirror.sum())

    # 7) 발화 중복 나열: 한 사람이 같은 공지를 수십~수백 명에게 개별 발송한 것.
    #    (발신자, 본문) 이 BCAST_GAP 안에서 연속하며 수신자가 임계 이상인 묶음.
    #    지우지 않고 대표 1행으로 접되 수신자 목록을 남긴다 - 197명 명부 자체가
    #    그 시점 조직 규모의 증거다.
    d = df.sort_values(["nick_f", "ru", "ts"], kind="mergesort").copy()
    gap = d.groupby(d.nick_f + "\x00" + d.ru)["ts"].diff().dt.total_seconds()
    d["_grp"] = (gap.isna() | (gap > BCAST_GAP)).cumsum()
    n_recip = d.groupby("_grp")["nick_t"].transform("nunique")
    d["_bcast"] = n_recip >= BCAST_MIN_RECIPIENTS
    drop = d._bcast & d.duplicated("_grp", keep="first")
    stat["브로드캐스트 병합"] = int(drop.sum())

    kept = d[~drop].sort_values("ts", kind="mergesort")
    bcasts = _collect_broadcasts(d)
    conv = kept[~kept._bcast].drop(columns=["_grp", "_bcast"])

    stat["정제 후 대화"] = len(conv)
    stat["공지"] = len(bcasts)
    if verbose:
        _print_stat(stat)
    return conv, bcasts


def _collect_broadcasts(d: pd.DataFrame) -> list[dict]:
    """브로드캐스트 묶음 -> 수신자 목록을 보존한 공지 레코드."""
    out = []
    for grp, g in d[d._bcast].groupby("_grp"):
        recips = sorted(set(g.nick_t))
        out.append({
            "sender": g.nick_f.iloc[0],
            "recipients": recips,
            "n_recipients": len(recips),
            "ts_start": g.ts.min(), "ts_end": g.ts.max(),
            "ru": g.ru.iloc[0], "en": g.en.iloc[0],
            "src_rows": [int(x) for x in g.src_row],
        })
    return sorted(out, key=lambda x: x["ts_start"])


def _print_stat(stat: dict) -> None:
    print("[정제]", file=sys.stderr)
    for k, v in stat.items():
        print(f"  {k:<18}{v:>9,}", file=sys.stderr)


# --------------------------------------------------------------------------
# ② 세션 분리
# --------------------------------------------------------------------------
def split_sessions(conv: pd.DataFrame, gap_hours: float) -> pd.DataFrame:
    """dyad + 시간 gap 으로 세션 번호를 매긴다.

    dyad 로 먼저 정렬하는 것이 핵심이다. cumsum 은 전역 누적이라 시간순
    상태에서 돌리면 동시에 진행 중인 다른 대화가 한 세션에 섞인다.
    """
    c = conv.sort_values(["dyad", "ts"], kind="mergesort").copy()
    gap = c.groupby("dyad")["ts"].diff().dt.total_seconds()
    c["sid"] = (gap.isna() | (gap > gap_hours * 3600)).cumsum()

    bad = c.groupby("sid").dyad.nunique()
    if (bad > 1).any():
        raise RuntimeError("세션에 여러 dyad 가 섞였습니다. 정렬 순서를 확인하세요.")
    return c


# --------------------------------------------------------------------------
# ③ 청킹
# --------------------------------------------------------------------------
def load_tokenizer(name: str):
    from transformers import AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(name)


def count_tokens(tok, texts: list[str], batch: int = 2000,
                 label: str = "", verbose: bool = True) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.int32)
    for i in range(0, len(texts), batch):
        ids = tok(texts[i:i + batch], add_special_tokens=False)["input_ids"]
        out[i:i + len(ids)] = [len(x) for x in ids]
        if verbose:
            print(f"\r  토큰 계산 {label} {min(i + batch, len(texts)):,}/{len(texts):,}",
                  end="", file=sys.stderr)
    if verbose:
        print(file=sys.stderr)
    return out


def header_of(dyad: str, ts_start, ts_end, part: int, n_parts: int) -> str:
    """청크마다 다시 붙는 대화 헤더.

    발화 줄에는 시:분만 있어서 날짜가 없고, 한쪽이 연달아 말한 구간은 화자가
    1명뿐이라 상대방도 알 수 없다(전체 청크의 25.6%). 헤더가 그 둘을 담는다.
    """
    d0, d1 = ts_start.strftime("%Y-%m-%d"), ts_end.strftime("%Y-%m-%d")
    when = d0 if d0 == d1 else f"{d0}~{d1[5:]}"
    tail = f" ({part}/{n_parts})" if n_parts > 1 else ""
    return f"[대화] {dyad} | {when} {ts_start:%H:%M}~{ts_end:%H:%M}{tail}"


def plan_splits(ntok: np.ndarray, budget: int, overlap: int) -> list[tuple[int, int]]:
    """토큰 수 배열 -> [start, end) 구간 목록. 발화 경계에서만 자른다."""
    n = len(ntok)
    spans, start = [], 0
    while start < n:
        end, acc = start, 0
        while end < n and (acc + ntok[end] <= budget or end == start):
            acc += ntok[end]
            end += 1
        spans.append((start, end))
        if end >= n:
            break
        # overlap 만큼 되감되 최소 한 칸은 전진해야 무한루프가 나지 않는다
        back, acc2 = end, 0
        while back > start + 1 and acc2 < overlap:
            back -= 1
            acc2 += ntok[back]
        start = max(back, start + 1)
    return spans


def cut_long_message(tok, prefix: str, body: str, n_pieces: int) -> list[str]:
    """budget 을 혼자 넘는 발화를 토큰 기준 n_pieces 조각으로 나눈다.

    조각마다 '시:분 화자:' 프리픽스를 다시 붙여 화자 귀속을 유지한다.
    n_pieces 는 원문·영문에 같은 값을 쓰므로 조각 수가 어긋나지 않는다.
    """
    ids = tok(body, add_special_tokens=False)["input_ids"]
    if n_pieces <= 1 or not ids:
        return [prefix + body]
    step = -(-len(ids) // n_pieces)
    out = []
    for i in range(n_pieces):
        part = ids[i * step:(i + 1) * step]
        out.append(prefix + (tok.decode(part).strip() if part else "…"))
    return out


def expand(tok, prefix: list[str], bodies: list[str], ntok: np.ndarray,
           pieces: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """긴 발화를 조각으로 펼친다 -> (원본 발화 인덱스, 텍스트, 토큰 수)."""
    e_msg, e_txt, e_tok = [], [], []
    for i, k in enumerate(pieces):
        if k <= 1:
            e_msg.append(i)
            e_txt.append(prefix[i] + bodies[i])
            e_tok.append(int(ntok[i]))
            continue
        parts = cut_long_message(tok, prefix[i], bodies[i], int(k))
        cnt = [len(x) for x in tok(parts, add_special_tokens=False)["input_ids"]]
        e_msg.extend([i] * len(parts))
        e_txt.extend(parts)
        e_tok.extend(cnt)
    return np.asarray(e_msg), e_txt, np.asarray(e_tok)


def build_chunks(sess: pd.DataFrame, tok, chunk_size: int, overlap: int,
                 basis: str, verbose: bool = True) -> tuple[list[dict], list[dict]]:
    """세션 -> 청크 레코드(원문/영문). basis 에 따라 분할 경계를 맞추거나 따로 잡는다."""
    hdr_budget = 32                          # 헤더 몫으로 떼어두는 토큰
    budget = chunk_size - hdr_budget

    prefix = [f"{ts:%H:%M} {nk}: " for ts, nk in zip(sess.ts, sess.nick_f)]
    bodies = {f: sess[f].tolist() for f in ("ru", "en")}
    lines = {f: [p + b for p, b in zip(prefix, bodies[f])] for f in ("ru", "en")}
    ntok = {f: count_tokens(tok, lines[f], label=f, verbose=verbose)
            for f in ("ru", "en")}

    # 조각 수는 두 언어에 같은 값을 써야 청크 인덱스가 일치한다.
    if basis == "max":
        base = np.maximum(ntok["ru"], ntok["en"])
        plan_src = {"ru": base, "en": base}
    elif basis in ("ru", "en"):
        plan_src = {"ru": ntok[basis], "en": ntok[basis]}
    else:                                    # each - 언어별로 따로 (인덱스 어긋남)
        plan_src = ntok

    exp = {f: expand(tok, prefix, bodies[f], ntok[f],
                     np.maximum(1, np.ceil(plan_src[f] / budget).astype(int)))
           for f in ("ru", "en")}

    sid_arr = sess.sid.to_numpy()
    ts_arr = sess.ts.to_numpy()
    row_arr = sess.src_row.to_numpy()
    dyad_arr = sess.dyad.to_numpy()

    out = {"ru": [], "en": []}
    for field in ("ru", "en"):
        e_msg, e_txt, e_tok = exp[field]
        e_plan = (np.maximum(exp["ru"][2], exp["en"][2])
                  if basis == "max" else
                  (exp[basis][2] if basis in ("ru", "en") else e_tok))
        e_sid = sid_arr[e_msg]
        for sid in pd.unique(e_sid):
            idx = np.flatnonzero(e_sid == sid)
            spans = plan_splits(e_plan[idx], budget, overlap)
            for part, (a, b) in enumerate(spans, start=1):
                sel = idx[a:b]
                msgs = pd.unique(e_msg[sel])
                dyad = dyad_arr[msgs[0]]
                t0, t1 = pd.Timestamp(ts_arr[msgs[0]]), pd.Timestamp(ts_arr[msgs[-1]])
                out[field].append({
                    "chunk_index": len(out[field]),
                    "session_id": int(sid),
                    "kind": "conversation",
                    "participants": dyad.split(" | "),
                    "ts_start": t0.isoformat(), "ts_end": t1.isoformat(),
                    "n_messages": int(len(msgs)),
                    "part": part, "n_parts": len(spans),
                    "n_tokens": int(e_tok[sel].sum()) + hdr_budget,
                    "src_rows": [int(row_arr[m]) for m in msgs],
                    "text": (header_of(dyad, t0, t1, part, len(spans)) + "\n"
                             + "\n".join(e_txt[i] for i in sel)),
                })
    return out["ru"], out["en"]


def build_broadcast_chunks(bcasts: list[dict], start_index: dict) -> dict:
    """공지 -> 청크. 수신자 목록을 본문에 넣어 그 시점 명부를 검색 가능하게 한다."""
    out = {"ru": [], "en": []}
    for i, b in enumerate(bcasts):
        for field in ("ru", "en"):
            recips = ", ".join(b["recipients"])
            head = (f"[공지] {b['sender']} -> {b['n_recipients']}명 | "
                    f"{b['ts_start']:%Y-%m-%d %H:%M}")
            out[field].append({
                "chunk_index": start_index[field] + i,
                "session_id": -(i + 1),
                "kind": "broadcast",
                "participants": [b["sender"]],
                "recipients": b["recipients"],
                "ts_start": b["ts_start"].isoformat(),
                "ts_end": b["ts_end"].isoformat(),
                "n_messages": len(b["src_rows"]),
                "part": 1, "n_parts": 1,
                "n_tokens": None,
                "src_rows": b["src_rows"],
                "text": f"{head}\n수신: {recips}\n{b['sender']}: {b[field]}",
            })
    return out


# --------------------------------------------------------------------------
def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(tag: str, rows: list[dict], chunk_size: int) -> None:
    conv = [r for r in rows if r["kind"] == "conversation"]
    n = np.array([r["n_tokens"] for r in conv])
    over = int((n > chunk_size).sum())
    print(f"  {tag:<10} 청크 {len(rows):>7,}  (대화 {len(conv):,} + 공지 "
          f"{len(rows) - len(conv):,})  토큰 중앙 {int(np.median(n)):>4} "
          f"최대 {int(n.max()):>6}  초과 {over}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="jabberchat2020.csv -> 검색용 청크 (원문/영문)")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--gap-hours", type=float, default=1.0,
                    help="이 시간 이상 끊기면 다른 주제로 보고 세션 분리 (기본 1)")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--split-basis", choices=("max", "ru", "en", "each"),
                    default="max",
                    help="분할 경계 기준. max/ru/en 은 원문·영문 청크 인덱스를 "
                         "일치시켜 QA 정답을 공유할 수 있게 한다 (기본 max). "
                         "each 는 각각 따로 잘라 인덱스가 어긋난다.")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--stats-only", action="store_true", help="저장하지 않는다")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Windows 기본 콘솔 코드페이지에서 한글이 깨지지 않게 한다.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if args.overlap >= args.chunk_size:
        raise SystemExit("--overlap 은 --chunk-size 보다 작아야 합니다.")
    verbose = not args.quiet

    conv, bcasts = clean(args.csv, verbose=verbose)
    sess = split_sessions(conv, args.gap_hours)
    if verbose:
        size = sess.groupby("sid").size()
        print(f"\n[세션] gap {args.gap_hours}시간 -> {len(size):,}개 "
              f"(메시지 중앙 {int(size.median())}, 최대 {int(size.max())})",
              file=sys.stderr)

    tok = load_tokenizer(args.tokenizer)
    ru, en = build_chunks(sess, tok, args.chunk_size, args.overlap,
                          args.split_basis, verbose=verbose)
    extra = build_broadcast_chunks(bcasts, {"ru": len(ru), "en": len(en)})
    ru += extra["ru"]
    en += extra["en"]

    print(f"\n[청크] {args.chunk_size}/{args.overlap} 토큰, "
          f"경계기준={args.split_basis}")
    report("jabber_ru", ru, args.chunk_size)
    report("jabber_en", en, args.chunk_size)
    if args.split_basis != "each" and len(ru) != len(en):
        print("  [!] 청크 수가 다릅니다. 정답 공유가 불가능합니다.")

    if args.stats_only:
        print("\nSTATS-ONLY: 저장하지 않았습니다.")
        return

    write_jsonl(os.path.join(args.out, "jabber_ru.jsonl"), ru)
    write_jsonl(os.path.join(args.out, "jabber_en.jsonl"), en)
    meta = [{k: v for k, v in r.items() if k != "text"} for r in ru]
    write_jsonl(os.path.join(args.out, "sessions.jsonl"), meta)
    print(f"\n저장 위치: {args.out}")


if __name__ == "__main__":
    main()
