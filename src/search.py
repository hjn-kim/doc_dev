# -*- coding: utf-8 -*-
"""BGE-M3 / KURE-v1 검색 평가 (dense, sparse, 하이브리드).

질문을 인코딩 -> 문서 청크 벡터와 유사도 -> 상위 10개 순위
-> Recall@5 / Recall@10 / MRR@10 / nDCG@10 을 계산하고 result.csv 로 저장한다.

비교하는 6가지 구성 (질문 인코더 / 문서 인코더 / 방식)

    bb-dense    bge  / bge   dense
    bb-sparse   bge  / bge   sparse          어휘 매칭만
    bb-hybrid   bge  / bge   dense + sparse
    bk          bge  / kure  dense
    kb          kure / bge   dense
    kk          kure / kure  dense

sparse 를 쓰는 구성은 bge-bge 뿐이다. sparse 헤드(sparse_linear.pt)는 BAAI/bge-m3
저장소에만 있고 KURE-v1 에는 없다.

    하이브리드 점수 = dense 코사인 + (--sparse-weight) x sparse 어휘 점수

문서 청크 벡터는 모델별로 다른 폴더에서 읽는다.
    bge  -> data/emb/bgem3   (embedding_bge_txt.py 로 생성, dense + sparse)
    kure -> data/emb/kurev1  (embedding_kure_txt.py 로 생성, dense 전용)

검색 범위는 기본이 --scope doc (질문이 속한 문서 안에서만 검색) 이다.
합성증거 문서 7종은 같은 20사건을 7개 언어로 번역한 것이라, --scope corpus 로 두면
한국어 질의가 다른 언어판 대신 한국어판 청크를 집어내 다국어 점수가 무너진다.

사용 예:
    python src/search.py                                      # 6개 구성 -> result.csv
    python src/search.py --configs bb-dense,bb-sparse,bb-hybrid
    python src/search.py --sparse-weight 0.5
    python src/search.py --neighbor-tolerance 1               # 인접 청크도 정답 인정
    python src/search.py --scope corpus
    python src/search.py --csv out/exp.csv --dump out/exp.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_DIR = os.path.join(ROOT, "data", "qa")
DEFAULT_CSV = os.path.join(ROOT, "result.csv")

MODELS = {
    "bge":  {"id": "BAAI/bge-m3",
             "emb": os.path.join(ROOT, "data", "emb", "bgem3"),
             "script": "embedding_bge_txt.py"},
    "kure": {"id": "nlpai-lab/KURE-v1",
             "emb": os.path.join(ROOT, "data", "emb", "kurev1"),
             "script": "embedding_kure_txt.py"},
}

# 구성 이름 -> (질문 인코더, 문서 인코더, 방식)
#   dense  : 코사인만 / sparse : 어휘 매칭만 / hybrid : dense + w x sparse
CONFIGS = {
    "bb-dense":  ("bge", "bge", "dense"),
    "bb-sparse": ("bge", "bge", "sparse"),
    "bb-hybrid": ("bge", "bge", "hybrid"),
    "bk":        ("bge", "kure", "dense"),
    "kb":        ("kure", "bge", "dense"),
    "kk":        ("kure", "kure", "dense"),
}
DEFAULT_CONFIGS = "bb-dense,bb-sparse,bb-hybrid,bk,kb,kk"
MODE_LABEL = {"dense": "dense", "sparse": "sparse", "hybrid": "dense + sparse"}

K_LIST = (5, 10)
K_MAX = 10

# 문서 태그 앞부분에서 언어를 판별한다. 질문은 항상 한국어이므로 ko 문서는
# 동일언어 검색, 나머지는 교차언어 검색이 되어 성격이 다르다. 그래서 따로 집계한다.
LANG_PREFIXES = ("ko", "ch", "en", "pil", "pl", "rs", "uz", "vn")
KO_LANG = "ko"

# 과제 유형: 증거찾기(합성증거 QA) vs 답변찾기(법령·보고서 QA)
TASK_EVIDENCE = "증거찾기"
TASK_ANSWER = "답변찾기"

GROUP_ALL = "ALL"
GROUP_KO = "ALL(한국어)"
GROUP_MULTI = "ALL(다국어)"
GROUP_EVID = "ALL(증거찾기)"
GROUP_ANS = "ALL(답변찾기)"

CSV_FIELDS = ["config", "question_model", "doc_model", "mode", "sparse_weight",
              "scope", "neighbor_tolerance", "language", "task", "document",
              "n_queries", "recall@5", "recall@10", "mrr@10", "ndcg@10"]


# --------------------------------------------------------------------------
# 데이터 로딩
# --------------------------------------------------------------------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def doc_language(tag: str) -> str:
    """'pil' 이 'pl' 보다 먼저 걸리도록 긴 접두사부터 확인한다."""
    for pre in sorted(LANG_PREFIXES, key=len, reverse=True):
        if tag.startswith(pre):
            return pre
    return "?"


def doc_task(tag: str) -> str:
    return TASK_EVIDENCE if "합성증거" in tag else TASK_ANSWER


def load_qa(qa_dir: str) -> list[dict]:
    """QA json 을 읽어 문서 목록을 만든다. 청크 벡터는 아직 읽지 않는다."""
    docs = []
    for qa_path in sorted(glob.glob(os.path.join(qa_dir, "*_qa.json"))):
        with open(qa_path, encoding="utf-8") as f:
            qa = json.load(f)
        tag = os.path.basename(qa_path).split("_qa.json")[0]
        for pair in qa["qa_pairs"]:
            # 구버전은 정답 청크가 하나(answer_chunk_index),
            # 합성증거 QA 는 여러 개(answer_chunk_indices)일 수 있다. 리스트로 통일한다.
            gold = (list(pair["answer_chunk_indices"]) if "answer_chunk_indices" in pair
                    else [pair["answer_chunk_index"]])
            if not gold:
                raise ValueError(f"{qa['source']} {pair['id']}: 정답 청크가 비어 있습니다.")
            pair["_gold"] = gold
        docs.append({
            "tag": tag,
            "source": qa["source"],
            "npz_name": qa.get("embedding_file") or f"{tag}_embeddings.npz",
            "language": doc_language(tag),
            "task": doc_task(tag),
            "qa_pairs": qa["qa_pairs"],
        })
    if not docs:
        raise SystemExit(f"QA 파일이 없습니다: {qa_dir}")
    return docs


def load_vectors(docs: list[dict], model_key: str, need_sparse: bool):
    """모델별 폴더에서 청크 벡터를 읽어 전역 행렬(dense)과 CSR(sparse)을 만든다."""
    emb_dir = MODELS[model_key]["emb"]
    script = MODELS[model_key]["script"]
    if not os.path.isdir(emb_dir) or not glob.glob(os.path.join(emb_dir, "*.npz")):
        raise SystemExit(
            f"[{model_key}] 임베딩이 없습니다: {emb_dir}\n"
            f"  먼저 만들어 주세요: python data/{script} "
            f"--out {os.path.relpath(emb_dir, ROOT)} --device cuda")

    mats, sparse_parts, offset = [], [], 0
    layout = {}
    for doc in docs:
        path = os.path.join(emb_dir, doc["npz_name"])
        if not os.path.exists(path):
            stem = doc["tag"]
            alt = os.path.join(emb_dir, f"{stem}_embeddings.npz")
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(
                    f"[{model_key}] {doc['tag']} 의 임베딩이 없습니다.\n  찾은 곳: {path}")
        npz = np.load(path, allow_pickle=True)
        emb = l2_normalize(np.asarray(npz["embeddings"], dtype=np.float32))
        n = emb.shape[0]

        for pair in doc["qa_pairs"]:
            for g in pair["_gold"]:
                if not (0 <= g < n):
                    raise ValueError(
                        f"{doc['source']} {pair['id']}: 정답 청크 {g} 가 "
                        f"범위(0~{n - 1})를 벗어납니다. 임베딩과 QA 가 어긋난 상태입니다.")

        if need_sparse:
            if "sparse_indices" not in npz:
                raise SystemExit(
                    f"[{model_key}] {doc['tag']} 에 sparse 가 없습니다.\n"
                    f"  하이브리드 구성을 쓰려면 embedding_bge_txt.py 로 다시 만들어야 합니다"
                    f" (--no-sparse 없이).")
            sparse_parts.append((npz["sparse_indices"], npz["sparse_values"],
                                 npz["sparse_indptr"], int(npz["sparse_dim"])))

        layout[doc["tag"]] = {"offset": offset, "n_chunks": n}
        mats.append(emb)
        offset += n

    dense_all = np.vstack(mats)
    sparse_all = _stack_csr(sparse_parts) if need_sparse else None
    return dense_all, sparse_all, layout


def _stack_csr(parts):
    """문서별 CSR 조각을 하나의 큰 CSR 로 이어붙인다."""
    from scipy.sparse import csr_matrix
    dim = parts[0][3]
    idx_all, val_all = [], []
    ptr_all = [np.array([0], dtype=np.int64)]
    total = 0
    for idx, val, ptr, d in parts:
        if d != dim:
            raise ValueError(f"sparse 어휘 크기가 다릅니다: {d} vs {dim}")
        idx_all.append(idx)
        val_all.append(val)
        ptr_all.append(ptr[1:] + total)
        total += len(idx)
    return csr_matrix((np.concatenate(val_all), np.concatenate(idx_all),
                       np.concatenate(ptr_all)),
                      shape=(sum(len(p[2]) - 1 for p in parts), dim))


# --------------------------------------------------------------------------
# 질문 인코딩
# --------------------------------------------------------------------------
class Encoder:
    """dense(CLS + L2 정규화), 필요하면 sparse(BGE-M3 lexical 가중치)도 함께."""

    def __init__(self, model_key: str, device: str | None = None,
                 max_length: int = 512, with_sparse: bool = False):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_key = model_key
        self.model_id = MODELS[model_key]["id"]
        self.with_sparse = with_sparse
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(device).eval()
        self.skip_ids = {i for i in (self.tok.cls_token_id, self.tok.sep_token_id,
                                     self.tok.eos_token_id, self.tok.pad_token_id,
                                     self.tok.unk_token_id) if i is not None}
        self.vocab_size = int(self.model.config.vocab_size)

        self.sparse_linear = None
        if with_sparse:
            from huggingface_hub import hf_hub_download
            state = torch.load(hf_hub_download("BAAI/bge-m3", "sparse_linear.pt"),
                               map_location="cpu")
            lin = torch.nn.Linear(self.model.config.hidden_size, 1)
            lin.load_state_dict(state)
            self.sparse_linear = lin.to(device).eval()

    def encode(self, texts, batch_size: int = 16, label: str = "", verbose: bool = False):
        torch = self.torch
        dense_out, sparse_out = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                dense = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)
                sw = (torch.relu(self.sparse_linear(hidden)).squeeze(-1)
                      if self.sparse_linear is not None else None)
            dense_out.append(dense.float().cpu().numpy())
            if sw is not None:
                w = sw.float().cpu().numpy()
                ids = enc["input_ids"].cpu().numpy()
                mask = enc["attention_mask"].cpu().numpy()
                for row_ids, row_w, row_m in zip(ids, w, mask):
                    d: dict[int, float] = {}
                    for tid, val, m in zip(row_ids, row_w, row_m):
                        if not m or val <= 0:
                            continue
                        tid = int(tid)
                        if tid in self.skip_ids:
                            continue
                        if val > d.get(tid, 0.0):
                            d[tid] = float(val)
                    sparse_out.append(d)
            if verbose:
                print(f"\r  {label} {min(i + batch_size, len(texts))}/{len(texts)}",
                      end="", file=sys.stderr)
        if verbose:
            print(file=sys.stderr)
        dense = l2_normalize(np.vstack(dense_out).astype(np.float32))
        return dense, (sparse_out if self.sparse_linear is not None else None)


def dicts_to_csr(dicts, vocab_size: int):
    from scipy.sparse import csr_matrix
    indptr = np.zeros(len(dicts) + 1, dtype=np.int64)
    idx, val = [], []
    for i, d in enumerate(dicts):
        for k in sorted(d):
            idx.append(k)
            val.append(d[k])
        indptr[i + 1] = len(idx)
    return csr_matrix((np.asarray(val, dtype=np.float32),
                       np.asarray(idx, dtype=np.int32), indptr),
                      shape=(len(dicts), vocab_size))


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------
def relevant_set(gold: list[int], n_chunks: int, tolerance: int) -> set[int]:
    """정답 청크(여러 개일 수 있음)와 overlap 보정용 인접 청크를 합친 집합."""
    rel: set[int] = set()
    for g in gold:
        rel.update(range(max(0, g - tolerance), min(n_chunks - 1, g + tolerance) + 1))
    return rel


def query_metrics(ranked: list[int], rel: set[int]) -> dict:
    """단일 질의 지표. 모두 '첫 적중' 기준이다.

    주의 - recall@k 는 표준 Recall 이 아니라 hit-rate(= success@k) 다.
      표준 Recall@k = |상위 k ∩ 정답| / |정답| 이라 정답 청크가 많을수록 불리하다.
      여기서는 정답 청크 중 하나라도 상위 k 에 들면 1.0 으로 센다. 정답이 여러
      청크에 걸치는 것은 하나의 증거블록이 overlap 때문에 쪼개진 결과일 뿐,
      전부 회수해야 하는 별개 정답이 아니기 때문이다.
    """
    first_rank = 0
    for pos, cid in enumerate(ranked[:K_MAX], start=1):
        if cid in rel:
            first_rank = pos
            break
    m = {f"recall@{k}": float(0 < first_rank <= k) for k in K_LIST}
    m["mrr@10"] = 1.0 / first_rank if first_rank else 0.0
    m["ndcg@10"] = 1.0 / math.log2(first_rank + 1) if first_rank else 0.0
    m["rank"] = first_rank
    return m


def average(rows: list[dict]) -> dict:
    keys = ("recall@5", "recall@10", "mrr@10", "ndcg@10")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------
def evaluate(docs, target_docs, dense_all, sparse_all, layout, encoder,
             scope: str, tolerance: int, sparse_weight: float,
             batch_size: int, verbose: bool, mode: str = "dense"):
    results = []
    for doc in target_docs:
        lay = layout[doc["tag"]]
        off, n_chunks = lay["offset"], lay["n_chunks"]
        questions = [p["question"] for p in doc["qa_pairs"]]
        qd, qs = encoder.encode(questions, batch_size=batch_size,
                                label=f"질문 {doc['tag'][:14]}", verbose=verbose)

        # 검색 범위에 맞춰 문서 벡터를 자른다.
        if scope == "doc":
            dpool = dense_all[off:off + n_chunks]
            spool = None if sparse_all is None else sparse_all[off:off + n_chunks]
        else:
            dpool, spool = dense_all, sparse_all

        sparse_scores = None
        if mode in ("sparse", "hybrid"):
            if spool is None or qs is None:
                raise SystemExit(f"[{mode}] sparse 벡터가 없습니다. "
                                 "embedding_bge_txt.py 로 sparse 를 포함해 만들어야 합니다.")
            qcsr = dicts_to_csr(qs, spool.shape[1])
            sparse_scores = np.asarray((qcsr @ spool.T).todense())

        if mode == "dense":
            scores = qd @ dpool.T
        elif mode == "sparse":
            scores = sparse_scores
        else:
            scores = (qd @ dpool.T) + sparse_weight * sparse_scores

        top = np.argsort(-scores, axis=1)[:, :K_MAX]

        for row, (pair, cand) in enumerate(zip(doc["qa_pairs"], top)):
            if scope == "doc":
                ranked = [int(c) for c in cand]
                ranked_docs = [doc["tag"]] * len(ranked)
            else:
                ranked, ranked_docs = [], []
                for gid in cand:
                    owner = _owner_of(layout, int(gid))
                    ranked.append(int(gid) - layout[owner]["offset"])
                    ranked_docs.append(owner)

            # 전역 검색에서는 다른 문서의 같은 로컬 인덱스가 정답으로 오인되면 안 된다.
            rel = relevant_set(pair["_gold"], n_chunks, tolerance)
            masked = [cid if dtag == doc["tag"] else -1
                      for cid, dtag in zip(ranked, ranked_docs)]

            m = query_metrics(masked, rel)
            results.append({
                "doc": doc["tag"], "language": doc["language"], "task": doc["task"],
                "id": pair["id"], "question": pair["question"], "gold": pair["_gold"],
                "rank": m["rank"], "top_chunks": ranked, "top_docs": ranked_docs,
                "top_scores": [round(float(scores[row, g]), 4) for g in cand],
                **{k: m[k] for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10")},
            })
    return results


def _owner_of(layout, global_id: int) -> str:
    for tag, lay in layout.items():
        if lay["offset"] <= global_id < lay["offset"] + lay["n_chunks"]:
            return tag
    raise IndexError(global_id)


# --------------------------------------------------------------------------
# 출력
# --------------------------------------------------------------------------
def next_available_path(path: str) -> str:
    """result.csv 가 이미 있으면 result(1).csv, result(2).csv ... 로 비켜 쓴다."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{stem}({i}){ext}"):
        i += 1
    return f"{stem}({i}){ext}"


def build_rows(results, cfg_name: str, qm: str, dm: str, mode: str,
               sparse_weight: float, scope: str, tolerance: int) -> list[dict]:
    def row(document, language, task, subset):
        a = average(subset)
        return {
            "config": cfg_name, "question_model": qm, "doc_model": dm,
            "mode": mode,
            "sparse_weight": sparse_weight if mode == "hybrid" else "",
            "scope": scope, "neighbor_tolerance": tolerance,
            "language": language, "task": task, "document": document,
            "n_queries": len(subset),
            **{k: round(a[k], 4) for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10")},
        }

    rows = []
    by_doc = {}
    for r in results:
        by_doc.setdefault(r["doc"], []).append(r)
    for tag in sorted(by_doc):
        s = by_doc[tag]
        rows.append(row(tag, s[0]["language"], s[0]["task"], s))

    ko = [r for r in results if r["language"] == KO_LANG]
    multi = [r for r in results if r["language"] != KO_LANG]
    evid = [r for r in results if r["task"] == TASK_EVIDENCE]
    ans = [r for r in results if r["task"] == TASK_ANSWER]
    if ko:
        rows.append(row(GROUP_KO, KO_LANG, "mixed", ko))
    if multi:
        rows.append(row(GROUP_MULTI, "multi", "mixed", multi))
    if evid:
        rows.append(row(GROUP_EVID, "mixed", TASK_EVIDENCE, evid))
    if ans:
        rows.append(row(GROUP_ANS, "mixed", TASK_ANSWER, ans))
    rows.append(row(GROUP_ALL, "all", "all", results))
    return rows


def print_table(rows, cfg_name: str, qm: str, dm: str, mode: str):
    width = max(len(r["document"]) for r in rows)
    header = (f"{'document'.ljust(width)}  {'N':>4} {'R@5':>7} {'R@10':>7} "
              f"{'MRR@10':>7} {'nDCG@10':>8}")

    print(f"\n[{cfg_name}]  질문={qm}, 문서={dm}, 방식={MODE_LABEL[mode]}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["document"] == GROUP_KO:
            print("-" * len(header))
        print(f"{r['document'].ljust(width)}  {r['n_queries']:>4} "
              f"{r['recall@5']:>7.3f} {r['recall@10']:>7.3f} "
              f"{r['mrr@10']:>7.3f} {r['ndcg@10']:>8.3f}")
    print("-" * len(header))


def compare_block(all_rows, cfg_names, group: str, title: str, note: str = ""):
    """한 그룹(전체/한국어/다국어/과제유형)에서 구성들을 나란히 비교."""
    picked = {r["config"]: r for r in all_rows if r["document"] == group}
    present = [c for c in cfg_names if c in picked]
    if not present:
        return
    n = picked[present[0]]["n_queries"]
    width = 14 + 12 * len(present)
    print(f"{title}  (질의 {n}개)")
    if note:
        print(f"  {note}")
    print("-" * width)
    print(f"{'metric':<14}" + "".join(f"{c:>12}" for c in present))
    print("-" * width)
    for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10"):
        print(f"{k:<14}" + "".join(f"{picked[c][k]:>12.3f}" for c in present))
    print("-" * width)


def task_compare(all_rows, cfg_names):
    """증거찾기 vs 답변찾기를 구성별로 나란히 놓고 비교."""
    ev = {r["config"]: r for r in all_rows if r["document"] == GROUP_EVID}
    an = {r["config"]: r for r in all_rows if r["document"] == GROUP_ANS}
    present = [c for c in cfg_names if c in ev and c in an]
    if not present:
        return
    n_ev = ev[present[0]]["n_queries"]
    n_an = an[present[0]]["n_queries"]
    width = 12 + 11 * 4 + 12
    print("■ 과제 유형 비교 — 증거찾기 vs 답변찾기")
    print(f"  증거찾기: 합성증거 20사건, '~에 해당하는 증거를 찾아주세요' ({n_ev}문항)")
    print(f"  답변찾기: 법령·보고서, '~는 무엇인가' ({n_an}문항)")
    print("-" * width)
    print(f"{'config':<12}" + f"{'증거 R@5':>11}{'증거 nDCG':>11}"
          + f"{'답변 R@5':>11}{'답변 nDCG':>11}" + f"{'nDCG 차이':>12}")
    print("-" * width)
    for c in present:
        d = an[c]["ndcg@10"] - ev[c]["ndcg@10"]
        print(f"{c:<12}{ev[c]['recall@5']:>11.3f}{ev[c]['ndcg@10']:>11.3f}"
              f"{an[c]['recall@5']:>11.3f}{an[c]['ndcg@10']:>11.3f}{d:>+12.3f}")
    print("-" * width)
    print("  (nDCG 차이 = 답변찾기 − 증거찾기. 양수면 답변찾기가 더 쉽다는 뜻)")


def print_comparison(all_rows, cfg_names):
    print("\n" * 3, end="")
    compare_block(all_rows, cfg_names, GROUP_KO, "■ 구성 비교 — 한국어 문서",
                  "질문·본문이 모두 한국어 (동일언어 검색)")
    print("\n" * 3, end="")
    compare_block(all_rows, cfg_names, GROUP_MULTI, "■ 구성 비교 — 외국어 문서 (한국어 제외)",
                  "질문은 한국어, 본문은 ch/en/pil/rs/uz/vn (교차언어 검색)")
    print("\n" * 3, end="")
    task_compare(all_rows, cfg_names)
    print("\n※ recall@k 는 hit-rate 입니다. 정답 청크 중 하나라도 상위 k 에 들면 1.0 으로"
          " 세며, 표준 Recall(|상위k ∩ 정답| / |정답|)이 아닙니다.")


def print_misses(results, limit: int):
    misses = [r for r in results if r["rank"] == 0 or r["rank"] > 3]
    if not misses:
        print("\n순위 3위 밖으로 밀린 질의가 없습니다.")
        return
    misses.sort(key=lambda r: (r["rank"] == 0, -r["rank"]), reverse=True)
    print(f"\n실패/저순위 질의 상위 {min(limit, len(misses))}건 (총 {len(misses)}건)")
    print("-" * 78)
    for r in misses[:limit]:
        rank = r["rank"] if r["rank"] else "10위 밖"
        print(f"[{r['id']}] rank={rank}  gold={r['gold']}")
        print(f"  Q: {r['question'][:70]}")
        print("  검색된 상위 3: " + ", ".join(
            f"{d[:12]}:{c}({s})" for d, c, s
            in zip(r["top_docs"][:3], r["top_chunks"][:3], r["top_scores"][:3])))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="BGE-M3 / KURE-v1 검색 평가 (dense / 하이브리드)")
    ap.add_argument("--configs", default=DEFAULT_CONFIGS,
                    help=f"쉼표 구분. 사용 가능: {', '.join(CONFIGS)}")
    ap.add_argument("--qa-dir", default=QA_DIR)
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="결과 CSV. 이미 있으면 result(1).csv 처럼 번호를 붙인다.")
    ap.add_argument("--overwrite", action="store_true", help="번호를 붙이지 않고 덮어쓴다")
    ap.add_argument("--sparse-weight", type=float, default=0.3,
                    help="하이브리드 점수 = dense + w x sparse (기본 0.3)")
    ap.add_argument("--device", default=None, help="cuda / cpu (기본: 자동)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--scope", choices=("doc", "corpus"), default="doc",
                    help="doc: 질문이 속한 문서 안에서만 검색(기본) / corpus: 전체 문서")
    ap.add_argument("--neighbor-tolerance", type=int, default=0,
                    help="정답 청크 ±N 을 정답으로 인정 (overlap 보정용)")
    ap.add_argument("--doc", default=None, help="특정 문서만 평가 (파일명 앞부분)")
    ap.add_argument("--dump", default=None, help="질의별 결과를 JSON 으로 저장")
    ap.add_argument("--show-misses", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in cfg_names if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"알 수 없는 구성: {unknown}. 사용 가능: {list(CONFIGS)}")

    docs = load_qa(args.qa_dir)
    target = docs
    if args.doc:
        target = [d for d in docs if d["tag"].startswith(args.doc)]
        if not target:
            raise SystemExit(f"'{args.doc}' 로 시작하는 문서가 없습니다.")
    n_q = sum(len(d["qa_pairs"]) for d in target)
    verbose = not args.quiet

    if verbose:
        print(f"문서 {len(target)}개 / 질문 {n_q}개 / 구성 {len(cfg_names)}개", file=sys.stderr)

    # 문서 벡터는 모델별로 한 번만 읽어 재사용한다.
    vec_cache: dict[tuple[str, bool], tuple] = {}
    enc_cache: dict[tuple[str, bool], Encoder] = {}
    all_rows, dumps = [], {}

    for cfg in cfg_names:
        qm, dm, mode = CONFIGS[cfg]
        need_sparse = mode in ("sparse", "hybrid")
        if verbose:
            print(f"\n=== {cfg}: 질문={qm}, 문서={dm}, "
                  f"{MODE_LABEL[mode]} ===", file=sys.stderr)

        vkey = (dm, need_sparse)
        if vkey not in vec_cache:
            vec_cache[vkey] = load_vectors(docs, dm, need_sparse=need_sparse)
        dense_all, sparse_all, layout = vec_cache[vkey]

        ekey = (qm, need_sparse)
        if ekey not in enc_cache:
            t0 = time.time()
            enc_cache[ekey] = Encoder(qm, device=args.device, with_sparse=need_sparse)
            if verbose:
                print(f"모델 로딩 {time.time() - t0:.1f}s "
                      f"(device={enc_cache[ekey].device})", file=sys.stderr)
        encoder = enc_cache[ekey]

        t0 = time.time()
        results = evaluate(docs, target, dense_all, sparse_all, layout, encoder,
                           args.scope, args.neighbor_tolerance, args.sparse_weight,
                           args.batch_size, verbose, mode=mode)
        elapsed = time.time() - t0

        rows = build_rows(results, cfg, qm, dm, mode, args.sparse_weight,
                          args.scope, args.neighbor_tolerance)
        all_rows.extend(rows)
        dumps[cfg] = results
        print_table(rows, cfg, qm, dm, mode)
        if verbose:
            print(f"검색 소요: {elapsed:.1f}s ({elapsed / n_q * 1000:.0f} ms/질의)")
        if args.show_misses:
            print_misses(results, args.show_misses)

    print_comparison(all_rows, cfg_names)

    csv_path = args.csv if args.overwrite else next_available_path(args.csv)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCSV 저장: {csv_path}  ({len(all_rows)}행)")

    if args.dump:
        dump_path = args.dump if args.overwrite else next_available_path(args.dump)
        os.makedirs(os.path.dirname(os.path.abspath(dump_path)) or ".", exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump({"config": {"configs": cfg_names, "scope": args.scope,
                                  "sparse_weight": args.sparse_weight,
                                  "neighbor_tolerance": args.neighbor_tolerance},
                       "summary": all_rows, "queries": dumps}, f,
                      ensure_ascii=False, indent=2)
        print(f"JSON 저장: {dump_path}")


if __name__ == "__main__":
    main()
