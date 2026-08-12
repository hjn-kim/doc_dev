# -*- coding: utf-8 -*-
"""BGE-M3 / KURE-v1 질문 임베딩 검색 평가.

질문을 각 모델로 인코딩 -> data/emb 의 청크 임베딩과 내적 -> 상위 10개 청크 순위
-> Recall@5 / Recall@10 / MRR@10 / nDCG@10 을 계산하고 result.csv 로 저장한다.

두 모델 모두 XLM-RoBERTa 기반, 1024차원, CLS 풀링 + L2 정규화라 파이프라인이 같다.
(KURE-v1 은 BGE-M3 를 한국어로 파인튜닝한 모델이다.)

주의: 기본값은 data/emb 에 저장된 BGE-M3 청크 벡터를 그대로 쓰고 질문 쪽 인코더만
바꾼다. KURE-v1 질문 벡터를 BGE-M3 청크 벡터와 내적하는 것은 엄밀히는 서로 다른
벡터 공간을 비교하는 것이므로, 공정한 모델 비교를 원하면 --reencode-chunks 를 써서
청크도 같은 모델로 다시 인코딩해야 한다.

사용 예:
    python src/search.py                              # 두 모델 비교 -> result.csv
    python src/search.py --models bge-m3              # 한 모델만
    python src/search.py --reencode-chunks            # 청크도 같은 모델로 재인코딩(공정 비교)
    python src/search.py --neighbor-tolerance 1       # 인접 청크도 정답 인정(overlap 보정)
    python src/search.py --scope doc                  # 해당 문서 안에서만 검색
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
EMB_DIR = os.path.join(ROOT, "data", "emb")
QA_DIR = os.path.join(ROOT, "data", "qa")
CACHE_DIR = os.path.join(ROOT, "data", "emb_cache")
DEFAULT_CSV = os.path.join(ROOT, "result.csv")

MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "kure-v1": "nlpai-lab/KURE-v1",
}
STORED_CHUNK_MODEL = "bge-m3"      # data/emb 의 청크 벡터를 만든 모델
K_LIST = (5, 10)
K_MAX = 10
CSV_FIELDS = ["model", "model_id", "chunk_encoder", "scope", "neighbor_tolerance",
              "document", "n_queries", "recall@5", "recall@10", "mrr@10", "ndcg@10"]


# --------------------------------------------------------------------------
# 데이터 로딩
# --------------------------------------------------------------------------
def load_corpus(qa_dir: str, emb_dir: str):
    """QA json + 대응하는 npz 를 읽어 문서 리스트와 전역 임베딩 행렬을 만든다."""
    docs, mats, offset = [], [], 0

    for qa_path in sorted(glob.glob(os.path.join(qa_dir, "*_qa.json"))):
        with open(qa_path, encoding="utf-8") as f:
            qa = json.load(f)

        emb_path = os.path.join(emb_dir, qa.get("embedding_file", ""))
        if not os.path.exists(emb_path):
            # 파일명을 바꾼 뒤 json 안의 embedding_file 이 낡은 경우가 있어
            # QA 파일명 기준으로 한 번 더 찾아본다.
            stem = os.path.basename(qa_path).split("_qa.json")[0]
            fallback = os.path.join(emb_dir, f"{stem}_embeddings.npz")
            if os.path.exists(fallback):
                emb_path = fallback
            else:
                raise FileNotFoundError(
                    f"{os.path.basename(qa_path)} 에 대응하는 임베딩을 찾지 못했습니다.\n"
                    f"  json 의 embedding_file: {qa.get('embedding_file')}\n"
                    f"  파일명 기준 폴백:       {os.path.basename(fallback)}"
                )

        npz = np.load(emb_path, allow_pickle=True)
        emb = l2_normalize(np.asarray(npz["embeddings"], dtype=np.float32))
        texts = [str(t) for t in npz["texts"]]
        n_chunks = emb.shape[0]

        for pair in qa["qa_pairs"]:
            gold = pair["answer_chunk_index"]
            if not (0 <= gold < n_chunks):
                raise ValueError(
                    f"{qa['source']} {pair['id']}: answer_chunk_index={gold} 가 "
                    f"청크 범위(0~{n_chunks - 1})를 벗어납니다."
                )

        docs.append({
            "source": qa["source"],
            "tag": os.path.basename(qa_path).split("_qa.json")[0],
            "n_chunks": n_chunks,
            "offset": offset,          # 전역 행렬에서의 시작 위치
            "texts": texts,
            "qa_pairs": qa["qa_pairs"],
        })
        mats.append(emb)
        offset += n_chunks

    if not docs:
        raise SystemExit(f"QA 파일이 없습니다: {qa_dir}")

    return docs, np.vstack(mats)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------
# 인코딩
# --------------------------------------------------------------------------
class Encoder:
    """BGE-M3 / KURE-v1 dense 임베딩 = 마지막 레이어 CLS 토큰 + L2 정규화.

    두 모델 모두 질의에 지시문(instruction) 접두사를 붙이지 않는다.
    """

    def __init__(self, model_id: str, device: str | None = None, max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    def encode(self, texts, batch_size: int = 16, label: str = "", verbose: bool = False):
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            vec = hidden[:, 0]                                  # CLS pooling
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
            out.append(vec.float().cpu().numpy())
            if verbose:
                print(f"\r  {label} {min(i + batch_size, len(texts))}/{len(texts)}",
                      end="", file=sys.stderr)
        if verbose:
            print(file=sys.stderr)
        return np.vstack(out).astype(np.float32)


def chunk_matrix(docs, emb_all, model_key: str, encoder, reencode: bool,
                 batch_size: int, verbose: bool):
    """청크 임베딩 행렬을 준비한다.

    reencode=False 면 data/emb 에 저장된 BGE-M3 벡터를 그대로 쓴다.
    reencode=True 면 해당 모델로 청크를 다시 인코딩하고 결과를 캐시한다.
    """
    if not reencode:
        return emb_all, f"{STORED_CHUNK_MODEL}(stored)"

    cache_dir = os.path.join(CACHE_DIR, model_key)
    os.makedirs(cache_dir, exist_ok=True)
    mats = []
    for doc in docs:
        cache = os.path.join(cache_dir, f"{doc['tag']}.npy")
        if os.path.exists(cache):
            mat = np.load(cache)
            if mat.shape[0] != doc["n_chunks"]:
                raise ValueError(f"캐시 청크 수 불일치: {cache}")
        else:
            if verbose:
                print(f"  [{model_key}] 청크 재인코딩: {doc['tag'][:20]} "
                      f"({doc['n_chunks']}개)", file=sys.stderr)
            mat = encoder.encode(doc["texts"], batch_size=batch_size,
                                 label="청크", verbose=verbose)
            np.save(cache, mat)
        mats.append(l2_normalize(mat.astype(np.float32)))
    return np.vstack(mats), f"{model_key}(re-encoded)"


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------
def relevant_set(gold: int, n_chunks: int, tolerance: int) -> set[int]:
    """overlap 때문에 정답 근거가 인접 청크에 걸칠 수 있어 허용 범위를 둔다."""
    return set(range(max(0, gold - tolerance), min(n_chunks - 1, gold + tolerance) + 1))


def query_metrics(ranked: list[int], rel: set[int]) -> dict:
    """단일 질의 지표.

    질의마다 정답 청크는 하나다. tolerance>0 으로 rel 이 여러 개가 되더라도
    그것은 "인접 청크를 맞혀도 정답으로 인정한다"는 관대한 매칭을 뜻하지,
    회수해야 할 정답이 늘어난다는 뜻이 아니다. 따라서 첫 적중만 센다.
    (tolerance=0 이면 정답 1개짜리 표준 지표와 정확히 같아진다.)
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
def evaluate(all_docs, target_docs, chunk_emb, encoder, scope: str, tolerance: int,
             batch_size: int, verbose: bool):
    """target_docs 의 질문만 평가한다.

    scope='corpus' 일 때 후보는 항상 전체 코퍼스이므로, 검색 결과가 어느 문서의
    청크인지 되찾을 때는 필터링된 target_docs 가 아니라 all_docs 를 봐야 한다.
    """
    results = []

    for doc in target_docs:
        questions = [p["question"] for p in doc["qa_pairs"]]
        qvec = encoder.encode(questions, batch_size=batch_size,
                              label=f"질문 {doc['tag'][:14]}", verbose=verbose)

        if scope == "doc":
            pool = chunk_emb[doc["offset"]: doc["offset"] + doc["n_chunks"]]
            scores = qvec @ pool.T
        else:
            scores = qvec @ chunk_emb.T

        top = np.argsort(-scores, axis=1)[:, :K_MAX]

        for row, (pair, cand) in enumerate(zip(doc["qa_pairs"], top)):
            if scope == "doc":
                ranked = [int(c) for c in cand]
                ranked_docs = [doc["tag"]] * len(ranked)
            else:
                ranked, ranked_docs = [], []
                for gid in cand:
                    owner = _owner_of(all_docs, int(gid))
                    ranked.append(int(gid) - owner["offset"])
                    ranked_docs.append(owner["tag"])

            # 전역 검색에서는 다른 문서의 같은 로컬 인덱스가 정답으로 오인되면 안 된다.
            rel = relevant_set(pair["answer_chunk_index"], doc["n_chunks"], tolerance)
            masked = [cid if dtag == doc["tag"] else -1
                      for cid, dtag in zip(ranked, ranked_docs)]

            m = query_metrics(masked, rel)
            results.append({
                "doc": doc["tag"],
                "id": pair["id"],
                "question": pair["question"],
                "gold": pair["answer_chunk_index"],
                "rank": m["rank"],
                "top_chunks": ranked,
                "top_docs": ranked_docs,
                "top_scores": [round(float(scores[row, g]), 4) for g in cand],
                **{k: m[k] for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10")},
            })

    return results


def _owner_of(docs, global_id: int):
    for doc in docs:
        if doc["offset"] <= global_id < doc["offset"] + doc["n_chunks"]:
            return doc
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


def build_rows(results, model_key: str, model_id: str, chunk_encoder: str,
               scope: str, tolerance: int) -> list[dict]:
    rows = []
    by_doc = {}
    for r in results:
        by_doc.setdefault(r["doc"], []).append(r)

    for tag in sorted(by_doc):
        a = average(by_doc[tag])
        rows.append({
            "model": model_key, "model_id": model_id, "chunk_encoder": chunk_encoder,
            "scope": scope, "neighbor_tolerance": tolerance,
            "document": tag, "n_queries": len(by_doc[tag]),
            **{k: round(a[k], 4) for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10")},
        })

    a = average(results)
    rows.append({
        "model": model_key, "model_id": model_id, "chunk_encoder": chunk_encoder,
        "scope": scope, "neighbor_tolerance": tolerance,
        "document": "ALL", "n_queries": len(results),
        **{k: round(a[k], 4) for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10")},
    })
    return rows


def print_table(rows: list[dict], model_key: str, chunk_encoder: str):
    width = max(len(r["document"]) for r in rows)
    header = (f"{'document'.ljust(width)}  {'N':>4} {'R@5':>7} {'R@10':>7} "
              f"{'MRR@10':>7} {'nDCG@10':>8}")
    print(f"\n[{model_key}]  질문 인코더={model_key}, 청크 인코더={chunk_encoder}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["document"] == "ALL":
            print("-" * len(header))
        print(f"{r['document'].ljust(width)}  {r['n_queries']:>4} "
              f"{r['recall@5']:>7.3f} {r['recall@10']:>7.3f} "
              f"{r['mrr@10']:>7.3f} {r['ndcg@10']:>8.3f}")
    print("-" * len(header))


def print_comparison(all_rows: list[dict], model_keys: list[str]):
    """모델별 ALL 행을 나란히 비교."""
    alls = {r["model"]: r for r in all_rows if r["document"] == "ALL"}
    if len(alls) < 2:
        return
    metrics = ("recall@5", "recall@10", "mrr@10", "ndcg@10")
    base = model_keys[0]
    print("\n모델 비교 (ALL, micro)")
    print("-" * 58)
    print(f"{'metric':<12}" + "".join(f"{m:>12}" for m in model_keys) + f"{'차이':>12}")
    print("-" * 58)
    for k in metrics:
        vals = [alls[m][k] for m in model_keys if m in alls]
        diff = vals[-1] - vals[0] if len(vals) >= 2 else 0.0
        print(f"{k:<12}" + "".join(f"{v:>12.3f}" for v in vals)
              + f"{diff:>+12.3f}")
    print("-" * 58)
    print(f"(차이 = {model_keys[-1]} − {base})")


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
        print(f"[{r['id']}] rank={rank}  gold=chunk {r['gold']}")
        print(f"  Q: {r['question'][:70]}")
        print("  검색된 상위 3: "
              + ", ".join(f"{d[:12]}:{c}({s})" for d, c, s
                          in zip(r["top_docs"][:3], r["top_chunks"][:3], r["top_scores"][:3])))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="BGE-M3 / KURE-v1 청크 검색 평가")
    ap.add_argument("--models", default="bge-m3,kure-v1",
                    help=f"쉼표로 구분. 사용 가능: {', '.join(MODELS)}")
    ap.add_argument("--qa-dir", default=QA_DIR)
    ap.add_argument("--emb-dir", default=EMB_DIR)
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="결과 CSV 경로. 이미 있으면 result(1).csv 처럼 번호를 붙인다.")
    ap.add_argument("--overwrite", action="store_true", help="번호를 붙이지 않고 덮어쓴다")
    ap.add_argument("--device", default=None, help="cuda / cpu (기본: 자동)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--scope", choices=("corpus", "doc"), default="corpus",
                    help="corpus: 전체 문서 청크를 후보로 검색(기본) / doc: 해당 문서 안에서만")
    ap.add_argument("--neighbor-tolerance", type=int, default=0,
                    help="정답 청크 ±N 을 정답으로 인정 (overlap=128 보정용)")
    ap.add_argument("--reencode-chunks", action="store_true",
                    help="청크도 각 모델로 재인코딩해 공정 비교 (data/emb_cache 에 캐시)")
    ap.add_argument("--doc", default=None, help="특정 문서만 평가 (파일명 앞부분으로 매칭)")
    ap.add_argument("--dump", default=None, help="질의별 결과를 JSON 으로 저장")
    ap.add_argument("--show-misses", type=int, default=0, help="저순위 질의 N건 출력")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise SystemExit(f"알 수 없는 모델: {unknown}. 사용 가능: {list(MODELS)}")

    docs, emb_all = load_corpus(args.qa_dir, args.emb_dir)

    target = docs
    if args.doc:
        target = [d for d in docs if d["tag"].startswith(args.doc)]
        if not target:
            raise SystemExit(f"'{args.doc}' 로 시작하는 문서가 없습니다: "
                             + ", ".join(d["tag"][:12] for d in docs))

    n_q = sum(len(d["qa_pairs"]) for d in target)
    verbose = not args.quiet
    if verbose:
        print(f"문서 {len(target)}개 / 질문 {n_q}개 / 후보 청크 {emb_all.shape[0]:,}개 "
              f"(dim={emb_all.shape[1]})", file=sys.stderr)
        if not args.reencode_chunks and any(m != STORED_CHUNK_MODEL for m in model_keys):
            print("주의: 청크 벡터는 BGE-M3 로 만든 것을 그대로 사용합니다. "
                  "공정한 모델 비교는 --reencode-chunks 를 쓰세요.", file=sys.stderr)

    all_rows, dumps = [], {}
    for model_key in model_keys:
        model_id = MODELS[model_key]
        if verbose:
            print(f"\n=== {model_key} ({model_id}) 로딩 ===", file=sys.stderr)
        t0 = time.time()
        encoder = Encoder(model_id, device=args.device)
        if verbose:
            print(f"device={encoder.device}, 로딩 {time.time() - t0:.1f}s", file=sys.stderr)

        chunk_emb, chunk_encoder = chunk_matrix(
            docs, emb_all, model_key, encoder, args.reencode_chunks,
            args.batch_size, verbose)

        t0 = time.time()
        results = evaluate(all_docs=docs, target_docs=target, chunk_emb=chunk_emb,
                           encoder=encoder, scope=args.scope,
                           tolerance=args.neighbor_tolerance,
                           batch_size=args.batch_size, verbose=verbose)
        elapsed = time.time() - t0

        rows = build_rows(results, model_key, model_id, chunk_encoder,
                          args.scope, args.neighbor_tolerance)
        all_rows.extend(rows)
        dumps[model_key] = results

        print_table(rows, model_key, chunk_encoder)
        if verbose:
            print(f"검색 소요: {elapsed:.1f}s ({elapsed / n_q * 1000:.0f} ms/질의)")
        if args.show_misses:
            print_misses(results, args.show_misses)

        del encoder

    print_comparison(all_rows, model_keys)

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
        payload = {
            "config": {
                "models": {m: MODELS[m] for m in model_keys},
                "scope": args.scope,
                "neighbor_tolerance": args.neighbor_tolerance,
                "reencode_chunks": args.reencode_chunks,
                "n_candidate_chunks": int(emb_all.shape[0]),
                "k_max": K_MAX,
            },
            "summary": all_rows,
            "queries": dumps,
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"JSON 저장: {dump_path}")


if __name__ == "__main__":
    main()
