#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""청크 JSONL 임베딩 -> .npz (BGE-M3 dense+sparse / KURE-v1 dense)

chunking.py 가 만든 data/chunks/*.jsonl 을 읽어 검색용 벡터를 만든다. 청킹은
이미 끝나 있으므로 여기서는 자르지 않는다. 기존 embedding_bge_txt.py 는 .txt
전체를 받아 자체적으로 512/128 슬라이딩을 하는데, 그걸 그대로 쓰면 세션 경계를
무시하고 다시 잘라 서로 다른 대화가 한 청크에 섞인다.

    dense  : 마지막 레이어 CLS 토큰 -> L2 정규화 (1024차원). 내적 = 코사인.
    sparse : ReLU(sparse_linear(hidden_state)) 로 토큰별 lexical 가중치를 구하고,
             같은 토큰 id 가 여러 번 나오면 최댓값만 남긴다 (BGE-M3 방식).
             특수토큰(CLS/SEP/EOS/PAD/UNK)은 제외한다.

sparse 헤드(sparse_linear.pt)는 BAAI/bge-m3 저장소에만 있다. KURE-v1 에는 없어
--model kure 는 dense 만 만든다.

사용:
    python src/embed.py                          # bge, 원문+영문
    python src/embed.py --model kure
    python src/embed.py --device cuda --batch-size 32
    python src/embed.py --limit 100 --dry-run    # 소규모 확인

저장 형식 (.npz) - search.py 가 읽는 형식과 동일:
    embeddings      float32 (N, 1024)  L2 정규화된 dense 벡터
    texts           <U      (N,)       청크 원문
    chunk_index     int32   (N,)       청크 순번
    token_start     int32   (N,)       누적 토큰 위치 시작
    token_end       int32   (N,)       누적 토큰 위치 끝
    token_count     int32   (N,)       청크 토큰 수
    sparse_indices  int32              CSR 열 인덱스 (토큰 id)   [dense 전용이면 없음]
    sparse_values   float32            CSR 값 (lexical 가중치)
    sparse_indptr   int64   (N+1,)     CSR 행 포인터
    sparse_dim      int64              어휘 크기
    info            str                설정/출처를 담은 JSON 문자열
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = ROOT / "data" / "chunks"

MODELS = {
    "bge": {"id": "BAAI/bge-m3", "out": "emb/bgem3", "sparse": True},
    "kure": {"id": "nlpai-lab/KURE-v1", "out": "emb/kurev1", "sparse": False},
}
SPARSE_HEAD_REPO = "BAAI/bge-m3"


# --------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HybridEncoder:
    """dense(CLS + L2 정규화) 와 sparse(학습된 lexical 가중치) 를 한 번에 뽑는다."""

    def __init__(self, model_id: str, device: str | None = None, dtype: str = "auto",
                 max_length: int = 512, with_sparse: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.sparse_head_repo = SPARSE_HEAD_REPO if with_sparse else None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = None if dtype in (None, "auto") else getattr(torch, dtype)

        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {"dtype": torch_dtype} if torch_dtype is not None else {}
        self.model = AutoModel.from_pretrained(model_id, **kw).to(self.device).eval()
        self.vocab_size = int(self.model.config.vocab_size)

        # lexical 가중치에서 제외할 특수토큰
        self.skip_ids = {i for i in (self.tok.cls_token_id, self.tok.sep_token_id,
                                     self.tok.eos_token_id, self.tok.pad_token_id,
                                     self.tok.unk_token_id) if i is not None}

        self.sparse_linear = None
        if with_sparse:
            from huggingface_hub import hf_hub_download
            state = torch.load(hf_hub_download(SPARSE_HEAD_REPO, "sparse_linear.pt"),
                               map_location="cpu")
            lin = torch.nn.Linear(self.model.config.hidden_size, 1)
            lin.load_state_dict(state)
            self.sparse_linear = lin.to(self.device).eval()
            if torch_dtype is not None:
                self.sparse_linear = self.sparse_linear.to(torch_dtype)

    def encode(self, texts: list[str], batch_size: int = 8, progress: bool = False):
        """-> (dense (N,1024) float32, sparse [dict{token_id: weight}] 또는 None)"""
        torch = self.torch
        dense_out: list[np.ndarray] = []
        sparse_out: list[dict[int, float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            # sparse_linear 는 새로 만든 레이어라 파라미터가 grad 를 요구한다.
            # dense/sparse 계산을 모두 no_grad 안에서 처리해야 numpy() 로 뺄 수 있다.
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
                        if val > d.get(tid, 0.0):        # 같은 토큰은 최댓값만
                            d[tid] = float(val)
                    sparse_out.append(d)

            if progress:
                print(f"\r    {min(i + batch_size, len(texts)):,}/{len(texts):,}",
                      end="", file=sys.stderr)
        if progress:
            print(file=sys.stderr)

        dense = np.vstack(dense_out).astype(np.float32)
        # fp16 로 돌면 정규화가 미세하게 어긋난다. float32 에서 다시 맞춘다.
        dense /= np.clip(np.linalg.norm(dense, axis=1, keepdims=True), 1e-12, None)
        return dense, (sparse_out if self.sparse_linear is not None else None)


# --------------------------------------------------------------------------
def sparse_to_csr(dicts: list[dict[int, float]]):
    indptr = np.zeros(len(dicts) + 1, dtype=np.int64)
    indices: list[int] = []
    values: list[float] = []
    for i, d in enumerate(dicts):
        for k in sorted(d):
            indices.append(k)
            values.append(d[k])
        indptr[i + 1] = len(indices)
    return (np.asarray(indices, dtype=np.int32),
            np.asarray(values, dtype=np.float32), indptr)


def save_npz(path: Path, dense: np.ndarray, sparse, chunks: list[dict],
             info: dict, vocab_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # 청크가 독립적이라 원문 토큰 스트림이 없다. 순서를 보존하도록 누적합으로
    # 채운다 (search.py 는 이 필드를 읽지 않지만 형식을 맞춰 둔다).
    counts = np.array([c.get("n_tokens") or 0 for c in chunks], dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)

    payload = dict(
        embeddings=dense.astype(np.float32),
        texts=np.array([c["text"] for c in chunks]),
        chunk_index=np.array([c["chunk_index"] for c in chunks], dtype=np.int32),
        token_start=starts,
        token_end=(starts + counts).astype(np.int32),
        token_count=counts,
        info=np.array(json.dumps(info, ensure_ascii=False)),
    )
    if sparse is not None:
        idx, val, ptr = sparse_to_csr(sparse)
        payload.update(sparse_indices=idx, sparse_values=val, sparse_indptr=ptr,
                       sparse_dim=np.array(int(vocab_size), dtype=np.int64))
    np.savez_compressed(path, **payload)


# --------------------------------------------------------------------------
def run_file(src: Path, dst: Path, encoder, args) -> dict | None:
    if dst.exists() and not args.overwrite:
        print("  건너뜀 (이미 있음). 다시 만들려면 --overwrite")
        return None

    chunks = read_jsonl(src)
    if not chunks:
        print("  [!] 청크가 없습니다. 건너뜁니다.")
        return None

    kinds = {}
    for c in chunks:
        kinds[c.get("kind", "?")] = kinds.get(c.get("kind", "?"), 0) + 1
    counts = [c.get("n_tokens") or 0 for c in chunks]
    print(f"  청크 {len(chunks):,}개 {kinds} "
          f"(토큰 평균 {sum(counts) // max(len(counts), 1)} / 최대 {max(counts)})")

    limit = args.max_length + args.slack
    over = sum(1 for n in counts if n > limit)
    if over:
        print(f"  [!] {over}개 청크가 인코더 상한 {limit}토큰을 넘어 뒤가 잘립니다. "
              f"--slack 을 올리세요.")

    if args.limit > 0:
        chunks = chunks[:args.limit]
        print(f"  --limit {args.limit} 적용 -> {len(chunks)}개만 임베딩")

    if encoder is None:                       # dry-run
        print("  미리보기: " + chunks[0]["text"][:120].replace("\n", " ⏎ "))
        return None

    started = time.time()
    dense, sparse = encoder.encode([c["text"] for c in chunks],
                                   batch_size=args.batch_size, progress=True)
    elapsed = time.time() - started

    info = {
        "source": src.name,
        "source_field": "chunking.py",
        "model": encoder.model_id,
        "dim": int(dense.shape[1]),
        "normalized": True,
        "sparse": sparse is not None,
        "sparse_head": encoder.sparse_head_repo,
        "vocab_size": encoder.vocab_size,
        "chunk_size": args.max_length,
        "overlap": None,
        "n_chunks": len(chunks),
        "device": encoder.device,
        "elapsed_sec": round(elapsed, 1),
    }
    save_npz(dst, dense, sparse, chunks, info, encoder.vocab_size)

    nnz = sum(len(d) for d in sparse) if sparse else 0
    extra = f" / sparse 평균 {nnz / max(len(chunks), 1):.0f}토큰" if sparse else ""
    print(f"  dense {dense.shape[0]:,} x {dense.shape[1]}{extra} - "
          f"{elapsed / 60:.1f}분 ({elapsed / max(len(chunks), 1):.3f}초/청크)")
    print(f"  -> {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f}MB)")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(
        description="chunking.py 가 만든 청크 JSONL 을 임베딩한다.",
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS, type=Path,
                    help="청크 JSONL 폴더 또는 파일 (기본: data/chunks)")
    ap.add_argument("--out", default=None, type=Path,
                    help="결과 폴더 (기본: data/emb/<모델>)")
    ap.add_argument("--model", default="bge",
                    help="bge / kure, 또는 HuggingFace 모델 id")
    ap.add_argument("--max-length", type=int, default=512,
                    help="인코더 입력 상한. chunking.py 의 --chunk-size 와 맞춘다")
    ap.add_argument("--slack", type=int, default=32,
                    help="상한에 더할 여유 토큰. chunking.py 의 헤더 예산이 실제 헤더보다 "
                         "짧으면 청크가 몇 토큰 넘칠 수 있어 기본 32를 둔다 (기본 32)")
    ap.add_argument("--batch-size", type=int, default=8, help="GPU 면 32 정도까지")
    ap.add_argument("--device", default=None, help="cpu / cuda (기본: 자동 판별)")
    ap.add_argument("--dtype", default="auto", help="auto / float32 / float16")
    ap.add_argument("--no-sparse", action="store_true", help="dense 만 저장한다")
    ap.add_argument("--limit", type=int, default=0, help="파일당 앞 N개만 (테스트용)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="모델을 올리지 않는다")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    preset = MODELS.get(args.model)
    model_id = preset["id"] if preset else args.model
    out_name = preset["out"] if preset else "emb/custom"
    with_sparse = bool(preset["sparse"]) if preset else True
    if args.no_sparse:
        with_sparse = False

    src = args.chunks.resolve()
    files = [src] if src.is_file() else sorted(src.glob("*.jsonl"))
    files = [f for f in files if f.stem != "sessions"]
    if not files:
        sys.exit(f"청크 JSONL 이 없습니다: {src}\n먼저 data/chunking.py 를 실행하세요.")

    out_dir = (args.out.resolve() if args.out
               else ROOT / "data" / Path(*out_name.split("/")))

    print(f"원본    : {src}")
    print(f"출력    : {out_dir}")
    print(f"대상    : {len(files)}개 파일")
    print(f"모델    : {model_id}")
    print(f"벡터    : dense" + (" + sparse (하이브리드)" if with_sparse else " (dense 전용)"))
    if preset and not preset["sparse"] and not args.no_sparse:
        print("          [i] 이 모델에는 sparse 헤드가 없어 dense 만 만듭니다.")

    encoder = None
    if not args.dry_run:
        print("\n모델 로드 중... (최초 실행 시 다운로드 약 2.2GB)")
        encoder = HybridEncoder(model_id, device=args.device, dtype=args.dtype,
                                max_length=args.max_length + args.slack + 2,
                                with_sparse=with_sparse)
        print(f"장치    : {encoder.device}")
        if encoder.device == "cpu":
            print("          [!] CPU 라 느립니다. --limit 로 먼저 확인을 권합니다.")

    started = time.time()
    done = skipped = total = 0
    for i, f in enumerate(files, 1):
        dst = out_dir / f"{f.stem}_embeddings.npz"
        print(f"\n[{i}/{len(files)}] {f.name}")
        info = run_file(f, dst, encoder, args)
        if info is None:
            skipped += 1
        else:
            done += 1
            total += info["n_chunks"]

    if args.dry_run:
        print("\nDRY-RUN: 모델을 올리지 않고 종료합니다.")
        return
    print(f"\n전체 완료 - {(time.time() - started) / 60:.1f}분")
    print(f"  임베딩 {done}개 파일 / 청크 {total:,}개 / 건너뜀 {skipped}개")


if __name__ == "__main__":
    main()
