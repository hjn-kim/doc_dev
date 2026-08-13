#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TXT 문서 임베딩 - BAAI/bge-m3, dense + sparse 하이브리드 (.npz)

txt 폴더의 *.txt 를 토큰 기준으로 청킹한 뒤 두 가지 벡터를 함께 만든다.

    dense  : 마지막 레이어 CLS 토큰 -> L2 정규화 (1024차원). 내적 = 코사인.
    sparse : ReLU(sparse_linear(hidden_state)) 로 토큰별 lexical 가중치를 구하고,
             같은 토큰 id 가 여러 번 나오면 최댓값만 남긴다 (BGE-M3 방식).
             특수토큰(CLS/SEP/EOS/PAD/UNK)은 제외한다.
             질의-문서 점수는 공통 토큰의 가중치 곱의 합이다.

sparse 헤드(sparse_linear.pt)는 BAAI/bge-m3 저장소에만 있다. KURE-v1 에는 없으므로
KURE 쪽은 embedding_kure_txt.py 로 dense 만 만든다.

경로는 스크립트 위치에 고정하지 않는다. --data 를 주지 않으면 현재 작업 폴더와
스크립트 폴더 주변에서 *.txt 가 있는 폴더를 찾고, --out 을 주지 않으면 그 폴더의
형제 폴더인 emb/ 에 저장한다.

    document_dev/data/txt  -> document_dev/data/emb/bgem3   (스크립트: data/ 안)
    ~/doc_dev/txt          -> ~/doc_dev/emb/bgem3           (스크립트: 어디에 두든)

사용 예 (python -m 이 아니라 파일 경로로 실행한다):
    python data/embedding_bge_txt.py --dry-run          # 경로/청킹만 확인
    python data/embedding_bge_txt.py                    # 전체 (512/128)
    python data/embedding_bge_txt.py --device cuda --batch-size 32
    python data/embedding_bge_txt.py --no-sparse        # dense 만
    python data/embedding_bge_txt.py --data txt/ --out emb/bgem3 --overwrite

저장 형식 (.npz):
    embeddings      float32 (N, 1024)  L2 정규화된 dense 벡터
    texts           <U      (N,)       청크 원문
    chunk_index     int32   (N,)       청크 순번
    token_start     int32   (N,)       원문 토큰 기준 시작 위치
    token_end       int32   (N,)       원문 토큰 기준 끝 위치
    token_count     int32   (N,)       청크 토큰 수
    sparse_indices  int32              CSR 열 인덱스 (토큰 id)      [--no-sparse 면 없음]
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

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = "BAAI/bge-m3"
SPARSE_HEAD_REPO = "BAAI/bge-m3"          # sparse_linear.pt 를 가진 저장소
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 128
TEXT_KEY_CANDIDATES = ("text", "content", "body")


# --------------------------------------------------------------------------
# 본문 추출
# --------------------------------------------------------------------------
def read_text(path: Path) -> str:
    """UTF-8 로 읽고, 안 되면 cp949 로 한 번 더 시도한다."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949", errors="replace")


def extract_text(raw: str, field: str | None) -> tuple[str, str]:
    """(본문, 어디서 뽑았는지). .txt 지만 내용이 JSON 인 파일은 본문 key 만 골라낸다."""
    if not raw.lstrip().startswith("{"):
        return raw, "(평문)"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, "(평문)"
    if not isinstance(data, dict):
        return raw, "(평문)"
    if field:
        if not isinstance(data.get(field), str):
            sys.exit(f"'{field}' 를 문자열로 찾지 못했습니다. 있는 key: {list(data)}")
        return data[field], field
    for cand in TEXT_KEY_CANDIDATES:
        if isinstance(data.get(cand), str) and data[cand].strip():
            return data[cand], cand
    sys.exit(f"JSON 인데 text/content/body 가 없습니다. 있는 key: {list(data)}")


# --------------------------------------------------------------------------
# 토큰 기준 청킹
# --------------------------------------------------------------------------
def chunk_by_tokens(text: str, tokenizer, size: int, overlap: int) -> list[dict]:
    """토큰 size 개씩, 앞 청크와 overlap 개를 겹치도록 자른다 (stride = size - overlap)."""
    if overlap >= size:
        sys.exit(f"--overlap({overlap}) 은 --chunk-size({size}) 보다 작아야 합니다.")
    ids = tokenizer.encode(text, add_special_tokens=False)
    stride = size - overlap
    chunks: list[dict] = []
    for start in range(0, len(ids), stride):
        window = ids[start:start + size]
        if not window:
            break
        # 마지막 조각이 앞 청크에 완전히 포함되면 버린다.
        if chunks and len(window) <= overlap:
            break
        chunks.append({
            "index": len(chunks),
            "token_start": start,
            "token_end": start + len(window),
            "token_count": len(window),
            "text": tokenizer.decode(window, skip_special_tokens=True),
        })
        if start + size >= len(ids):
            break
    return chunks


# --------------------------------------------------------------------------
# 하이브리드 인코더
# --------------------------------------------------------------------------
class HybridEncoder:
    """dense(CLS + L2 정규화) 와 sparse(학습된 lexical 가중치) 를 한 번에 뽑는다."""

    def __init__(self, model_id: str, device: str | None = None, dtype: str = "auto",
                 max_length: int = 512, with_sparse: bool = True,
                 sparse_head_repo: str = SPARSE_HEAD_REPO):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.with_sparse = with_sparse
        self.sparse_head_repo = sparse_head_repo if with_sparse else None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        torch_dtype = None if dtype in (None, "auto") else getattr(torch, dtype)

        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {"dtype": torch_dtype} if torch_dtype is not None else {}
        self.model = AutoModel.from_pretrained(model_id, **kw).to(device).eval()
        self.vocab_size = int(self.model.config.vocab_size)

        # lexical 가중치에서 제외할 특수토큰
        self.skip_ids = {i for i in (self.tok.cls_token_id, self.tok.sep_token_id,
                                     self.tok.eos_token_id, self.tok.pad_token_id,
                                     self.tok.unk_token_id) if i is not None}

        self.sparse_linear = None
        if with_sparse:
            from huggingface_hub import hf_hub_download
            state = torch.load(hf_hub_download(sparse_head_repo, "sparse_linear.pt"),
                               map_location="cpu")
            lin = torch.nn.Linear(self.model.config.hidden_size, 1)
            lin.load_state_dict(state)
            self.sparse_linear = lin.to(device).eval()
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
                           max_length=self.max_length, return_tensors="pt").to(self.device)
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
                print(f"\r    {min(i + batch_size, len(texts))}/{len(texts)}",
                      end="", file=sys.stderr)
        if progress:
            print(file=sys.stderr)

        dense = np.vstack(dense_out).astype(np.float32)
        # fp16 로 돌면 정규화가 미세하게 어긋난다. float32 에서 다시 맞춘다.
        dense /= np.clip(np.linalg.norm(dense, axis=1, keepdims=True), 1e-12, None)
        return dense, (sparse_out if self.sparse_linear is not None else None)


# --------------------------------------------------------------------------
# 저장
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
    payload = dict(
        embeddings=dense.astype(np.float32),
        texts=np.array([c["text"] for c in chunks]),
        chunk_index=np.array([c["index"] for c in chunks], dtype=np.int32),
        token_start=np.array([c["token_start"] for c in chunks], dtype=np.int32),
        token_end=np.array([c["token_end"] for c in chunks], dtype=np.int32),
        token_count=np.array([c["token_count"] for c in chunks], dtype=np.int32),
        info=np.array(json.dumps(info, ensure_ascii=False)),
    )
    if sparse is not None:
        idx, val, ptr = sparse_to_csr(sparse)
        payload.update(sparse_indices=idx, sparse_values=val, sparse_indptr=ptr,
                       sparse_dim=np.array(int(vocab_size), dtype=np.int64))
    np.savez_compressed(path, **payload)


# --------------------------------------------------------------------------
# 경로 해석
# --------------------------------------------------------------------------
def data_candidates() -> list[Path]:
    cwd = Path.cwd()
    seen, out = set(), []
    for base in (cwd, SCRIPT_DIR, SCRIPT_DIR.parent):
        for sub in ("txt", "data/txt", "data", ""):
            cand = (base / sub).resolve() if sub else base.resolve()
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def find_data_dir() -> Path | None:
    for cand in data_candidates():
        if cand.is_dir() and any(cand.glob("*.txt")):
            return cand
    return None


def default_out_for(data_root: Path, name: str = "emb/bgem3") -> Path:
    """txt 폴더의 형제 폴더 emb/<모델> 을 기본 출력지로 삼는다.

        <base>/txt -> <base>/emb/bgem3   (bge)
        <base>/txt -> <base>/emb/kurev1  (kure)
    """
    base = data_root.parent if data_root.is_dir() else data_root.parent.parent
    return base.joinpath(*name.split("/"))


def collect_files(data: Path) -> list[Path]:
    if data.is_file():
        return [data]
    files = sorted(data.glob("*.txt"))
    if not files:
        files = sorted(data.rglob("*.txt"))
        if files:
            print(f"  [i] {data} 바로 아래에는 없어 하위 폴더까지 찾았습니다 ({len(files)}개)")
    return files


# --------------------------------------------------------------------------
# 파일 하나 처리
# --------------------------------------------------------------------------
def run_file(src: Path, dst: Path, tokenizer, encoder, args) -> dict | None:
    if dst.exists() and not args.overwrite:
        print("  건너뜀 (이미 있음). 다시 만들려면 --overwrite")
        return None

    text, origin = extract_text(read_text(src), args.field)
    chunks = chunk_by_tokens(text, tokenizer, args.chunk_size, args.overlap)
    if not chunks:
        print("  [!] 청크가 없습니다. 건너뜁니다.")
        return None

    counts = [c["token_count"] for c in chunks]
    print(f"  본문 {origin} {len(text):,}자 -> 청크 {len(chunks)}개 "
          f"(토큰 최소 {min(counts)} / 평균 {sum(counts) // len(counts)} / 최대 {max(counts)})")

    if args.limit > 0:
        chunks = chunks[:args.limit]
        print(f"  --limit {args.limit} 적용 -> {len(chunks)}개만 임베딩")

    if encoder is None:      # dry-run
        print("  미리보기: " + chunks[0]["text"][:120].replace("\n", " ⏎ "))
        return None

    started = time.time()
    dense, sparse = encoder.encode([c["text"] for c in chunks],
                                   batch_size=args.batch_size, progress=True)
    elapsed = time.time() - started

    info = {
        "source": src.name,
        "source_field": origin,
        "model": args.model,
        "dim": int(dense.shape[1]),
        "normalized": True,
        "sparse": sparse is not None,
        "sparse_head": encoder.sparse_head_repo,
        "vocab_size": encoder.vocab_size,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "n_chunks": len(chunks),
        "device": encoder.device,
        "elapsed_sec": round(elapsed, 1),
    }
    save_npz(dst, dense, sparse, chunks, info, encoder.vocab_size)

    nnz = sum(len(d) for d in sparse) if sparse else 0
    extra = f" / sparse 평균 {nnz / max(len(chunks), 1):.0f}토큰" if sparse else ""
    print(f"  dense {dense.shape[0]}개 x {dense.shape[1]}차원{extra} - "
          f"{elapsed / 60:.1f}분 ({elapsed / max(len(chunks), 1):.2f}초/청크)")
    print(f"  -> {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f}MB)")
    return info


# --------------------------------------------------------------------------
def build_parser(model_default: str, out_name: str, desc: str):
    p = argparse.ArgumentParser(description=desc,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default=None, type=Path,
                   help="임베딩할 TXT 파일 또는 폴더\n"
                        "(기본: 현재 폴더/스크립트 폴더 주변에서 *.txt 가 있는 곳을 자동 탐색)")
    p.add_argument("--out", default=None, type=Path,
                   help=f"결과 폴더 (기본: --data 폴더의 형제 폴더 {out_name}/)")
    p.add_argument("--field", default=None,
                   help="내용이 JSON 일 때 본문으로 쓸 key (기본: text -> content -> body)")
    p.add_argument("--model", default=model_default, help=f"임베딩 모델 (기본: {model_default})")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("--batch-size", type=int, default=8, help="GPU 면 32 정도까지 올려도 된다")
    p.add_argument("--device", default=None, help="cpu / cuda (기본: 자동 판별)")
    p.add_argument("--dtype", default="auto", help="auto / float32 / float16")
    p.add_argument("--limit", type=int, default=0, help="파일당 앞 N개 청크만 (테스트용)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="모델을 올리지 않고 청킹만 확인")
    return p


def run(args, with_sparse: bool, out_name: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if args.data is not None:
        data_root = args.data.resolve()
        if not data_root.exists():
            sys.exit(f"경로를 찾을 수 없습니다: {data_root}\n(현재 작업 폴더: {Path.cwd()})")
    else:
        found = find_data_dir()
        if found is None:
            looked = "\n".join(f"  - {c}" for c in data_candidates())
            sys.exit("*.txt 가 있는 폴더를 찾지 못했습니다. --data 로 지정하세요.\n"
                     f"(현재 작업 폴더: {Path.cwd()})\n찾아본 곳:\n{looked}")
        data_root = found

    out_dir = args.out.resolve() if args.out is not None else default_out_for(data_root, out_name)

    files = collect_files(data_root)
    if not files:
        sys.exit(f"TXT 파일이 없습니다: {data_root}")

    print(f"원본    : {data_root}")
    print(f"출력    : {out_dir}")
    print(f"대상    : {len(files)}개 파일")
    print(f"모델    : {args.model}")
    print(f"벡터    : dense" + (" + sparse (하이브리드)" if with_sparse else " (dense 전용)"))
    print(f"청킹    : {args.chunk_size}토큰 / 중복 {args.overlap}토큰 "
          f"(간격 {args.chunk_size - args.overlap})")

    print("\n토크나이저 로드 중...")
    from transformers import AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    encoder = None
    if not args.dry_run:
        print("모델 로드 중... (최초 실행 시 다운로드에 약 2.2GB)")
        encoder = HybridEncoder(args.model, device=args.device, dtype=args.dtype,
                                max_length=args.chunk_size + 2, with_sparse=with_sparse)
        print(f"장치    : {encoder.device}")
        if encoder.device == "cpu":
            print("          [!] CPU 라 느립니다. --limit 로 먼저 소규모 확인을 권합니다.")

    total_started = time.time()
    done = skipped = total_chunks = 0
    for idx, src in enumerate(files, 1):
        dst = out_dir / f"{src.stem}_embeddings.npz"
        print(f"\n[{idx}/{len(files)}] {src.name}")
        try:
            info = run_file(src, dst, tokenizer, encoder, args)
        except OSError as exc:
            print(f"  [!] 읽기 실패, 건너뜁니다: {exc}")
            skipped += 1
            continue
        if info is None:
            skipped += 1
        else:
            done += 1
            total_chunks += info["n_chunks"]

    if args.dry_run:
        print("\nDRY-RUN: 모델을 올리지 않고 종료합니다.")
        return
    elapsed = time.time() - total_started
    print(f"\n전체 완료 - {elapsed / 60:.1f}분")
    print(f"  임베딩 {done}개 파일 / 청크 {total_chunks:,}개 / 건너뜀 {skipped}개")


def main() -> None:
    p = build_parser(DEFAULT_MODEL, "emb/bgem3",
                     "txt/*.txt 를 청킹해 bge-m3 dense + sparse 임베딩을 만든다.")
    p.add_argument("--no-sparse", action="store_true",
                   help="sparse 를 만들지 않고 dense 만 저장한다")
    args = p.parse_args()
    run(args, with_sparse=not args.no_sparse, out_name="emb/bgem3")


if __name__ == "__main__":
    main()
