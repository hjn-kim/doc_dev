#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TXT 문서 임베딩 - nlpai-lab/KURE-v1, dense 전용 (.npz)

embedding_bge_txt.py 와 청킹·경로 처리·저장 형식을 그대로 공유하고,
모델과 출력 폴더만 KURE 쪽으로 바꾼다.

    <base>/txt  ->  <base>/emb/kurev1/{파일명}_embeddings.npz

dense 만 만드는 이유
    BGE-M3 의 sparse 는 저장소에 함께 있는 sparse_linear.pt (Linear 1024->1) 로
    계산한다. KURE-v1 저장소에는 model.safetensors 뿐이고 sparse 헤드가 없다.
    BGE-M3 의 헤드를 KURE 인코더 위에 얹으면 계산은 되지만 그 헤드는 BGE-M3
    인코더와 함께 학습된 것이라 KURE 에 대해서는 근거 없는 근사치가 된다.
    그래서 KURE 는 dense 만 만들고, 하이브리드 비교는 bge-bge 조합에서만 한다.

KURE-v1 은 BGE-M3 를 한국어로 파인튜닝한 모델이라 구조(XLM-RoBERTa)·차원(1024)·
풀링(CLS + L2 정규화)·토크나이저가 모두 같다. 따라서 dense 쪽 저장 형식은
BGE 결과와 완전히 호환된다.

사용 예:
    python data/embedding_kure_txt.py --dry-run
    python data/embedding_kure_txt.py --device cuda --batch-size 32
    python data/embedding_kure_txt.py --data txt/ --out emb/kurev1 --overwrite

저장 형식 (.npz): embeddings / texts / chunk_index / token_start / token_end /
                  token_count / info   (sparse_* 없음)
"""
from __future__ import annotations

import sys
from pathlib import Path

# 같은 폴더의 embedding_bge_txt.py 를 재사용한다 (어느 위치에서 실행하든 동작하도록).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_bge_txt import build_parser, run  # noqa: E402

DEFAULT_MODEL = "nlpai-lab/KURE-v1"
OUT_NAME = "emb/kurev1"


def main() -> None:
    p = build_parser(DEFAULT_MODEL, OUT_NAME,
                     "txt/*.txt 를 청킹해 KURE-v1 dense 임베딩을 만든다 (sparse 없음).")
    args = p.parse_args()
    run(args, with_sparse=False, out_name=OUT_NAME)


if __name__ == "__main__":
    main()
