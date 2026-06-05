#!/usr/bin/env python3
"""Index and query a PDF with LangChain + ChromaDB.

Examples:
  python3 rag_chroma_pdf.py ingest
  python3 rag_chroma_pdf.py query "quy định học phí là gì?"
  python3 rag_chroma_pdf.py reset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PDF = Path(
    "/root/embed/so-tay-sinh-vien-dot-5-hoc-vien-cong-nghe-buu-chinh-vien-thong-2025.pdf"
)
DEFAULT_DB_DIR = Path("/root/embed/chroma_db")
DEFAULT_ENV_FILE = Path(__file__).resolve().with_name(".env")
DEFAULT_COLLECTION = "ptit_student_handbook_2025"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
DEFAULT_LLM_BASE_URL = "http://171.226.10.154:8080/v1"
DEFAULT_LLM_MAX_TOKENS = 4096
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANKER_CACHE: dict[tuple[str, str], Any] = {}


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved chunks. Default: 5.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["vector", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval mode. hybrid combines Chroma vector search and BM25. Default: hybrid.",
    )
    parser.add_argument(
        "--vector-candidates",
        type=int,
        default=30,
        help="Vector candidates before hybrid rerank. Default: 30.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.7,
        help="BM25 weight in hybrid mode, from 0.0 to 1.0. Default: 0.7.",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help=f"Optional cross-encoder reranker model. Example: {DEFAULT_RERANKER_MODEL}",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=20,
        help="Number of retrieved candidates to rerank before returning top k. Default: 20.",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=4,
        help="Reranker batch size on CPU. Default: 4.",
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
        help="Max token length for reranker pairs. Default: 512.",
    )


def add_llm_args(
    parser: argparse.ArgumentParser,
    llm_model: str,
    llm_base_url: str,
    llm_max_tokens: int,
    llm_temperature: float,
) -> None:
    parser.add_argument(
        "--llm-model",
        default=llm_model,
        help=f"OpenAI-compatible chat model. Default: {llm_model}",
    )
    parser.add_argument(
        "--llm-base-url",
        default=llm_base_url,
        help=f"OpenAI-compatible base URL. Default: {llm_base_url}",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="API key. Defaults to OPENAI_API_KEY or LLM_API_KEY environment variable.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=llm_temperature,
        help=f"LLM temperature. Default: {llm_temperature}.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=llm_max_tokens,
        help=f"LLM max output tokens. Default: {llm_max_tokens}.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="LLM request timeout in seconds. Default: 60.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved context before the answer.",
    )


def parse_args() -> argparse.Namespace:
    load_env()
    llm_model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    llm_base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(DEFAULT_LLM_MAX_TOKENS)))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", str(DEFAULT_LLM_TEMPERATURE)))

    parser = argparse.ArgumentParser(
        description="LangChain PDF RAG helper using ChromaDB."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF file to ingest. Default: {DEFAULT_PDF}",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help=f"Persistent ChromaDB directory. Default: {DEFAULT_DB_DIR}",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection name. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Hugging Face embedding model. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--embedding-task",
        default=None,
        help="Optional SentenceTransformers task for embedding models like Jina, e.g. retrieval.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom Hugging Face model code for embeddings. Default: true.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device, for example cpu or cuda. Default: cpu.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="Text chunk size in characters. Default: 900.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Text chunk overlap in characters. Default: 150.",
    )
    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=64,
        help="Number of chunks to embed/add to Chroma per batch during ingest. Default: 64.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Load PDF, split it, and add chunks to ChromaDB.")
    subparsers.add_parser("info", help="Print ChromaDB collection info.")
    subparsers.add_parser("reset", help="Delete the local ChromaDB directory.")
    subparsers.add_parser("check-env", help="Check LLM environment variables without printing secrets.")

    query_parser = subparsers.add_parser("query", help="Similarity search against ChromaDB.")
    add_retrieval_args(query_parser)
    query_parser.add_argument("question", help="Question or search query.")
    query_parser.add_argument(
        "--json",
        action="store_true",
        help="Print retrieved chunks as JSON.",
    )

    answer_parser = subparsers.add_parser(
        "answer",
        help="Retrieve PDF context from ChromaDB and answer with ChatOpenAI.",
    )
    add_retrieval_args(answer_parser)
    answer_parser.add_argument("question", help="Question to answer with retrieved context.")
    add_llm_args(
        answer_parser,
        llm_model,
        llm_base_url,
        llm_max_tokens,
        llm_temperature,
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive RAG chat against the PDF.",
    )
    add_retrieval_args(chat_parser)
    add_llm_args(
        chat_parser,
        llm_model,
        llm_base_url,
        llm_max_tokens,
        llm_temperature,
    )
    return parser.parse_args()


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(DEFAULT_ENV_FILE)


def require_pdf(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")


def make_embeddings(args: argparse.Namespace):
    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs = {
        "device": args.device,
        "trust_remote_code": args.trust_remote_code,
    }
    encode_kwargs = {"normalize_embeddings": True}
    if args.embedding_task:
        model_kwargs["model_kwargs"] = {"default_task": args.embedding_task}
        encode_kwargs["task"] = args.embedding_task

    return HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def make_vector_store(args: argparse.Namespace):
    import chromadb
    from langchain_chroma import Chroma

    args.db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.db_dir))
    return Chroma(
        client=client,
        collection_name=args.collection,
        embedding_function=make_embeddings(args),
    )


def get_chroma_collection(args: argparse.Namespace):
    import chromadb

    client = chromadb.PersistentClient(path=str(args.db_dir))
    return client.get_or_create_collection(args.collection)


def load_indexed_documents(args: argparse.Namespace) -> list:
    from langchain_core.documents import Document

    collection = get_chroma_collection(args)
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    docs = []
    for content, metadata in zip(documents, metadatas):
        if content:
            docs.append(Document(page_content=content, metadata=metadata or {}))
    return docs


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def bm25_rank(question: str, docs: list, k: int) -> list[tuple]:
    query_terms = tokenize_text(question)
    if not query_terms or not docs:
        return []

    tokenized_docs = [tokenize_text(doc.page_content) for doc in docs]
    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    k1 = 1.5
    b = 0.75
    ranked = []
    for doc, tokens in zip(docs, tokenized_docs):
        term_freq = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
            score += idf * (freq * (k1 + 1)) / denom
        if score > 0:
            ranked.append((doc, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def doc_key(doc) -> tuple:
    metadata = doc.metadata or {}
    return (
        metadata.get("source", ""),
        metadata.get("page", ""),
        metadata.get("chunk_index", ""),
        hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:16],
    )


def normalize_scores(scores: dict, higher_is_better: bool) -> dict:
    if not scores:
        return {}
    values = list(scores.values())
    if not higher_is_better:
        values = [-value for value in values]
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {key: 1.0 for key in scores}

    normalized = {}
    for key, value in scores.items():
        comparable = value if higher_is_better else -value
        normalized[key] = (comparable - min_value) / (max_value - min_value)
    return normalized


def retrieve_base(
    args: argparse.Namespace,
    vector_store,
    question: str,
    k: int,
) -> list[tuple]:
    if args.search_mode == "vector":
        return vector_store.similarity_search_with_score(question, k=k)

    docs = load_indexed_documents(args)
    if args.search_mode == "bm25":
        return bm25_rank(question, docs, k)

    vector_k = max(k, args.vector_candidates)
    vector_results = vector_store.similarity_search_with_score(question, k=vector_k)
    bm25_results = bm25_rank(question, docs, vector_k)

    docs_by_key = {}
    vector_distances = {}
    bm25_scores = {}

    for doc, distance in vector_results:
        key = doc_key(doc)
        docs_by_key[key] = doc
        vector_distances[key] = distance

    for doc, score in bm25_results:
        key = doc_key(doc)
        docs_by_key[key] = doc
        bm25_scores[key] = score

    vector_norm = normalize_scores(vector_distances, higher_is_better=False)
    bm25_norm = normalize_scores(bm25_scores, higher_is_better=True)
    bm25_weight = min(max(args.bm25_weight, 0.0), 1.0)
    vector_weight = 1.0 - bm25_weight

    ranked = []
    for key, doc in docs_by_key.items():
        score = vector_weight * vector_norm.get(key, 0.0) + bm25_weight * bm25_norm.get(key, 0.0)
        doc.metadata["retrieval_mode"] = "hybrid"
        doc.metadata["vector_score"] = vector_norm.get(key, 0.0)
        doc.metadata["bm25_score"] = bm25_norm.get(key, 0.0)
        ranked.append((doc, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def load_reranker(model_name: str, device: str):
    cache_key = (model_name, device)
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    reranker = {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }
    _RERANKER_CACHE[cache_key] = reranker
    return reranker


def rerank_results(
    args: argparse.Namespace,
    question: str,
    results: list[tuple],
) -> list[tuple]:
    if not args.reranker_model or not results:
        return results[: args.k]

    reranker = load_reranker(args.reranker_model, args.device)
    torch = reranker["torch"]
    tokenizer = reranker["tokenizer"]
    model = reranker["model"]
    device = reranker["device"]

    scored = []
    for start in range(0, len(results), args.reranker_batch_size):
        batch = results[start : start + args.reranker_batch_size]
        pairs = [[question, doc.page_content] for doc, _score in batch]
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=args.reranker_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            scores = logits.view(-1).float().cpu().tolist()
        for (doc, base_score), rerank_score in zip(batch, scores):
            doc.metadata["reranker_model"] = args.reranker_model
            doc.metadata["base_retrieval_score"] = base_score
            doc.metadata["reranker_score"] = rerank_score
            scored.append((doc, rerank_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: args.k]


def retrieve(args: argparse.Namespace, vector_store, question: str) -> list[tuple]:
    candidate_k = args.k
    if args.reranker_model:
        candidate_k = max(args.k, args.rerank_candidates)
    candidates = retrieve_base(args, vector_store, question, candidate_k)
    return rerank_results(args, question, candidates)


def load_pdf_pages(pdf_path: Path):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(pdf_path), mode="page")
    docs = loader.load()
    return [doc for doc in docs if doc.page_content.strip()]


def split_documents(docs: list, chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["source_file"] = Path(chunk.metadata.get("source", "")).name
    return chunks


def stable_chunk_ids(chunks: list) -> list[str]:
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        chunk_index = chunk.metadata.get("chunk_index", "")
        digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:16]
        ids.append(f"{source}:{page}:{chunk_index}:{digest}")
    return ids


def ingest(args: argparse.Namespace) -> None:
    require_pdf(args.pdf)
    start = time.perf_counter()

    print(f"PDF: {args.pdf}")
    print(f"ChromaDB: {args.db_dir}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {args.embedding_model}")

    docs = load_pdf_pages(args.pdf)
    if not docs:
        raise RuntimeError("No text extracted from PDF.")
    print(f"Loaded pages with text: {len(docs)}")

    chunks = split_documents(docs, args.chunk_size, args.chunk_overlap)
    ids = stable_chunk_ids(chunks)
    print(f"Chunks: {len(chunks)}")

    vector_store = make_vector_store(args)
    for start_index in range(0, len(chunks), args.add_batch_size):
        end_index = min(start_index + args.add_batch_size, len(chunks))
        vector_store.add_documents(
            documents=chunks[start_index:end_index],
            ids=ids[start_index:end_index],
        )
        print(f"Indexed chunks: {end_index}/{len(chunks)}", flush=True)

    elapsed = time.perf_counter() - start
    print(f"Done. Indexed {len(chunks)} chunks in {elapsed:.2f}s")


def query(args: argparse.Namespace) -> None:
    vector_store = make_vector_store(args)
    results = retrieve(args, vector_store, args.question)

    if args.json:
        payload = []
        for doc, score in results:
            payload.append(
                {
                    "score": score,
                    "reranker_score": doc.metadata.get("reranker_score"),
                    "base_retrieval_score": doc.metadata.get("base_retrieval_score"),
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content": doc.page_content,
                }
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Question: {args.question}")
    print(f"Top {len(results)} chunks:")
    for rank, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page")
        page_label = doc.metadata.get("page_label", page)
        chunk_index = doc.metadata.get("chunk_index")
        source = doc.metadata.get("source_file") or Path(doc.metadata.get("source", "")).name
        content = " ".join(doc.page_content.split())
        score_name = "reranker_score" if doc.metadata.get("reranker_score") is not None else "score"
        print()
        print(f"[{rank}] {score_name}={score:.4f} page={page_label} chunk={chunk_index} source={source}")
        print(content[:1200])


def make_llm(args: argparse.Namespace):
    from langchain_openai import ChatOpenAI

    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY/LLM_API_KEY or pass --llm-api-key."
        )

    return ChatOpenAI(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.request_timeout,
    )


def format_context(results: list[tuple]) -> str:
    blocks = []
    for rank, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        chunk_index = doc.metadata.get("chunk_index", "?")
        source = doc.metadata.get("source_file") or Path(doc.metadata.get("source", "")).name
        content = " ".join(doc.page_content.split())
        blocks.append(
            f"[{rank}] source={source} page={page} chunk={chunk_index} score={score:.4f}\n"
            f"{content}"
        )
    return "\n\n".join(blocks)


def make_answer_prompt(question: str, context: str) -> str:
    return f"""Bạn là trợ lý RAG chuyên trả lời dựa trên tài liệu sổ tay sinh viên.

NGUYÊN TẮC BẮT BUỘC:
1. Chỉ dùng thông tin trong CONTEXT. Không dùng kiến thức ngoài, không tự suy đoán.
2. Đọc toàn bộ CONTEXT trước khi trả lời; ưu tiên đoạn có từ khóa, điều kiện, thời hạn, số liệu, biểu mẫu hoặc quy định khớp trực tiếp với câu hỏi.
3. Nếu CONTEXT chỉ chứa một phần thông tin, hãy trả lời phần có căn cứ và nói rõ phần nào chưa thấy trong tài liệu đã truy xuất.
4. Với câu hỏi về điều kiện/quy trình/thời hạn/biểu mẫu, phải liệt kê đủ các ý tìm thấy, giữ nguyên số ngày, tỷ lệ %, tên đơn vị, tên biểu mẫu nếu có.
5. Với bảng hoặc biểu mẫu, phân biệt từng bảng/trang/mục. Chỉ nói một trường/mục xuất hiện trong một bảng nếu chính tên trường đó xuất hiện trong đoạn context của bảng đó. Không được suy rộng rằng một mục "áp dụng thống nhất" cho mọi bảng nếu từng bảng không cùng hiển thị mục đó.
   - Nếu bảng A có "Điểm rèn luyện" nhưng bảng B không hiển thị trường này, phải nói rõ "chỉ thấy ở bảng A; chưa thấy ở bảng B".
6. Nếu các đoạn context có vẻ mâu thuẫn hoặc không đủ để chọn một kết luận duy nhất, hãy nói rõ thay vì chọn bừa.
7. Mỗi ý factual phải có nguồn dạng [page X, chunk Y]. Không nhắc score.

CÁCH TRẢ LỜI:
- Bắt đầu bằng câu trả lời trực tiếp 1-2 câu.
- Nếu có nhiều điều kiện/bước, dùng bullet ngắn.
- Kết thúc bằng dòng "Nguồn:" liệt kê các [page X, chunk Y] đã dùng.
- Không trả lời dài lan man; không trích nguyên văn quá dài.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def answer_question(
    args: argparse.Namespace,
    vector_store,
    llm,
    question: str,
    show_context: bool,
) -> str:
    results = retrieve(args, vector_store, question)
    context = format_context(results)

    if show_context:
        print("Retrieved context:")
        print(context)
        print()

    response = llm.invoke(make_answer_prompt(question, context))
    return response.content


def answer(args: argparse.Namespace) -> None:
    vector_store = make_vector_store(args)
    llm = make_llm(args)
    print(answer_question(args, vector_store, llm, args.question, args.show_context))


def chat(args: argparse.Namespace) -> None:
    vector_store = make_vector_store(args)
    llm = make_llm(args)

    print("RAG chat ready. Type a question, or 'exit'/'quit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", ":q"}:
            break

        try:
            reply = answer_question(args, vector_store, llm, question, args.show_context)
        except Exception as error:
            print(f"Error: {error}")
            continue
        print(f"\nAssistant: {reply}")


def info(args: argparse.Namespace) -> None:
    import chromadb

    if not args.db_dir.exists():
        print(f"ChromaDB directory does not exist: {args.db_dir}")
        return
    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_or_create_collection(args.collection)
    print(f"ChromaDB: {args.db_dir}")
    print(f"Collection: {args.collection}")
    print(f"Count: {collection.count()}")


def reset(args: argparse.Namespace) -> None:
    if args.db_dir.exists():
        shutil.rmtree(args.db_dir)
        print(f"Deleted {args.db_dir}")
    else:
        print(f"Nothing to delete: {args.db_dir}")


def check_env(args: argparse.Namespace) -> None:
    del args
    load_env()
    print(f"Env file: {DEFAULT_ENV_FILE}")
    print(f"Env file exists: {DEFAULT_ENV_FILE.exists()}")
    for key in [
        "OPENAI_API_KEY",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
    ]:
        value = os.getenv(key)
        status = "set" if value and value.strip() else "missing/empty"
        length = len(value.strip()) if value else 0
        print(f"{key}: {status} len={length}")


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.command == "ingest":
        ingest(args)
    elif args.command == "query":
        query(args)
    elif args.command == "answer":
        answer(args)
    elif args.command == "chat":
        chat(args)
    elif args.command == "info":
        info(args)
    elif args.command == "reset":
        reset(args)
    elif args.command == "check-env":
        check_env(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
