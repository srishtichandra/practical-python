#!/usr/bin/env python3
"""
Pipeline for converting natural language instructions into Linear Temporal Logic (LTL)
formulas using retrieval-augmented generation (RAG).

Steps performed:
1. Load an ontology file containing task/environmental facts.
2. Chunk and embed the ontology with the selected embedding model.
3. Index the embeddings with FAISS and retrieve the top-k most relevant chunks.
4. Send the retrieved context plus the natural language instruction to an LLM prompt
   that requests only an LTL formula.
5. Print the raw LTL formula and a planner-ready JSON payload that can be consumed by
   tools such as Spot or LTLf2DFA.

Usage example:
    python ltl_rag_pipeline.py --ontology ontology.txt --instruction "Always avoid obstacles."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

from langchain.docstore.document import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - backwards compatability
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as exc:  # pragma: no cover - defer failure until actually used
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    _openai_import_error = exc
else:
    _openai_import_error = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.chat_models import ChatOllama
    from langchain_community.vectorstores import FAISS
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "langchain-community package is required. Install with `pip install langchain-community`."
    ) from exc


def load_ontology_documents(
    ontology_path: Path, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Load an ontology text file and split it into overlapping chunks for retrieval.
    """
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found at {ontology_path}")

    text = ontology_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = splitter.create_documents([text], metadatas=[{"source": str(ontology_path)}])
    return documents


def get_embedding_model(provider: str, model_name: str | None) -> object:
    """
    Create an embedding model instance for the given provider.
    """
    provider = provider.lower()
    if provider == "openai":
        if OpenAIEmbeddings is None:  # pragma: no cover - handled at runtime
            raise RuntimeError(
                "OpenAI embeddings requested but cannot import langchain_openai. "
                "Install langchain-openai and set OPENAI_API_KEY."
            ) from _openai_import_error
        return OpenAIEmbeddings(model=model_name or "text-embedding-3-small")

    if provider in {"hf", "huggingface"}:
        return HuggingFaceEmbeddings(model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2")

    raise ValueError(f"Unsupported embedding provider '{provider}'. Choose 'openai' or 'hf'.")


def get_chat_model(provider: str, model_name: str, temperature: float):
    """
    Create an LLM chat model for the given provider.
    """
    provider = provider.lower()
    if provider == "openai":
        if ChatOpenAI is None:  # pragma: no cover
            raise RuntimeError(
                "OpenAI LLM requested but langchain-openai is unavailable. "
                "Install langchain-openai and set OPENAI_API_KEY."
            ) from _openai_import_error
        return ChatOpenAI(model=model_name, temperature=temperature)

    if provider == "ollama":
        return ChatOllama(model=model_name, temperature=temperature)

    raise ValueError(f"Unsupported LLM provider '{provider}'. Choose 'openai' or 'ollama'.")


def build_rag_chain(retriever, llm):
    """
    Compose the LangChain RAG pipeline that formats retrieved documents, injects them
    into the prompt, queries the LLM, and parses the response to a plain string.
    """

    def format_docs(docs: Sequence[Document]) -> str:
        return "\n\n".join(doc.page_content.strip() for doc in docs)

    prompt = PromptTemplate(
        input_variables=["instruction", "context"],
        template=(
            "You are an expert in formal methods tasked with expressing tasks in Linear Temporal Logic (LTL).\n"
            "Ground your reasoning in the provided ontology facts.\n\n"
            "Ontology facts:\n"
            "{context}\n\n"
            "Instruction: {instruction}\n\n"
            "Return ONLY the LTL formula suitable for consumption by an automated planner. "
            "Do not include prose, explanations, or code fences."
        ),
    )

    rag_chain = (
        RunnableParallel(
            instruction=RunnablePassthrough(),
            context=retriever | RunnableLambda(format_docs),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def generate_ltl_formula(
    instruction: str,
    ontology_path: Path,
    embedding_provider: str,
    embedding_model: str | None,
    llm_provider: str,
    llm_model: str,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    temperature: float,
):
    """
    Execute the full RAG pipeline and return the formula along with retrieved context.
    """
    documents = load_ontology_documents(ontology_path, chunk_size, chunk_overlap)
    embeddings = get_embedding_model(embedding_provider, embedding_model)
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    llm = get_chat_model(llm_provider, llm_model, temperature=temperature)
    rag_chain = build_rag_chain(retriever, llm)

    formula = rag_chain.invoke(instruction).strip()
    retrieved_docs = retriever.invoke(instruction)
    retrieved_context = [doc.page_content for doc in retrieved_docs]

    return formula, retrieved_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert natural language instructions to LTL formulas using a LangChain RAG pipeline."
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        required=True,
        help="Path to ontology text file containing task/environment facts.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Natural language instruction to translate into LTL.",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["openai", "hf", "huggingface"],
        default="hf",
        help="Embedding provider to use for indexing.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model identifier (defaults to provider-specific sensible choice).",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider to use for generating LTL.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model identifier (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of ontology chunks to retrieve for each instruction.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Character length for ontology text chunks.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Overlap characters between ontology chunks.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        choices=["spot", "ltlf2dfa"],
        default="spot",
        help="Downstream planner target for payload metadata.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        formula, retrieved_context = generate_ltl_formula(
            instruction=args.instruction,
            ontology_path=args.ontology,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            temperature=args.temperature,
        )
    except Exception as exc:  # pragma: no cover - runtime error surface
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Requirement 4: print the raw LTL formula
    print(formula)

    # Planner-friendly payload (e.g., can be piped into Spot/LTLf2DFA tooling scripts).
    planner_payload = {
        "ltl_formula": formula,
        "planner_target": args.planner,
        "retrieved_context": retrieved_context,
        "metadata": {
            "ontology_path": str(args.ontology),
            "llm_model": args.llm_model,
            "embedding_provider": args.embedding_provider,
        },
    }
    print(json.dumps(planner_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
