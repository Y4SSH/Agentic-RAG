from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfReadError

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


@dataclass(slots=True)
class IngestionSummary:
    source_directory: Path
    documents_read: int = 0
    chunks_created: int = 0
    persisted: bool = False
    collection_name: str = ""
    warnings: list[str] = field(default_factory=list)

    def log_lines(self) -> list[str]:
        lines = [
            f"Source directory: {self.source_directory}",
            f"Total documents read: {self.documents_read}",
            f"Total chunks created: {self.chunks_created}",
            f"Vector collection: {self.collection_name}",
            f"Persistence confirmed: {self.persisted}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return lines


def _build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


def _build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )


def _normalize_source(path_text: str | None, fallback_index: int) -> str:
    if not path_text:
        return f"unknown-source-{fallback_index}"
    return str(Path(path_text).resolve())


def _chunk_identifier(document: Document, chunk_index: int) -> str:
    source = _normalize_source(document.metadata.get("source"), chunk_index)
    page = str(document.metadata.get("page", "unknown"))
    payload = f"{source}|{page}|{chunk_index}|{document.page_content}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _load_documents_with_directory_loader(data_dir: Path) -> list[Document]:
    loader = PyPDFDirectoryLoader(str(data_dir))
    return loader.load()


def _load_documents_with_file_fallback(
    data_dir: Path, warnings: list[str]
) -> list[Document]:
    documents: list[Document] = []
    pdf_files = sorted(data_dir.rglob("*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            file_documents = loader.load()
            documents.extend(file_documents)
        except (PdfReadError, OSError, ValueError, RuntimeError) as exc:
            warning = f"Skipped corrupted or unreadable PDF {pdf_file.name}: {exc}"
            logger.exception(warning)
            warnings.append(warning)
    return documents


def discover_and_load_documents(
    data_dir: Path | None = None,
) -> tuple[list[Document], list[str]]:
    settings = get_settings()
    source_dir = Path(data_dir or settings.data_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    pdf_files = sorted(source_dir.rglob("*.pdf"))
    if not pdf_files:
        return [], warnings

    try:
        documents = _load_documents_with_directory_loader(source_dir)
    except (PdfReadError, OSError, ValueError, RuntimeError) as exc:
        warning = f"Directory loader failed for {source_dir}: {exc}. Falling back to file-by-file loading."
        logger.exception(warning)
        warnings.append(warning)
        documents = _load_documents_with_file_fallback(source_dir, warnings)

    return documents, warnings


def _prepare_vectorstore(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=_build_embeddings(settings),
    )


def ingest_directory(data_dir: Path | None = None) -> IngestionSummary:
    settings = get_settings()
    source_dir = Path(data_dir or settings.data_dir)
    summary = IngestionSummary(
        source_directory=source_dir, collection_name=settings.collection_name
    )

    try:
        documents, warnings = discover_and_load_documents(source_dir)
        summary.warnings.extend(warnings)
        summary.documents_read = len(documents)

        if not documents:
            summary.persisted = True
            print("No PDF documents were found to ingest.")
            for line in summary.log_lines():
                print(line)
            return summary

        splitter = _build_splitter(settings)
        chunks = splitter.split_documents(documents)
        summary.chunks_created = len(chunks)

        vectorstore = _prepare_vectorstore(settings)
        ids = [
            _chunk_identifier(document, index) for index, document in enumerate(chunks)
        ]
        vectorstore.add_documents(chunks, ids=ids)

        if hasattr(vectorstore, "persist"):
            vectorstore.persist()

        summary.persisted = True
        print("Vector store ingestion completed successfully.")
        for line in summary.log_lines():
            print(line)
        return summary
    except FileNotFoundError as exc:
        warning = f"Ingestion failed because the source directory is missing: {exc}"
        logger.exception(warning)
        summary.warnings.append(warning)
    except (PdfReadError, OSError, ValueError, RuntimeError) as exc:
        warning = f"Ingestion failed while reading or chunking PDFs: {exc}"
        logger.exception(warning)
        summary.warnings.append(warning)
    except Exception as exc:  # pragma: no cover - defensive safety net
        warning = f"Unexpected ingestion failure: {exc}"
        logger.exception(warning)
        summary.warnings.append(warning)

    for line in summary.log_lines():
        print(line)
    return summary


def main() -> None:
    ingest_directory()


if __name__ == "__main__":
    main()
