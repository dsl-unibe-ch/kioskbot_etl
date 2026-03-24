"""Step 4 — Index: build an Azure AI Search index from processed_data.xlsx."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from tqdm import tqdm

#NoQA: F401 Used to prevent circular import of RunContext from pipeline.py during static type checking
if TYPE_CHECKING:
    from src.etl_crawler.config import AppSettings
    from src.etl_crawler.pipeline import RunContext

logger = logging.getLogger(__name__)

HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_REQUEST_TIMEOUT = 408
HTTP_STATUS_TOO_MANY_REQUESTS = 429
HTTP_STATUS_INTERNAL_SERVER_ERROR = 500
HTTP_STATUS_BAD_GATEWAY = 502
HTTP_STATUS_SERVICE_UNAVAILABLE = 503
HTTP_STATUS_GATEWAY_TIMEOUT = 504

INDEX_CREATE_MAX_ATTEMPTS = 4
INDEX_CREATE_RETRY_SECONDS = 5
INDEX_COUNT_CHECK_ATTEMPTS = 24
INDEX_COUNT_CHECK_SLEEP_SECONDS = 5
INDEX_EXPORT_PAGE_SIZE = 1000



def german2english(text: str, settings: AppSettings) -> str:
    """Translate German text to English."""
    translation_system_prompt = """Translate in English the following text.

    ** Important: **
    - do not translate links, email addresses, names of people and places.
    - keep the format of the original text (e.g. if there are bullet points, keep them).

    Text: {input}
    The translated text is: {{output}}
    """

    translation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", translation_system_prompt),
            ("human", "{input}"),
        ]
    )

    chat_client = AzureChatOpenAI(
        azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_version=settings.AZURE_OPENAI_CHAT_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_PRIMARY_KEY,
    )

    translation_chain = translation_prompt | chat_client
    return translation_chain.invoke({"input": text}).content # type: ignore

class AzureEmbeddingWrapper:
    """Adapter so SemanticChunker can call our Azure embedding client."""

    def __init__(self, embedding_client: AzureOpenAI, deployment: str) -> None:
        self._client = embedding_client
        self._deployment = deployment

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self._deployment)
        return resp.data[0].embedding


def _generate_chunk_title(chunk: str, settings: AppSettings) -> str:
    """Generate a concise and informative title for a document chunk."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the following document chunk, generate a concise and informative "
         "title that summarizes its main topic or purpose.\n\n"
         "Document chunk: {input}\nThe title generated is: {{output}}"),
        ("human", "{input}"),
    ])
    chat = AzureChatOpenAI(
        azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_version=settings.AZURE_OPENAI_CHAT_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_PRIMARY_KEY,
    )
    try:
        return (prompt | chat).invoke({"input": chunk}).content
    except Exception:
        return "No title generated"


def _build_index_schema(index_name: str, settings: AppSettings) -> SearchIndex:
    vectorizer = AzureOpenAIVectorizer(
        vectorizer_name="myTextEmbedding3LargeVectorizer",
        parameters=AzureOpenAIVectorizerParameters(
            resource_url=settings.AZURE_OPENAI_VECTORIZER_ENDPOINT,
            deployment_name=settings.AZURE_OPENAI_SEARCH_EMBEDDING_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_PRIMARY_KEY,
            model_name=settings.AZURE_OPENAI_SEARCH_EMBEDDING_DEPLOYMENT,
        ),
    )
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnswAlgorithm",
                parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"},
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myVectorProfile",
                algorithm_configuration_name="myHnswAlgorithm",
                vectorizer_name="myTextEmbedding3LargeVectorizer",
            )
        ],
        vectorizers=[vectorizer],
    )
    semantic_config = SemanticConfiguration(
        name="mySemanticConfig",
        prioritized_fields={
            "title_field": SemanticField(field_name="Title_Chunk"),
            "content_fields": [
                SemanticField(field_name="chunk"),
                SemanticField(field_name="Example_Questions"),
            ],
            "keywords_fields": [SemanticField(field_name="Keyword")],
        },
    )
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="DocumentID", type=SearchFieldDataType.String),
        SearchableField(name="Link", type=SearchFieldDataType.String),
        SearchableField(name="Title", type=SearchFieldDataType.String),
        SearchableField(name="Title_Chunk", type=SearchFieldDataType.String),
        SearchableField(name="Category", type=SearchFieldDataType.String),
        SearchableField(name="Local_Path", type=SearchFieldDataType.String),
        SearchableField(name="Local_Path_PDF", type=SearchFieldDataType.String),
        SearchableField(name="Date_Last_Modified", type=SearchFieldDataType.String),
        SearchableField(name="Data_Gathered_On", type=SearchFieldDataType.DateTimeOffset),
        SearchableField(name="chunk", type=SearchFieldDataType.String),
        SearchableField(name="chunk_translated", type=SearchFieldDataType.String),
        SearchableField(name="Keyword", type=SearchFieldDataType.String),
        SearchField(
            name="Example_Questions",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            facetable=False,
            filterable=False,
        ),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=3072,
            vector_search_profile_name="myVectorProfile",
            hidden=False,
            stored=True,
        ),
    ]
    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )


def _find_previous_run_json(data_dir: Path, env_label: str) -> Path | None:
    """Find the JSON export from the most recent previous run for this customer."""
    customer_dir = data_dir.parent
    if not customer_dir.exists():
        return None

    json_filename = f"processed_data_azure_semantic_search_{env_label}.json"
    candidates: list[Path] = []
    for entry in customer_dir.iterdir():
        if entry.is_dir() and entry != data_dir:
            json_path = entry / json_filename
            if json_path.exists():
                candidates.append(json_path)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _generate_change_summary(
    new_documents: list[dict],
    previous_json_path: Path | None,
) -> dict:
    """Compare new documents against a previous run's export."""
    summary: dict = {"new_chunk_count": len(new_documents)}

    if not previous_json_path or not previous_json_path.exists():
        summary["status"] = "first_run"
        return summary

    with previous_json_path.open(encoding="utf-8") as f:
        old_documents = json.load(f)

    old_links = {d["Link"] for d in old_documents}
    new_links = {d["Link"] for d in new_documents}

    summary.update({
        "status": "update",
        "previous_run": str(previous_json_path.parent.name),
        "old_chunk_count": len(old_documents),
        "added_sources": sorted(new_links - old_links),
        "removed_sources": sorted(old_links - new_links),
        "unchanged_source_count": len(old_links & new_links),
    })
    return summary


def _is_transient_http_error(exc: HttpResponseError) -> bool:
    """Return True if error is likely transient and retryable."""
    return exc.status_code in {
        HTTP_STATUS_REQUEST_TIMEOUT,
        HTTP_STATUS_TOO_MANY_REQUESTS,
        HTTP_STATUS_INTERNAL_SERVER_ERROR,
        HTTP_STATUS_BAD_GATEWAY,
        HTTP_STATUS_SERVICE_UNAVAILABLE,
        HTTP_STATUS_GATEWAY_TIMEOUT,
    }


def _create_index_with_retry(
    index_client: SearchIndexClient, index_schema: SearchIndex, index_name: str
) -> None:
    """Create index with retry for transient Azure Search errors."""
    for attempt in range(1, INDEX_CREATE_MAX_ATTEMPTS + 1):
        try:
            index_client.create_index(index_schema)
            logger.info("Index '%s' created.", index_name)
            return
        except HttpResponseError as exc:
            is_last_attempt = attempt == INDEX_CREATE_MAX_ATTEMPTS
            if _is_transient_http_error(exc) and not is_last_attempt:
                logger.warning(
                    "Transient error creating index '%s' (attempt %d/%d): %s. Retrying in %ds...",
                    index_name,
                    attempt,
                    INDEX_CREATE_MAX_ATTEMPTS,
                    exc,
                    INDEX_CREATE_RETRY_SECONDS,
                )
                time.sleep(INDEX_CREATE_RETRY_SECONDS)
                continue
            logger.exception(
                "Failed to create index '%s' after %d attempt(s).",
                index_name,
                attempt,
            )
            raise


def _assert_index_not_empty(
    index_client: SearchIndexClient,
    search_client: SearchClient,
    index_name: str,
) -> None:
    """Poll multiple count endpoints and fail if index remains empty."""
    doc_count_stats = 0
    doc_count_search = 0
    for attempt in range(1, INDEX_COUNT_CHECK_ATTEMPTS + 1):
        stats = index_client.get_index_statistics(index_name)
        doc_count_stats = int(stats.get("document_count", stats.get("documentCount", 0)))
        try:
            doc_count_search = int(search_client.get_document_count())
        except HttpResponseError:
            doc_count_search = 0

        if max(doc_count_stats, doc_count_search) > 0:
            logger.info(
                "Verified index '%s' is populated (stats=%d, search=%d).",
                index_name,
                doc_count_stats,
                doc_count_search,
            )
            return
        if attempt < INDEX_COUNT_CHECK_ATTEMPTS:
            logger.warning(
                "Index '%s' still empty after upload (attempt %d/%d, stats=%d, search=%d). "
                "Rechecking in %ds...",
                index_name,
                attempt,
                INDEX_COUNT_CHECK_ATTEMPTS,
                doc_count_stats,
                doc_count_search,
                INDEX_COUNT_CHECK_SLEEP_SECONDS,
            )
            time.sleep(INDEX_COUNT_CHECK_SLEEP_SECONDS)

    raise RuntimeError(
        f"ALARM: index '{index_name}' is empty after upload verification "
        f"(stats={doc_count_stats}, search={doc_count_search})."
    )


def _index_exists(index_client: SearchIndexClient, index_name: str) -> bool:
    """Return True if the index currently exists."""
    try:
        index_client.get_index(index_name)
        return True
    except HttpResponseError as exc:
        if exc.status_code == HTTP_STATUS_NOT_FOUND:
            return False
        raise


def _backup_existing_index(
    index_name: str,
    settings: AppSettings,
    output_dir: Path,
) -> Path | None:
    """Export current index documents to JSONL before replacement."""
    index_client = SearchIndexClient(
        settings.AZURE_SEARCH_ENDPOINT,
        AzureKeyCredential(settings.AZURE_SEARCH_SERVICE_PRIMARY_ADMIN_KEY),
    )
    if not _index_exists(index_client, index_name):
        logger.info("No existing index '%s' found. Skipping backup.", index_name)
        return None

    search_client = SearchClient(
        settings.AZURE_SEARCH_ENDPOINT,
        index_name,
        AzureKeyCredential(settings.AZURE_SEARCH_SERVICE_PRIMARY_ADMIN_KEY),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_path = output_dir / f"{index_name}_backup_before_replace_{int(time.time())}.jsonl"

    count = 0
    results = search_client.search(search_text="*", top=INDEX_EXPORT_PAGE_SIZE)
    with backup_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(dict(result), ensure_ascii=False) + "\n")
            count += 1

    logger.info("Backed up %d documents from '%s' to %s", count, index_name, backup_path)
    return backup_path


def run_etl(xlsx_path: Path, index_name: str, settings: AppSettings, output_dir: Path) -> dict:
    """Core ETL: prepare documents, generate change summary, then replace the index."""

    embedding_client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_SEARCH_EMBEDDING_DEPLOYMENT,
        api_version=settings.AZURE_OPENAI_SEARCH_EMBEDDING_API_VERSION,
        api_key=settings.AZURE_OPENAI_PRIMARY_KEY,
    )
    wrapper = AzureEmbeddingWrapper(embedding_client, settings.AZURE_OPENAI_SEARCH_EMBEDDING_DEPLOYMENT)
    text_splitter = SemanticChunker(wrapper)

    # --- Phase 1: prepare all documents (slow, may fail) ---
    df = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    documents: list[dict] = []
    chunk_id = 0
    logger.info("Processing %d rows from %s", len(df), xlsx_path.name)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows to create chunks"):
        chunks = text_splitter.create_documents([row["text"]])
        for chunk in chunks:
            content = chunk.page_content.strip()
            if not content:
                continue
            title_chunk = _generate_chunk_title(content, settings)
            vector = wrapper.embed_query(content)
            try:
                translated = german2english(content, settings)
            except Exception:
                translated = "No translation generated"
            documents.append({
                "chunk_id": f"doc_{chunk_id}",
                "DocumentID": row["DocumentID"],
                "Link": row["Link"],
                "Title": row["Title"],
                "Title_Chunk": title_chunk,
                "Category": row["Category"],
                "Local_Path": row["Local_Path"],
                "Local_Path_PDF": row["Local_Path_PDF"],
                "Date_Last_Modified": row["Date_Last_Modified"],
                "Data_Gathered_On": row["Data_Gathered_On"],
                "chunk": content,
                "chunk_translated": translated,
                "Keyword": row["Keyword"],
                "Example_Questions": row["Example_Questions"].split(","),
                "text_vector": vector,
            })
            chunk_id += 1

    logger.info("Prepared %d chunks. Saving local copy...", len(documents))
    env_label = settings.ENV or "dev"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = output_dir / f"processed_data_azure_semantic_search_{env_label}.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    # --- Phase 1.5: compare against previous run ---
    previous_json = _find_previous_run_json(output_dir, env_label)
    change_summary = _generate_change_summary(documents, previous_json)

    report_path = output_dir / "index_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(change_summary, f, indent=2, ensure_ascii=False)

    if change_summary.get("status") == "first_run":
        logger.info("First run — no previous index to compare against. %d new chunks.", len(documents))
    else:
        logger.info(
            "Change summary vs run %s: %d -> %d chunks, +%d sources, -%d sources, %d unchanged",
            change_summary.get("previous_run", "?"),
            change_summary.get("old_chunk_count", 0),
            change_summary["new_chunk_count"],
            len(change_summary.get("added_sources", [])),
            len(change_summary.get("removed_sources", [])),
            change_summary.get("unchanged_source_count", 0),
        )
        if change_summary.get("added_sources"):
            for src in change_summary["added_sources"]:
                logger.info("  + %s", src)
        if change_summary.get("removed_sources"):
            for src in change_summary["removed_sources"]:
                logger.info("  - %s", src)

    if not documents:
        raise RuntimeError(
            f"ALARM: refusing to replace index '{index_name}' because prepared document list is empty."
        )

    # --- Phase 2: replace index and upload (only after all docs are ready) ---
    backup_path = _backup_existing_index(index_name, settings, output_dir)
    if backup_path:
        logger.info("Created index backup before delete: %s", backup_path)

    index_client = SearchIndexClient(
        settings.AZURE_SEARCH_ENDPOINT,
        AzureKeyCredential(settings.AZURE_SEARCH_SERVICE_PRIMARY_ADMIN_KEY),
    )
    index_schema = _build_index_schema(index_name, settings)

    try:
        index_client.delete_index(index_name)
        logger.info("Deleted existing index '%s'.", index_name)
    except HttpResponseError as e:
        if e.status_code != HTTP_STATUS_NOT_FOUND:
            raise

    _create_index_with_retry(index_client, index_schema, index_name)

    search_client = SearchClient(
        settings.AZURE_SEARCH_ENDPOINT,
        index_name,
        AzureKeyCredential(settings.AZURE_SEARCH_SERVICE_PRIMARY_ADMIN_KEY),
    )
    try:
        upload_result = search_client.upload_documents(documents)
        failed_uploads = [res for res in upload_result if not res.succeeded]
        if failed_uploads:
            sample_errors = [
                f"{res.key}: {getattr(res, 'error_message', 'unknown upload error')}"
                for res in failed_uploads[:5]
            ]
            raise RuntimeError(
                f"ALARM: {len(failed_uploads)} document uploads failed for index '{index_name}'. "
                f"Sample errors: {' | '.join(sample_errors)}"
            )
        logger.info("Uploaded %d documents to '%s'", len(documents), index_name)
        for res in upload_result:
            logger.debug("  %s: %s", res.key, res.succeeded)
    except (HttpResponseError, ValueError, TypeError):
        logger.exception("Error uploading documents")
        raise

    _assert_index_not_empty(index_client, search_client, index_name)

    return {"index_name": index_name, "chunk_count": len(documents)}


# ---------------------------------------------------------------------------
# Step entry-point
# ---------------------------------------------------------------------------


@dataclass
class IndexResult:
    index_name: str
    chunk_count: int


def run(run_context: RunContext) -> IndexResult:
    """Build and populate the Azure AI Search index."""
    xlsx_path = run_context.data_dir / "processed_data.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"processed_data.xlsx not found in {run_context.data_dir}")

    index_name = f"kb-{run_context.customer_name}"
    summary = run_etl(xlsx_path, index_name, run_context.app_settings, run_context.data_dir)

    result = IndexResult(index_name=summary["index_name"], chunk_count=summary["chunk_count"])
    logger.info("Indexing finished: %s (%d chunks)", result.index_name, result.chunk_count)
    return result
