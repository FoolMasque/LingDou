"""
Specialized processors for different modalities

Includes:
- ContextExtractor: Universal context extraction for multimodal content
- ImageModalProcessor: Specialized processor for image content
- TableModalProcessor: Specialized processor for table content
- EquationModalProcessor: Specialized processor for equation content
- GenericModalProcessor: Processor for other modal content
"""

import re
import json
import time
import base64
from typing import Dict, Any, Tuple, List
from pathlib import Path
from dataclasses import dataclass
import requests

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
from lightrag.lightrag import LightRAG
from dataclasses import asdict
from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock
from lightrag.operate import extract_entities, merge_nodes_and_edges

# Import prompt templates
from raganything.prompt import PROMPTS


@dataclass
class ContextConfig:
    """Configuration for context extraction"""

    context_window: int = 1  # Window size for context extraction
    context_mode: str = "page"  # "page", "chunk", "token"
    max_context_tokens: int = 2000  # Maximum context tokens
    include_headers: bool = True  # Whether to include headers/titles
    include_captions: bool = True  # Whether to include image/table captions
    filter_content_types: List[str] = None  # Content types to include

    def __post_init__(self):
        if self.filter_content_types is None:
            self.filter_content_types = ["text"]


class ContextExtractor:
    """Universal context extractor supporting multiple content source formats"""

    def __init__(self, config: ContextConfig = None, tokenizer=None):
        """Initialize context extractor

        Args:
            config: Context extraction configuration
            tokenizer: Tokenizer for accurate token counting
        """
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer

    def extract_context(
        self,
        content_source: Any,
        current_item_info: Dict[str, Any],
        content_format: str = "auto",
    ) -> str:
        """Extract context for current item from content source

        Args:
            content_source: Source content (list, dict, or other format)
            current_item_info: Information about current item (page_idx, index, etc.)
            content_format: Format hint for content source ("minerU", "text_chunks", "auto", etc.)

        Returns:
            Extracted context text
        """
        if not content_source and not self.config.context_window:
            return ""

        try:
            # Use format hint if provided, otherwise auto-detect
            if content_format == "minerU" and isinstance(content_source, list):
                return self._extract_from_content_list(
                    content_source, current_item_info
                )
            elif content_format == "text_chunks" and isinstance(content_source, list):
                return self._extract_from_text_chunks(content_source, current_item_info)
            elif content_format == "text" and isinstance(content_source, str):
                return self._extract_from_text_source(content_source, current_item_info)
            else:
                # Auto-detect content source format
                if isinstance(content_source, list):
                    return self._extract_from_content_list(
                        content_source, current_item_info
                    )
                elif isinstance(content_source, dict):
                    return self._extract_from_dict_source(
                        content_source, current_item_info
                    )
                elif isinstance(content_source, str):
                    return self._extract_from_text_source(
                        content_source, current_item_info
                    )
                else:
                    logger.warning(
                        f"Unsupported content source type: {type(content_source)}"
                    )
                    return ""
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return ""

    def _extract_from_content_list(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context from MinerU-style content list

        Args:
            content_list: List of content items with page_idx and type info
            current_item_info: Current item information

        Returns:
            Context text from surrounding pages/chunks
        """
        if self.config.context_mode == "page":
            return self._extract_page_context(content_list, current_item_info)
        elif self.config.context_mode == "chunk":
            return self._extract_chunk_context(content_list, current_item_info)
        else:
            return self._extract_page_context(content_list, current_item_info)

    def _extract_page_context(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context based on page boundaries

        Args:
            content_list: List of content items
            current_item_info: Current item with page_idx

        Returns:
            Context text from surrounding pages
        """
        current_page = current_item_info.get("page_idx", 0)
        window_size = self.config.context_window

        start_page = max(0, current_page - window_size)
        end_page = current_page + window_size + 1

        context_texts = []

        for item in content_list:
            item_page = item.get("page_idx", 0)
            item_type = item.get("type", "")

            # Check if item is within context window and matches filter criteria
            if (
                start_page <= item_page < end_page
                and item_type in self.config.filter_content_types
            ):
                text_content = self._extract_text_from_item(item)
                if text_content and text_content.strip():
                    # Add page marker for better context understanding
                    if item_page != current_page:
                        context_texts.append(f"[Page {item_page}] {text_content}")
                    else:
                        context_texts.append(text_content)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _extract_chunk_context(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context based on content chunks

        Args:
            content_list: List of content items
            current_item_info: Current item with index info

        Returns:
            Context text from surrounding chunks
        """
        current_index = current_item_info.get("index", 0)
        window_size = self.config.context_window

        start_idx = max(0, current_index - window_size)
        end_idx = min(len(content_list), current_index + window_size + 1)

        context_texts = []

        for i in range(start_idx, end_idx):
            if i != current_index:
                item = content_list[i]
                item_type = item.get("type", "")

                if item_type in self.config.filter_content_types:
                    text_content = self._extract_text_from_item(item)
                    if text_content and text_content.strip():
                        context_texts.append(text_content)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _extract_text_from_item(self, item: Dict) -> str:
        """Extract text content from a content item

        Args:
            item: Content item dictionary

        Returns:
            Extracted text content
        """
        item_type = item.get("type", "")

        if item_type == "text":
            text = item.get("text", "")
            text_level = item.get("text_level", 0)

            # Add header indication for structured content·
            if self.config.include_headers and text_level > 0:
                return f"{'#' * text_level} {text}"
            return text

        elif item_type == "image" and self.config.include_captions:
            captions = item.get("img_caption", [])
            if captions:
                return f"[Image: {', '.join(captions)}]"

        elif item_type == "table" and self.config.include_captions:
            captions = item.get("table_caption", [])
            if captions:
                return f"[Table: {', '.join(captions)}]"

        return ""

    def _extract_from_dict_source(
        self, dict_source: Dict, current_item_info: Dict
    ) -> str:
        """Extract context from dictionary-based content source

        Args:
            dict_source: Dictionary containing content
            current_item_info: Current item information

        Returns:
            Extracted context text
        """
        # Handle different dictionary structures
        if "content" in dict_source:
            context = str(dict_source["content"])
        elif "text" in dict_source:
            context = str(dict_source["text"])
        else:
            # Try to extract any string values
            text_parts = []
            for value in dict_source.values():
                if isinstance(value, str):
                    text_parts.append(value)
            context = "\n".join(text_parts)

        return self._truncate_context(context)

    def _extract_from_text_source(
        self, text_source: str, current_item_info: Dict
    ) -> str:
        """Extract context from plain text source

        Args:
            text_source: Plain text content
            current_item_info: Current item information

        Returns:
            Truncated text context
        """
        return self._truncate_context(text_source)

    def _extract_from_text_chunks(
        self, text_chunks: List[str], current_item_info: Dict
    ) -> str:
        """Extract context from simple text chunks list

        Args:
            text_chunks: List of text strings
            current_item_info: Current item information with index

        Returns:
            Context text from surrounding chunks
        """
        current_index = current_item_info.get("index", 0)
        window_size = self.config.context_window

        start_idx = max(0, current_index - window_size)
        end_idx = min(len(text_chunks), current_index + window_size + 1)

        context_texts = []
        for i in range(start_idx, end_idx):
            if i != current_index:  # Exclude current chunk
                if i < len(text_chunks):
                    chunk_text = str(text_chunks[i]).strip()
                    if chunk_text:
                        context_texts.append(chunk_text)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _truncate_context(self, context: str) -> str:
        """Truncate context to maximum token limit

        Args:
            context: Context text to truncate

        Returns:
            Truncated context text
        """
        if not context:
            return ""

        # Use tokenizer if available for accurate token counting
        if self.tokenizer:
            tokens = self.tokenizer.encode(context)
            if len(tokens) <= self.config.max_context_tokens:
                return context

            # Truncate to max tokens and decode back to text
            truncated_tokens = tokens[: self.config.max_context_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)

            # Try to end at a sentence boundary
            last_period = truncated_text.rfind(".")
            last_newline = truncated_text.rfind("\n")

            if last_period > len(truncated_text) * 0.8:
                return truncated_text[: last_period + 1]
            elif last_newline > len(truncated_text) * 0.8:
                return truncated_text[:last_newline]
            else:
                return truncated_text + "..."
        else:
            # Fallback to character-based truncation if no tokenizer
            if len(context) <= self.config.max_context_tokens:
                return context

            # Simple truncation - fallback when no tokenizer available
            truncated = context[: self.config.max_context_tokens]

            # Try to end at a sentence boundary
            last_period = truncated.rfind(".")
            last_newline = truncated.rfind("\n")

            if last_period > len(truncated) * 0.8:
                return truncated[: last_period + 1]
            elif last_newline > len(truncated) * 0.8:
                return truncated[:last_newline]
            else:
                return truncated + "..."


class BaseModalProcessor:
    """Base class for modal processors"""

    def __init__(
        self,
        lightrag: LightRAG,
        modal_caption_func,
        context_extractor: ContextExtractor = None,
    ):
        """Initialize base processor

        Args:
            lightrag: LightRAG instance
            modal_caption_func: Function for generating descriptions
            context_extractor: Context extractor instance
        """
        self.lightrag = lightrag
        self.modal_caption_func = modal_caption_func

        # Use LightRAG's storage instances
        self.text_chunks_db = lightrag.text_chunks
        self.chunks_vdb = lightrag.chunks_vdb
        self.entities_vdb = lightrag.entities_vdb
        self.relationships_vdb = lightrag.relationships_vdb
        self.knowledge_graph_inst = lightrag.chunk_entity_relation_graph

        # Use LightRAG's configuration and functions
        self.embedding_func = lightrag.embedding_func
        self.llm_model_func = lightrag.llm_model_func
        self.global_config = asdict(lightrag)
        self.hashing_kv = lightrag.llm_response_cache
        self.tokenizer = lightrag.tokenizer

        # Initialize context extractor with tokenizer if not provided
        if context_extractor is None:
            self.context_extractor = ContextExtractor(tokenizer=self.tokenizer)
        else:
            self.context_extractor = context_extractor
            # Update tokenizer if context_extractor doesn't have one
            if self.context_extractor.tokenizer is None:
                self.context_extractor.tokenizer = self.tokenizer

        # Content source for context extraction
        self.content_source = None
        self.content_format = "auto"

    def set_content_source(self, content_source: Any, content_format: str = "auto"):
        """Set content source for context extraction

        Args:
            content_source: Source content for context extraction
            content_format: Format of content source ("minerU", "text_chunks", "auto")
        """
        self.content_source = content_source
        self.content_format = content_format
        logger.info(f"Content source set with format: {content_format}")

    def _get_context_for_item(self, item_info: Dict[str, Any]) -> str:
        """Get context for current processing item

        Args:
            item_info: Information about current item (page_idx, index, etc.)

        Returns:
            Context text for the item
        """
        if not self.content_source:
            return ""

        try:
            context = self.context_extractor.extract_context(
                self.content_source, item_info, self.content_format
            )
            if context:
                logger.debug(
                    f"Extracted context of length {len(context)} for item: {item_info}"
                )
            return context
        except Exception as e:
            logger.error(f"Error getting context for item {item_info}: {e}")
            return ""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Modal content to process
            content_type: Type of modal content
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (description, entity_info)
        """
        # Subclasses must implement this method
        raise NotImplementedError("Subclasses must implement this method")

    async def _create_entity_and_chunk(
        self,
        modal_chunk: str,
        entity_info: Dict[str, Any],
        file_path: str,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Create entity and text chunk"""
        # Create chunk
        chunk_id = compute_mdhash_id(str(modal_chunk), prefix="chunk-")
        tokens = len(self.tokenizer.encode(modal_chunk))

        # Use provided doc_id or generate one from chunk_id for backward compatibility
        actual_doc_id = doc_id if doc_id else chunk_id

        chunk_data = {
            "tokens": tokens,
            "content": modal_chunk,
            "chunk_order_index": chunk_order_index,
            "full_doc_id": actual_doc_id,  # Use proper document ID
            "file_path": file_path,
        }

        # Store chunk
        await self.text_chunks_db.upsert({chunk_id: chunk_data})

        # Store chunk in vector database for retrieval
        chunk_vdb_data = {
            chunk_id: {
                "content": modal_chunk,
                "full_doc_id": actual_doc_id,
                "tokens": tokens,
                "chunk_order_index": chunk_order_index,
                "file_path": file_path,
            }
        }
        await self.chunks_vdb.upsert(chunk_vdb_data)

        # Create entity node
        node_data = {
            "entity_id": entity_info["entity_name"],
            "entity_type": entity_info["entity_type"],
            "description": entity_info["summary"],
            "source_id": chunk_id,
            "file_path": file_path,
            "created_at": int(time.time()),
        }

        await self.knowledge_graph_inst.upsert_node(
            entity_info["entity_name"], node_data
        )

        # Insert entity into vector database
        entity_vdb_data = {
            compute_mdhash_id(entity_info["entity_name"], prefix="ent-"): {
                "entity_name": entity_info["entity_name"],
                "entity_type": entity_info["entity_type"],
                "content": f"{entity_info['entity_name']}\n{entity_info['summary']}",
                "source_id": chunk_id,
                "file_path": file_path,
            }
        }
        await self.entities_vdb.upsert(entity_vdb_data)

        # Process entity and relationship extraction
        chunk_results = await self._process_chunk_for_extraction(
            chunk_id, entity_info["entity_name"], batch_mode
        )

        return (
            entity_info["summary"],
            {
                "entity_name": entity_info["entity_name"],
                "entity_type": entity_info["entity_type"],
                "description": entity_info["summary"],
                "chunk_id": chunk_id,
            },
            chunk_results,
        )

    def _robust_json_parse(self, response: str) -> dict:
        """Robust JSON parsing with multiple fallback strategies"""

        # Strategy 1:
        # Try direct parsing first
        for json_candidate in self._extract_all_json_candidates(response):
            result = self._try_parse_json(json_candidate)
            if result:
                return result

        # Strategy 2: Try with basic cleanup
        for json_candidate in self._extract_all_json_candidates(response):
            cleaned = self._basic_json_cleanup(json_candidate)
            result = self._try_parse_json(cleaned)
            if result:
                return result

        # Strategy 3: Try progressive quote fixing
        for json_candidate in self._extract_all_json_candidates(response):
            fixed = self._progressive_quote_fix(json_candidate)
            result = self._try_parse_json(fixed)
            if result:
                return result

        # Strategy 4: Fallback to regex field extraction
        return self._extract_fields_with_regex(response)

    def _extract_all_json_candidates(self, response: str) -> list:
        """Extract all possible JSON candidates from response"""
        candidates = []

        # Method 1: JSON in code blocks
        import re

        json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        candidates.extend(json_blocks)

        # Method 2: Balanced braces
        brace_count = 0
        start_pos = -1

        for i, char in enumerate(response):
            if char == "{":
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    candidates.append(response[start_pos : i + 1])

        # Method 3: Simple regex fallback
        simple_match = re.search(r"\{.*\}", response, re.DOTALL)
        if simple_match:
            candidates.append(simple_match.group(0))

        return candidates

    def _try_parse_json(self, json_str: str) -> dict:
        """Try to parse JSON string, return None if failed"""
        if not json_str or not json_str.strip():
            return None

        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None

    def _basic_json_cleanup(self, json_str: str) -> str:
        """Basic cleanup for common JSON issues"""
        # Remove extra whitespace
        json_str = json_str.strip()

        # Fix common quote issues
        json_str = json_str.replace('"', '"').replace('"', '"')  # Smart quotes
        json_str = json_str.replace(""", "'").replace(""", "'")  # Smart apostrophes

        # Fix trailing commas (simple case)
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        return json_str

    def _progressive_quote_fix(self, json_str: str) -> str:
        """Progressive fixing of quote and escape issues"""
        # Only escape unescaped backslashes before quotes
        json_str = re.sub(r'(?<!\\)\\(?=")', r"\\\\", json_str)

        # Fix unescaped backslashes in string values (more conservative)
        def fix_string_content(match):
            content = match.group(1)
            # Only escape obvious problematic patterns
            content = re.sub(r"\\(?=[a-zA-Z])", r"\\\\", content)  # \alpha -> \\alpha
            return f'"{content}"'

        json_str = re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_string_content, json_str)
        return json_str

    def _extract_fields_with_regex(self, response: str) -> dict:
        """Extract required fields using regex as last resort"""
        logger.warning("Using regex fallback for JSON parsing")

        # Extract detailed_description
        desc_match = re.search(
            r'"detailed_description":\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL
        )
        description = desc_match.group(1) if desc_match else ""

        # Extract entity_name
        name_match = re.search(r'"entity_name":\s*"([^"]*(?:\\.[^"]*)*)"', response)
        entity_name = name_match.group(1) if name_match else "unknown_entity"

        # Extract entity_type
        type_match = re.search(r'"entity_type":\s*"([^"]*(?:\\.[^"]*)*)"', response)
        entity_type = type_match.group(1) if type_match else "unknown"

        # Extract summary
        summary_match = re.search(
            r'"summary":\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL
        )
        summary = summary_match.group(1) if summary_match else description[:100]

        return {
            "detailed_description": description,
            "entity_info": {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "summary": summary,
            },
        }

    def _extract_json_from_response(self, response: str) -> str:
        """Legacy method - now handled by _extract_all_json_candidates"""
        candidates = self._extract_all_json_candidates(response)
        return candidates[0] if candidates else None

    def _fix_json_escapes(self, json_str: str) -> str:
        """Legacy method - now handled by progressive strategies"""
        return self._progressive_quote_fix(json_str)

    async def _process_chunk_for_extraction(
        self, chunk_id: str, modal_entity_name: str, batch_mode: bool = False
    ):
        """Process chunk for entity and relationship extraction"""
        chunk_data = await self.text_chunks_db.get_by_id(chunk_id)
        if not chunk_data:
            logger.error(f"Chunk {chunk_id} not found")
            return

        # Create text chunk for vector database
        chunk_vdb_data = {
            chunk_id: {
                "content": chunk_data["content"],
                "full_doc_id": chunk_id,
                "tokens": chunk_data["tokens"],
                "chunk_order_index": chunk_data["chunk_order_index"],
                "file_path": chunk_data["file_path"],
            }
        }

        await self.chunks_vdb.upsert(chunk_vdb_data)

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Prepare chunk for extraction
        chunks = {chunk_id: chunk_data}

        # Extract entities and relationships
        chunk_results = await extract_entities(
            chunks=chunks,
            global_config=self.global_config,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.hashing_kv,
        )

        # Add "belongs_to" relationships for all extracted entities
        processed_chunk_results = []
        for maybe_nodes, maybe_edges in chunk_results:
            for entity_name in maybe_nodes.keys():
                if entity_name != modal_entity_name:  # Skip self-relationship
                    # Create belongs_to relationship
                    relation_data = {
                        "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                        "keywords": "belongs_to,part_of,contained_in",
                        "source_id": chunk_id,
                        "weight": 10.0,
                        "file_path": chunk_data.get("file_path", "manual_creation"),
                    }
                    await self.knowledge_graph_inst.upsert_edge(
                        entity_name, modal_entity_name, relation_data
                    )

                    relation_id = compute_mdhash_id(
                        entity_name + modal_entity_name, prefix="rel-"
                    )
                    relation_vdb_data = {
                        relation_id: {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "keywords": relation_data["keywords"],
                            "content": f"{relation_data['keywords']}\t{entity_name}\n{modal_entity_name}\n{relation_data['description']}",
                            "source_id": chunk_id,
                            "file_path": chunk_data.get("file_path", "manual_creation"),
                        }
                    }
                    await self.relationships_vdb.upsert(relation_vdb_data)

                    # Add to maybe_edges
                    maybe_edges[(entity_name, modal_entity_name)] = [relation_data]

            processed_chunk_results.append((maybe_nodes, maybe_edges))

        if not batch_mode:
            # Merge with correct file_path parameter
            file_path = chunk_data.get("file_path", "manual_creation")
            await merge_nodes_and_edges(
                chunk_results=chunk_results,
                knowledge_graph_inst=self.knowledge_graph_inst,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=self.global_config,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.hashing_kv,
                current_file_number=1,
                total_files=1,
                file_path=file_path,  # Pass the correct file_path
            )

            # Ensure all storage updates are complete
            await self.lightrag._insert_done()

        return processed_chunk_results


class ImageModalProcessor(BaseModalProcessor):
    """Processor specialized for image content"""

    def __init__(
        self,
        lightrag: LightRAG,
        modal_caption_func,
        context_extractor: ContextExtractor = None,
    ):
        """Initialize image processor

        Args:
            lightrag: LightRAG instance
            modal_caption_func: Function for generating descriptions (supporting image understanding)
            context_extractor: Context extractor instance
        """
        super().__init__(lightrag, modal_caption_func, context_extractor)

    # def _encode_image_to_base64(self, image_path: str) -> str:
    #     """Encode image to base64"""
    #     try:
    #         with open(image_path, "rb") as image_file:
    #             encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    #         return encoded_string
    #     except Exception as e:
    #         logger.error(f"Failed to encode image {image_path}: {e}")
    #         return ""
######################
    def _encode_image_to_base64(self, image_path: str) -> str:
        try:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                print("🔄 从 URL 下载图片:", image_path)
                # 从 URL 下载图片内容
                response = requests.get(image_path)
                response.raise_for_status()
                encoded_string = base64.b64encode(response.content).decode("utf-8")
            else:
                # 从本地读取
                print("📂 从本地读取图片:", image_path)
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""

    def _extract_cover_description(self, cover_resp: str, fallback: str = "") -> str:
        """优先从cover_resp的JSON里取description，取不到用fallback"""
        try:
            data = self._robust_json_parse(cover_resp)
            desc = data.get("description")
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
        except Exception:
            pass
        return (fallback or "").strip()

    def _write_jsonl_record(self, record: dict, out_path: str):
        """将record追加写入JSONL文件"""
        import json, os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    





#     async def generate_description_only(
#         self,
#         modal_content,
#         content_type: str,
#         item_info: Dict[str, Any] = None,
#         entity_name: str = None,
#     ) -> Tuple[str, Dict[str, Any]]:
#         """
#         Generate image description and entity info only, without entity relation extraction.
#         Used for batch processing stage 1.

#         Args:
#             modal_content: Image content to process
#             content_type: Type of modal content ("image")
#             item_info: Item information for context extraction
#             entity_name: Optional predefined entity name

#         Returns:
#             Tuple of (enhanced_caption, entity_info)
#         """
#         try:
#             # Parse image content (reuse existing logic)
#             if isinstance(modal_content, str):
#                 try:
#                     content_data = json.loads(modal_content)
#                 except json.JSONDecodeError:
#                     content_data = {"description": modal_content}
#             else:
#                 content_data = modal_content

#             # image_path = content_data.get("img_path")
#             image_paths = content_data.get("img_path", [])
#             if isinstance(image_paths, str):
#                 image_paths = [image_paths]


#             captions = content_data.get("img_caption", [])
#             footnotes = content_data.get("img_footnote", [])

        

#             # Convert to Path object and check if it exists
#             image_path_obj = Path(image_path)
#             # if not image_path_obj.exists():
#             #     raise FileNotFoundError(f"Image file not found: {image_path}")
#             ####################################################
#             # Validate image path
#             if not image_path:
#                 raise ValueError(f"No image path provided in modal_content: {modal_content}")



#             #  新增判断：如果是 URL，就不检查本地路径
#             if not (image_path.startswith("http://") or image_path.startswith("https://")):
#                 image_path_obj = Path(image_path)
#                 if not image_path_obj.exists():
#                     raise FileNotFoundError(f"tHIS Image file not found: {image_path}")
# ##############################################################################


#             # Extract context for current item
#             context = ""
#             if item_info:
#                 context = self._get_context_for_item(item_info)

#             # Build detailed visual analysis prompt with context
#             if context:
#                 vision_prompt = PROMPTS.get(
#                     "vision_prompt_with_context", PROMPTS["vision_prompt"]
#                 ).format(
#                     context=context,
#                     entity_name=entity_name
#                     if entity_name
#                     else "unique descriptive name for this image",
#                     image_path=image_path,
#                     captions=captions if captions else "None",
#                     footnotes=footnotes if footnotes else "None",
#                 )
#             else:
#                 vision_prompt = PROMPTS["vision_prompt"].format(
#                     entity_name=entity_name
#                     if entity_name
#                     else "unique descriptive name for this image",
#                     image_path=image_path,
#                     captions=captions if captions else "None",
#                     footnotes=footnotes if footnotes else "None",
#                 )




#             # Encode image to base64
#             image_base64 = self._encode_image_to_base64(image_path)
#             if not image_base64:
#                 raise RuntimeError(f"Failed to encode image to base64: {image_path}")
            
#             print(f"[DEBUG] 920当前 image_path: {image_path}")
#             print(f"[DEBUG] 920已成功编码为 base64: {image_base64[:50]}...")

#             # Call vision model with encoded image
#             response = await self.modal_caption_func(
#                 vision_prompt,
#                 image_data=image_base64,
#                 system_prompt=PROMPTS["IMAGE_ANALYSIS_SYSTEM"],
#             )

#             # Parse response (reuse existing logic)
#             enhanced_caption, entity_info = self._parse_response(response, entity_name)

#             return enhanced_caption, entity_info

#         except Exception as e:
#             logger.error(f"Error generating image description: {e}")
#             # Fallback processing
#             fallback_entity = {
#                 "entity_name": entity_name
#                 if entity_name
#                 else f"image_{compute_mdhash_id(str(modal_content))}",
#                 "entity_type": "image",
#                 "summary": f"Image content: {str(modal_content)[:100]}",
#             }
#             return str(modal_content), fallback_entity

    # def _extract_texts_from_resp(self,resp: str) -> str:
    #     """
    #     从模型响应中提取所有 'text'（或 'ocr_text'），合并为一个字符串。
    #     兼容 <|begin_of_box|>JSON<|end_of_box|> 包裹、转义JSON、以及纯文本。
    #     """
    #     if resp is None:
    #         return ""
    #     s = str(resp)

    #     texts = []

    #     # 1) 先按 box 分段
    #     boxes = re.findall(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", s, flags=re.S)
    #     segments = boxes if boxes else [s]

    #     for seg in segments:
    #         seg = seg.strip()

    #         # 2) 先尝试当作 JSON 直接解析
    #         data = None
    #         try:
    #             data = json.loads(seg)
    #         except Exception:
    #             # 3) 如果 seg 是转义的 JSON 字符串（里头有 \" 和 \\n）
    #             try:
    #                 data = json.loads(seg.encode("utf-8").decode("unicode_escape"))
    #             except Exception:
    #                 data = None

    #         # 4) JSON 解析成功：取 text/ocr_text/description
    #         if isinstance(data, dict):
    #             txt = data.get("text") or data.get("ocr_text") or data.get("description") or ""
    #             if isinstance(txt, str) and txt.strip():
    #                 # 反转义换行
    #                 txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()
    #                 texts.append(txt)
    #                 continue

    #         # 5) JSON解析失败：正则从 seg 里抓 "text":"..."
    #         m = re.search(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', seg)
    #         if m:
    #             raw = m.group(1)
    #             try:
    #                 # 反转义，例如 \n / \u4e00
    #                 raw = bytes(raw, "utf-8").decode("unicode_escape")
    #             except Exception:
    #                 pass
    #             texts.append(raw.replace("\r\n", "\n").replace("\r", "\n").strip())
    #             continue

    #         # 6) 最后兜底：把 seg 当纯文本清洗
    #         fallback = seg.replace("```", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    #         if fallback:
    #             texts.append(fallback)

    #     # 合并
    #     merged = "\n".join(t for t in texts if t)
    #     return merged

    #解决乱码
    def json_unescape_safe(self,s: str) -> str:
        """
        只在确实有转义的情况下，才用 JSON 方式反转义。
        避免把本来就正常的 UTF-8 中文再错误解码成乱码。
        """
        if not isinstance(s, str):
            return s
        # 只有当检测到常见转义符号，才尝试反转义
        if "\\u" in s or "\\n" in s or "\\t" in s or '\\"' in s or "\\r" in s:
            try:
                return json.loads(f'"{s}"')  # 最安全的字符串反转义方式
            except Exception:
                return s
        return s

    def looks_like_mojibake(self,s: str) -> bool:
        """粗略检测是否出现常见的 mojibake 片段"""
        return any(token in s for token in ["ä¸­", "å­", "é¢", "ä½", "æ", "ç¾"])

    def try_fix_mojibake(self,s: str) -> str:
        """
        尝试把 Latin-1 误解码的 UTF-8 文本修回（启发式）。
        仅在检测到疑似 mojibake 时尝试，失败就原样返回。
        """
        if not s or not self.looks_like_mojibake(s):
            return s
        try:
            return s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return s

    def extract_texts_from_resp(self,resp: str) -> str:
        """
        从模型响应中提取所有 'text'（或 'ocr_text'），合并为一个字符串。
        兼容 <|begin_of_box|>JSON<|end_of_box|> 包裹、转义 JSON、纯文本三种情况。
        """
        if resp is None:
            return ""
        s = str(resp)

        texts = []

        # 1) 先按 box 分段
        boxes = re.findall(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", s, flags=re.S)
        segments = boxes if boxes else [s]

        for seg in segments:
            seg = seg.strip()

            # 2) 优先按 JSON 解析（原始/转义）
            data = None
            try:
                data = json.loads(seg)
            except Exception:
                # seg 可能是“转义后的 JSON 串”
                try:
                    data = json.loads(seg.encode("utf-8").decode("unicode_escape"))
                except Exception:
                    data = None

            if isinstance(data, dict):
                txt = data.get("text") or data.get("ocr_text") or data.get("description") or ""
                txt = self.json_unescape_safe(txt)
                txt = self.try_fix_mojibake(txt)
                txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()
                if txt:
                    texts.append(txt)
                continue

            # 3) JSON 失败：正则提取 "text":"..."
            m = re.search(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', seg)
            if m:
                raw = m.group(1)
                raw = self.json_unescape_safe(raw)
                raw = self.try_fix_mojibake(raw)
                raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
                if raw:
                    texts.append(raw)
                continue

            # 4) 最后兜底：纯文本清洗
            fallback = seg.replace("```", "").replace("\r\n", "\n").replace("\r", "\n").strip()
            fallback = self.try_fix_mojibake(fallback)
            if fallback:
                texts.append(fallback)

        return "\n".join(t for t in texts if t)














    # def parse_fields_from_caption(self, caption: str) -> dict:
    #     """
    #     从 generate_product_caption 产出的 caption 中提取字段：
    #     - 风格, 子类, 商品名, subtitle(可多行), keyword
    #     """
    #     CAP_KEYS = ["风格", "子类", "商品名", "subtitle", "keyword"]
    #     out = {k: "" for k in CAP_KEYS}
    #     # 构造一个正则，抓取 label 后面的内容，直到下一个 - 标签 或 '以下是' 或 文本结束
    #     labels_union = "|".join(map(re.escape, CAP_KEYS))
    #     for label in CAP_KEYS:
    #         pat = rf"-\s*{re.escape(label)}\s*:\s*(.*?)(?=\n-\s*(?:{labels_union})\s*:|\n以下是|$)"
    #         m = re.search(pat, caption, flags=re.S)
    #         if m:
    #             out[label] = m.group(1).strip()
    #     return out

    def parse_fields_from_caption(self, caption: str) -> dict:
        """
        从 generate_product_caption 产出的 caption 中提取所有以 '- key: value' 格式出现的字段。
        支持任意字段名，value 可多行，直到下一个 '- key: ' 或 '以下是' 或文本结束。
        """
        out = {}
        # 匹配 '- key: value' 格式，key 为任意非冒号字符，value 直到下一个 '- key: ' 或 '以下是' 或文本结束
        pat = r"-\s*([^:]+?)\s*:\s*(.*?)(?=\n-\s*[^:]+?\s*:|\n以下是|$)"
        matches = re.findall(pat, caption, flags=re.S)
        for key, value in matches:
            out[key.strip()] = value.strip()
        return out

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate image description and entity info for multiple images (with summary).
        """
        try:
            # Parse modal content
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"description": modal_content}
            else:
                content_data = modal_content

            image_paths = content_data.get("img_path", [])



            # Handle img_path: dict (cover_pic + detail_images) or list (detail_images only)
            cover_url = ""
            detail_urls = []
            if isinstance(image_paths, dict):
                cover_url = image_paths.get("cover_pic", "")
                detail_urls = image_paths.get("detail_images", [])
                if isinstance(detail_urls, str):
                    detail_urls = [detail_urls]
            else:
                # If img_path is a list or string, treat all as detail_images
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                detail_urls = image_paths
                print(f"[DEBUG] 1020 detail_urls: {detail_urls}")     


            raw_caption = content_data.get("img_caption")
            raw_footnote = content_data.get("img_footnote")
            

            if isinstance(raw_caption, list):
                shared_caption = raw_caption[0] if raw_caption else "None"
            else:
                shared_caption = raw_caption or "None"

            if isinstance(raw_footnote, list):
                shared_footnote = raw_footnote[0] if raw_footnote else "None"
            else:
                shared_footnote = raw_footnote or "None"


            if not image_paths:
                raise ValueError("No image path provided in modal_content")

            context = self._get_context_for_item(item_info) if item_info else ""

            combined_captions = []
            combined_descriptions = []

            cover_description_final = ""   # 新增：保存封面的最终description
            detail_texts = []              # 新增：收集详情OCR文本



            # 处理封面 
            if cover_url:
                cover_prompt = PROMPTS.get(
                    "vision_prompt_cover", PROMPTS["vision_prompt"]
                ).format(
                    context=context,
                    entity_name=entity_name,
                    image_path=cover_url,
                    captions=shared_caption,
                    footnotes=shared_footnote,
                )
                
                cover_image_b64 = self._encode_image_to_base64(cover_url)
                if not cover_image_b64:
                    raise RuntimeError(f"Failed to encode image to base64: {cover_url}")

                cover_resp = await self.modal_caption_func(
                    cover_prompt,
                    image_data=cover_image_b64,
                    system_prompt=PROMPTS["IMAGE_ANALYSIS_SYSTEM"],
                )
                print("..........................[COVER RESP]........................")
                print(cover_resp)
                print(".............................................................")

                # 解析封面响应（支持6字段/旧schema/OCRschema）
                cover_caption, cover_entity = self._parse_response(
                    cover_resp, entity_name=f"{entity_name or 'image'}_cover"
                )

                cover_description_final = self._extract_cover_description(cover_resp, fallback=cover_caption)
                combined_captions.append(f"[封面] {cover_caption}")
                combined_descriptions.append(cover_entity.get("summary", ""))

            # 先过滤掉空链接，避免后面重复判断
            detail_urls = [u for u in (detail_urls or []) if u]

            for i, url in enumerate(detail_urls, start=1):
                safe_captions = ""
                detail_prompt = PROMPTS.get("detail_prompt", PROMPTS["vision_prompt"]).format(
                    context=context,
                    entity_name=entity_name or f"image_detail_{i}",
                    image_path=url,
                    captions=safe_captions,
                    footnotes=shared_footnote or "",
                )

                ocr_text = "" 
                detail_b64 = self._encode_image_to_base64(url)

                if detail_b64:
                    try:
                        detail_resp = await self.modal_caption_func(
                            detail_prompt,
                            image_data=detail_b64,
                            system_prompt=PROMPTS.get("DETAIL_OCR_SYSTEM", "只输出纯文本（OCR结果），禁止JSON或解释。"),
                        )
                        print(f"..........................[DETAIL {i} RESP]....................")
                        print(detail_resp)
                        print("...............................................................")
                        ocr_text = self.extract_texts_from_resp(detail_resp)
                        detail_texts.append(ocr_text)
                        
                    except Exception as e:
                        logger.error(f"OCR detail image failed (idx={i}, url={url}): {e}")
                        ocr_text = ""  # 失败兜底为空
                else:
                    # base64 编码失败或 url 不可用
                    logger.warning(f"Skip detail image (cannot encode): {url}")
                    ocr_text = ""


                # 详情的实体信息：轻量构造，summary 放OCR前80字符
                summary_piece = (ocr_text[:80] + "...") if (ocr_text and len(ocr_text) > 80) else ocr_text
                detail_entity = {
                    "entity_name": f"{entity_name or 'image'}_detail{i}",
                    "entity_type": "image",
                    "summary": summary_piece,
                }
                combined_captions.append(f"[详情{i}] {ocr_text}")
                combined_descriptions.append(detail_entity["summary"])


            # 汇总
            enhanced_caption = "\n\n".join([c for c in combined_captions if c])
            summary = " ".join([s for s in combined_descriptions if s])
            final_entity = {
                "entity_name": entity_name or "Image Group",
                "entity_type": "image collection",
                "description": enhanced_caption,
                "summary": summary,
            }
            # CAP_KEYS = ["风格", "子类", "商品名", "subtitle", "keyword"]

            cap_fields = self.parse_fields_from_caption(shared_caption)
            base = {k: (cap_fields.get(k) or (item_info or {}).get(k, "")) for k in cap_fields}


            # Build record with dynamic fields from caption
            record = {
                **base,  # Include all caption fields dynamically
                "cover_pic": cover_url,
                "description": cover_description_final,
                "text": "\n\n".join([t for t in detail_texts if t]),
                "detail_images": detail_urls,
            }

            #  写入文件（JSONL）
            self._write_jsonl_record(record, out_path="./exports/products_test_3.jsonl")
            return enhanced_caption, final_entity

            # record = {
            #     "风格": base["风格"],
            #     "子类": base["子类"],
            #     "商品名": base["商品名"],
            #     "subtitle": base["subtitle"],
            #     "keyword": base["keyword"],
            #     "cover_pic": cover_url,
            #     "description": cover_description_final,                 # 封面产出的描述
            #     "text": "\n\n".join([t for t in detail_texts if t]),    # 详情OCR合并
            #     "detail_images": detail_urls,
            # }

        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            fallback_entity = {
                "entity_name": entity_name or f"image_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "image",
                "summary": f"Image content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

            # for idx, image_path in enumerate(image_paths):
            #     vision_prompt = PROMPTS.get(
            #         "vision_prompt_with_context", PROMPTS["vision_prompt"]
            #     ).format(
            #         context=context,
            #         entity_name=entity_name or f"image_{idx+1}",
            #         image_path=image_path,
            #         captions=shared_caption,
            #         footnotes=shared_footnote,
            #     )

            #     image_base64 = self._encode_image_to_base64(image_path)
            #     if not image_base64:
            #         raise RuntimeError(f"Failed to encode image to base64: {image_path}")

            #     response = await self.modal_caption_func(
            #         vision_prompt,
            #         image_data=image_base64,
            #         system_prompt=PROMPTS["IMAGE_ANALYSIS_SYSTEM"],
            #     )
            #     print(".............................................................")
            #     print(response)
            #     print(".............................................................")

            #     # 使用统一结构提取 description 和 entity_info（含 summary）
            #     caption, parsed_entity = self._parse_response(response, entity_name=f"{entity_name or 'image'}_{idx+1}")
            #     combined_captions.append(f"[图 {idx+1}] {caption}")
            #     combined_descriptions.append(parsed_entity["summary"])

            # enhanced_caption = "\n\n".join(combined_captions)
            # ##summary = " ".join(combined_descriptions)[:300] + "..."
            # summary = " ".join(combined_descriptions)  




    # async def process_multimodal_content(
    #     self,
    #     modal_content,
    #     content_type: str,
    #     file_path: str = "manual_creation",
    #     entity_name: str = None,
    #     item_info: Dict[str, Any] = None,
    #     batch_mode: bool = False,
    #     doc_id: str = None,
    #     chunk_order_index: int = 0,
    # ) -> Tuple[str, Dict[str, Any]]:
    #     """Process image content with context support"""
    #     try:
    #         # Generate description and entity info
    #         enhanced_caption, entity_info = await self.generate_description_only(
    #             modal_content, content_type, item_info, entity_name
    #         )

    #         # Build complete image content
    #         if isinstance(modal_content, str):
    #             try:
    #                 content_data = json.loads(modal_content)
    #             except json.JSONDecodeError:
    #                 content_data = {"description": modal_content}
    #         else:
    #             content_data = modal_content

    #         image_path = content_data.get("img_path", "")
    #         captions = content_data.get("img_caption", [])
    #         footnotes = content_data.get("img_footnote", [])

    #         modal_chunk = PROMPTS["image_chunk"].format(
    #             image_path=image_path,
    #             captions=", ".join(captions) if captions else "None",
    #             footnotes=", ".join(footnotes) if footnotes else "None",
    #             enhanced_caption=enhanced_caption,
    #         )

    #         return await self._create_entity_and_chunk(
    #             modal_chunk,
    #             entity_info,
    #             file_path,
    #             batch_mode,
    #             doc_id,
    #             chunk_order_index,
    #         )

    #     except Exception as e:
    #         logger.error(f"Error processing image content: {e}")
    #         # Fallback processing
    #         fallback_entity = {
    #             "entity_name": entity_name
    #             if entity_name
    #             else f"image_{compute_mdhash_id(str(modal_content))}",
    #             "entity_type": "image",
    #             "summary": f"Image content: {str(modal_content)[:100]}",
    #         }
    #         return str(modal_content), fallback_entity




# #######################可以
#     async def process_multimodal_content(
#         self,
#         modal_content,
#         content_type: str,
#         file_path: str = "manual_creation",
#         entity_name: str = None,
#         item_info: Dict[str, Any] = None,
#         batch_mode: bool = False,
#         doc_id: str = None,
#         chunk_order_index: int = 0,
#     ) -> Tuple[str, Dict[str, Any]]:
#         """Process multi-image content with context support"""
#         try:
#             #  生成统一描述和一个合并实体
#             enhanced_caption, entity_info = await self.generate_description_only(
#                 modal_content, content_type, item_info, entity_name
#             )
#             # 解析 modal_content
#             if isinstance(modal_content, str):
#                 try:
#                     content_data = json.loads(modal_content)
#                 except json.JSONDecodeError:
#                     content_data = {"description": modal_content}
#             else:
#                 content_data = modal_content

#             image_paths = content_data.get("img_path", [])
#             captions = content_data.get("img_caption", [])
#             footnotes = content_data.get("img_footnote", [])

#             if isinstance(image_paths, str):
#                 image_paths = [image_paths]
#             if isinstance(captions, str):
#                 captions = [captions]
#             if isinstance(footnotes, str):
#                 footnotes = [footnotes]

#             # 拼接多个图片的 path/caption/footnote（用于 prompt 记录）
#             image_info_blocks = []
#             for i, path in enumerate(image_paths):
#                 cap = captions[0]
#                 foot = footnotes[0]
#                 image_info_blocks.append(f"[图 {i+1}]\nPath: {path}\nCaption: {cap}\nFootnote: {foot}")


#             combined_meta_info = "\n\n".join(image_info_blocks)

#             modal_chunk = PROMPTS["image_chunk"].format(
#                 image_path=combined_meta_info,
#                 captions=", ".join(captions) if captions else "None",
#                 footnotes=", ".join(footnotes) if footnotes else "None",
#                 enhanced_caption=enhanced_caption,
#             )
#             #("entity_info",entity_info)

#             return await self._create_entity_and_chunk(
#                 modal_chunk,
#                 entity_info,
#                 file_path,
#                 batch_mode,
#                 doc_id,
#                 chunk_order_index,
#             )
            

#         except Exception as e:
#             logger.error(f"Error processing image content: {e}")
#             fallback_entity = {
#                 "entity_name": entity_name or f"image_{compute_mdhash_id(str(modal_content))}",
#                 "entity_type": "image",
#                 "summary": f"Image content: {str(modal_content)[:100]}",
#             }
#             return str(modal_content), fallback_entity


    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any], str]:
        """Process multi-image content with context support (兼容 img_path dict/list)"""
        try:
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # 解析 modal_content
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"description": modal_content}
            else:
                content_data = modal_content or {}

            img_path = content_data.get("img_path", {})
            captions = content_data.get("img_caption", [])
            footnotes = content_data.get("img_footnote", [])

            # 统一成字符串（避免 captions[0] 越界/None 进入模板）
            if isinstance(captions, list):
                safe_caption = captions[0] if captions else ""
            else:
                safe_caption = captions or ""
            if isinstance(footnotes, list):
                safe_foot = footnotes[0] if footnotes else ""
            else:
                safe_foot = footnotes or ""

            # 兼容两种 img_path 结构
            cover_url = ""
            detail_urls = []
            if isinstance(img_path, dict):
                cover_url = img_path.get("cover_pic") or ""
                detail_urls = img_path.get("detail_images") or []
                if isinstance(detail_urls, str):
                    detail_urls = [detail_urls]
            else:
                # 旧结构：列表/字符串
                if isinstance(img_path, str):
                    img_path = [img_path]
                detail_urls = img_path or []
                

            # 过滤空链接
            detail_urls = [u for u in detail_urls if u]

            # 组装 image_info_blocks（用于记录/可读性）
            blocks = []
            if cover_url:
                blocks.append(f"[封面]\nPath: {cover_url}\nCaption: {safe_caption}\nFootnote: {safe_foot}")
            for i, u in enumerate(detail_urls, start=1):
                blocks.append(f"[详情{i}]\nPath: {u}\nCaption: \nFootnote: {safe_foot}")

            combined_meta_info = "\n\n".join(blocks) if blocks else "None"

            # 写入 chunk（如果你需要把 cover+details 的 meta 与 enhanced_caption 存到 RAG）
            modal_chunk = PROMPTS["image_chunk"].format(
                image_path=combined_meta_info,
                captions=safe_caption if safe_caption else "None",
                footnotes=safe_foot if safe_foot else "None",
                enhanced_caption=enhanced_caption,
            )

            # 注意：_create_entity_and_chunk 原本返回值是啥就保持即可；这里我们不依赖它的返回值
            await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

            return enhanced_caption, entity_info, combined_meta_info

        except Exception as e:
            logger.error(f"Error processing image content: {e}")
            fallback_entity = {
                "entity_name": entity_name or f"image_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "image",
                "summary": f"Image content: {str(modal_content)[:100]}",
            }
            # 也返回3个，第三个给空字符串，避免调用处再次报错
            return str(modal_content), fallback_entity, ""



    def _parse_response(self, response: str, entity_name: str = None) -> Tuple[str, Dict[str, Any]]:
        """Parse model response, supporting old schema, OCR schema, and 6-field schema"""
        try:
            data = self._robust_json_parse(response)


            # -------- 情况2：OCR结构(page2:goods) --------
            if "text" in data and "description" in data and "entity_info" in data:
                description = data.get("text", "")  #优先ocr
                entity_data = data.get("entity_info", {})
                for field in ["entity_name", "entity_type", "summary"]:
                    if field not in entity_data:
                        raise ValueError(f"Missing {field} in entity_info")
                if entity_data["entity_type"] not in ["image", "image collection"]:
                    entity_data["entity_type"] = "image"
                if entity_name:
                    entity_data["entity_name"] = entity_name
                return description, entity_data

            # -------- 情况1：旧结构 --------
            if "description" in data and "entity_info" in data:
                description = data.get("description", "")
                entity_data = data.get("entity_info", {})
                for field in ["entity_name", "entity_type", "summary"]:
                    if field not in entity_data:
                        raise ValueError(f"Missing {field} in entity_info")
                if entity_data["entity_type"] not in ["image", "image collection"]:
                    entity_data["entity_type"] = "image"
                if entity_name:
                    entity_data["entity_name"] = entity_name
                else:
                    entity_data["entity_name"] = (
                        entity_data["entity_name"] + f" ({entity_data['entity_type']})"
                    )
                return description, entity_data

            # -------- 其他情况 --------
            raise ValueError("Unsupported schema in response")

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing image analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name or f"image_{compute_mdhash_id(response)}",
                "entity_type": "image",
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity

##############################可行
    # def _parse_response(self, response: str, entity_name: str = None) -> Tuple[str, Dict[str, Any]]:
    #     """Parse model response"""
    #     try:
    #         response_data = self._robust_json_parse(response)
    #         description = response_data.get("detailed_description", "")
    #         entity_data = response_data.get("entity_info", {})

    #         if not description or not entity_data:
    #             raise ValueError("Missing required fields in response")

    #         for field in ["entity_name", "entity_type", "summary"]:
    #             if field not in entity_data:
    #                 raise ValueError(f"Missing {field} in entity_info")

    #         # 允许 entity_type 为 "image collection"
    #         if entity_data["entity_type"] not in ["image", "image collection"]:
    #             entity_data["entity_type"] = "image"

    #         # 添加 entity_name 修饰
    #         entity_data["entity_name"] = (
    #             entity_data["entity_name"] + f" ({entity_data['entity_type']})"
    #         )
    #         if entity_name:
    #             entity_data["entity_name"] = entity_name

    #         return description, entity_data

    #     except (json.JSONDecodeError, AttributeError, ValueError) as e:
    #         logger.error(f"Error parsing image analysis response: {e}")
    #         logger.debug(f"Raw response: {response}")
    #         fallback_entity = {
    #             "entity_name": entity_name or f"image_{compute_mdhash_id(response)}",
    #             "entity_type": "image",
    #             "summary": response[:100] + "..." if len(response) > 100 else response,
    #         }
    #         return response, fallback_entity 

    # def _parse_response(self, response: str, entity_name: str = None) -> Tuple[str, Dict[str, Any]]:
    #     """Parse model response"""
    #     try:
    #         response_data = self._robust_json_parse(response)
    #         description = response_data.get("detailed_description", "")
    #         entity_data = response_data.get("entity_info", {})

    #         if not description:
    #             raise ValueError("Missing 'detailed_description' in response")

    #         if not isinstance(entity_data, dict):
    #             entity_data = {}

    #         # 自动生成 entity_info 字段（如果缺失）
    #         entity_data.setdefault("entity_name", entity_name or f"image_{compute_mdhash_id(response)}")
    #         entity_data.setdefault("entity_type", "image")
    #         entity_data.setdefault("summary", description[:100] + "..." if len(description) > 100 else description)
    #         #print("summary", entity_data["summary"])

    #         # 允许 entity_type 为 image 或 image collection
    #         if entity_data["entity_type"] not in ["image", "image collection"]:
    #             entity_data["entity_type"] = "image"

    #         # 若 entity_name 已经自带类型标识就不再重复添加
    #         if " (" not in entity_data["entity_name"]:
    #             entity_data["entity_name"] += f" ({entity_data['entity_type']})"

    #         # 若调用者传入了 entity_name，则覆盖（用于手动命名）
    #         if entity_name:
    #             entity_data["entity_name"] = entity_name

    #         return description, entity_data

    #     except (json.JSONDecodeError, AttributeError, ValueError) as e:
    #         logger.error(f"Error parsing image analysis response: {e}")
    #         logger.debug(f"Raw response: {response}")

    #         fallback_entity = {
    #             "entity_name": entity_name or f"image_{compute_mdhash_id(response)}",
    #             "entity_type": "image",
    #             "summary": response[:100] + "..." if len(response) > 100 else response,
    #         }
    #         return response, fallback_entity



class TableModalProcessor(BaseModalProcessor):
    """Processor specialized for table content"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate table description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Table content to process
            content_type: Type of modal content ("table")
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        try:
            # Parse table content (reuse existing logic)
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"table_body": modal_content}
            else:
                content_data = modal_content

            table_img_path = content_data.get("img_path")
            table_caption = content_data.get("table_caption", [])
            table_body = content_data.get("table_body", "")
            table_footnote = content_data.get("table_footnote", [])

            # Extract context for current item
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # Build table analysis prompt with context
            if context:
                table_prompt = PROMPTS.get(
                    "table_prompt_with_context", PROMPTS["table_prompt"]
                ).format(
                    context=context,
                    entity_name=entity_name
                    if entity_name
                    else "descriptive name for this table",
                    table_img_path=table_img_path,
                    table_caption=table_caption if table_caption else "None",
                    table_body=table_body,
                    table_footnote=table_footnote if table_footnote else "None",
                )
            else:
                table_prompt = PROMPTS["table_prompt"].format(
                    entity_name=entity_name
                    if entity_name
                    else "descriptive name for this table",
                    table_img_path=table_img_path,
                    table_caption=table_caption if table_caption else "None",
                    table_body=table_body,
                    table_footnote=table_footnote if table_footnote else "None",
                )

            # Call LLM for table analysis
            response = await self.modal_caption_func(
                table_prompt,
                system_prompt=PROMPTS["TABLE_ANALYSIS_SYSTEM"],
            )

            # Parse response (reuse existing logic)
            enhanced_caption, entity_info = self._parse_table_response(
                response, entity_name
            )

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating table description: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"table_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "table",
                "summary": f"Table content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Process table content with context support"""
        try:
            # Generate description and entity info
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # Parse table content for building complete chunk
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"table_body": modal_content}
            else:
                content_data = modal_content

            table_img_path = content_data.get("img_path")
            table_caption = content_data.get("table_caption", [])
            table_body = content_data.get("table_body", "")
            table_footnote = content_data.get("table_footnote", [])

            # Build complete table content
            modal_chunk = PROMPTS["table_chunk"].format(
                table_img_path=table_img_path,
                table_caption=", ".join(table_caption) if table_caption else "None",
                table_body=table_body,
                table_footnote=", ".join(table_footnote) if table_footnote else "None",
                enhanced_caption=enhanced_caption,
            )

            return await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

        except Exception as e:
            logger.error(f"Error processing table content: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"table_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "table",
                "summary": f"Table content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    def _parse_table_response(
        self, response: str, entity_name: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Parse table analysis response"""
        try:
            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            if not description or not entity_data:
                raise ValueError("Missing required fields in response")

            if not all(
                key in entity_data for key in ["entity_name", "entity_type", "summary"]
            ):
                raise ValueError("Missing required fields in entity_info")

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing table analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"table_{compute_mdhash_id(response)}",
                "entity_type": "table",
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity


class EquationModalProcessor(BaseModalProcessor):
    """Processor specialized for equation content"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate equation description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Equation content to process
            content_type: Type of modal content ("equation")
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        try:
            # Parse equation content (reuse existing logic)
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"equation": modal_content}
            else:
                content_data = modal_content

            equation_text = content_data.get("text")
            equation_format = content_data.get("text_format", "")

            # Extract context for current item
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # Build equation analysis prompt with context
            if context:
                equation_prompt = PROMPTS.get(
                    "equation_prompt_with_context", PROMPTS["equation_prompt"]
                ).format(
                    context=context,
                    equation_text=equation_text,
                    equation_format=equation_format,
                    entity_name=entity_name
                    if entity_name
                    else "descriptive name for this equation",
                )
            else:
                equation_prompt = PROMPTS["equation_prompt"].format(
                    equation_text=equation_text,
                    equation_format=equation_format,
                    entity_name=entity_name
                    if entity_name
                    else "descriptive name for this equation",
                )

            # Call LLM for equation analysis
            response = await self.modal_caption_func(
                equation_prompt,
                system_prompt=PROMPTS["EQUATION_ANALYSIS_SYSTEM"],
            )

            # Parse response (reuse existing logic)
            enhanced_caption, entity_info = self._parse_equation_response(
                response, entity_name
            )

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating equation description: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"equation_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "equation",
                "summary": f"Equation content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Process equation content with context support"""
        try:
            # Generate description and entity info
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # Parse equation content for building complete chunk
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"equation": modal_content}
            else:
                content_data = modal_content

            equation_text = content_data.get("text")
            equation_format = content_data.get("text_format", "")

            # Build complete equation content
            modal_chunk = PROMPTS["equation_chunk"].format(
                equation_text=equation_text,
                equation_format=equation_format,
                enhanced_caption=enhanced_caption,
            )

            return await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

        except Exception as e:
            logger.error(f"Error processing equation content: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"equation_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "equation",
                "summary": f"Equation content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    def _parse_equation_response(
        self, response: str, entity_name: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Parse equation analysis response with robust JSON handling"""
        try:
            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            if not description or not entity_data:
                raise ValueError("Missing required fields in response")

            if not all(
                key in entity_data for key in ["entity_name", "entity_type", "summary"]
            ):
                raise ValueError("Missing required fields in entity_info")

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing equation analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"equation_{compute_mdhash_id(response)}",
                "entity_type": "equation",
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity


class GenericModalProcessor(BaseModalProcessor):
    """Generic processor for other types of modal content"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate generic modal description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Generic modal content to process
            content_type: Type of modal content
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        try:
            # Extract context for current item
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # Build generic analysis prompt with context
            if context:
                generic_prompt = PROMPTS.get(
                    "generic_prompt_with_context", PROMPTS["generic_prompt"]
                ).format(
                    context=context,
                    content_type=content_type,
                    entity_name=entity_name
                    if entity_name
                    else f"descriptive name for this {content_type}",
                    content=str(modal_content),
                )
            else:
                generic_prompt = PROMPTS["generic_prompt"].format(
                    content_type=content_type,
                    entity_name=entity_name
                    if entity_name
                    else f"descriptive name for this {content_type}",
                    content=str(modal_content),
                )

            # Call LLM for generic analysis
            response = await self.modal_caption_func(
                generic_prompt,
                system_prompt=PROMPTS["GENERIC_ANALYSIS_SYSTEM"].format(
                    content_type=content_type
                ),
            )

            # Parse response (reuse existing logic)
            enhanced_caption, entity_info = self._parse_generic_response(
                response, entity_name, content_type
            )

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating {content_type} description: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(str(modal_content))}",
                "entity_type": content_type,
                "summary": f"{content_type} content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Process generic modal content with context support"""
        try:
            # Generate description and entity info
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # Build complete content
            modal_chunk = PROMPTS["generic_chunk"].format(
                content_type=content_type.title(),
                content=str(modal_content),
                enhanced_caption=enhanced_caption,
            )

            return await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

        except Exception as e:
            logger.error(f"Error processing {content_type} content: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(str(modal_content))}",
                "entity_type": content_type,
                "summary": f"{content_type} content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    def _parse_generic_response(
        self, response: str, entity_name: str = None, content_type: str = "content"
    ) -> Tuple[str, Dict[str, Any]]:
        """Parse generic analysis response"""
        try:
            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            if not description or not entity_data:
                raise ValueError("Missing required fields in response")

            if not all(
                key in entity_data for key in ["entity_name", "entity_type", "summary"]
            ):
                raise ValueError("Missing required fields in entity_info")

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing {content_type} analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(response)}",
                "entity_type": content_type,
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity
