"""
Knowledge Tools - Custom Agno Toolkit for vector storage with LanceDB

Provides explicit storage control for research findings:
- Save findings with source attribution
- Semantic search over stored findings
- Source management and retrieval

Uses LiteLLM for embeddings, allowing routing through proxy servers.
"""
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from agno.tools import Toolkit
from agno.utils.log import logger

try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    logger.warning("lancedb package not installed. Run: pip install lancedb")

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("litellm package not installed. Run: pip install litellm")


# =============================================================================
# LanceDB Schema
# =============================================================================

# Schema for research findings table
FINDINGS_SCHEMA = {
    "id": str,                # UUID
    "content": str,           # Finding text
    "source_url": str,        # Origin URL
    "source_title": str,      # Page title
    "subtask_id": int,        # Which subtask
    "worker_id": str,         # Which worker
    "timestamp": str,         # ISO timestamp
    "verified": bool,         # URL verification status
    "search_type": str,       # "academic" | "general"
    "quality_score": float,   # 0.0-1.0 quality rating
    "vector": list,           # Embedding vector
}


def _calculate_quality_score(content: str, search_type: str, verified: bool) -> float:
    """
    Calculate quality score for a finding based on content characteristics.
    
    Scoring criteria:
    - Has quantitative data (numbers, percentages, statistics)
    - From academic source
    - Verified source
    - Content length (not too short, not padding)
    - Contains methodology/methods mentions
    
    Returns:
        float: Quality score from 0.0 to 1.0
    """
    import re
    
    score = 0.5  # Base score
    
    # Academic sources get bonus
    if search_type == "academic":
        score += 0.15
    
    # Verified sources get bonus
    if verified:
        score += 0.1
    
    # Check for quantitative data (numbers, percentages, statistics)
    number_pattern = r'\b\d+\.?\d*\s*(%|percent|million|billion|thousand|fold|x\b)'
    if re.search(number_pattern, content, re.IGNORECASE):
        score += 0.15
    
    # Check for specific metrics/benchmarks
    metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'rmsd', 'benchmark', 
                       'performance', 'improvement', 'increase', 'decrease', 'compared to']
    if any(kw in content.lower() for kw in metric_keywords):
        score += 0.1
    
    # Check for methodology mentions
    method_keywords = ['method', 'approach', 'technique', 'algorithm', 'model', 'architecture',
                       'trained', 'evaluated', 'experiment', 'study', 'trial', 'analysis']
    if any(kw in content.lower() for kw in method_keywords):
        score += 0.05
    
    # Penalty for very short content (likely low quality)
    if len(content) < 200:
        score -= 0.1
    
    # Cap between 0 and 1
    return max(0.0, min(1.0, score))


class KnowledgeTools(Toolkit):
    """
    Custom Agno Toolkit for knowledge base operations with LanceDB.
    
    Features:
    - Explicit save_finding() for controlled storage
    - Semantic search over stored findings
    - Source URL tracking and retrieval
    - LiteLLM embeddings (routes through proxy)
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 3072,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k_default: int = 10,
    ):
        """
        Initialize Knowledge Tools.
        
        Args:
            db_path: Path to LanceDB database directory
            embedding_model: Embedding model name
                - "text-embedding-3-large" (3072 dims, best quality)
                - "text-embedding-3-small" (1536 dims, faster)
            embedding_dimensions: Embedding vector dimensions (3072 for large, 1536 for small)
            api_base: LiteLLM API base URL (for proxy)
            api_key: LiteLLM API key
            top_k_default: Default number of results for search
        """
        self.db_path = db_path or os.getenv("LANCEDB_PATH", "./research_kb")
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.top_k_default = top_k_default
        
        # LiteLLM API configuration (supports proxy)
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Lazy-initialized clients
        self._db: Optional[lancedb.DBConnection] = None
        self._table = None
        
        # Register tools with Toolkit
        tools = [
            self.save_finding,
            self.search_knowledge,
            self.list_sources,
            self.get_finding,
            self.get_findings_by_subtask,
            self.get_findings_index,
        ]
        
        super().__init__(name="knowledge_tools", tools=tools)
    
    @property
    def db(self):
        """Lazy initialization of LanceDB connection"""
        if not LANCEDB_AVAILABLE:
            raise ImportError("lancedb package not installed. Run: pip install lancedb")
        
        if self._db is None:
            logger.info(f"Connecting to LanceDB at: {self.db_path}")
            self._db = lancedb.connect(self.db_path)
        
        return self._db
    
    @property
    def table(self):
        """Get or create the findings table"""
        if self._table is None:
            table_name = "findings"
            
            # Check if table exists
            existing_tables = self.db.table_names()
            
            if table_name in existing_tables:
                logger.info(f"Opening existing table: {table_name}")
                self._table = self.db.open_table(table_name)
            else:
                logger.info(f"Creating new table: {table_name}")
                # Create with initial empty record that matches schema
                initial_data = [{
                    "id": "init",
                    "content": "Initialization record",
                    "source_url": "",
                    "source_title": "",
                    "subtask_id": 0,
                    "worker_id": "system",
                    "timestamp": datetime.utcnow().isoformat(),
                    "verified": False,
                    "search_type": "init",
                    "quality_score": 0.0,
                    "vector": [0.0] * self.embedding_dimensions,
                }]
                self._table = self.db.create_table(table_name, data=initial_data)
        
        return self._table
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using LiteLLM (supports proxy routing)"""
        if not LITELLM_AVAILABLE:
            logger.error("LiteLLM not available")
            return [0.0] * self.embedding_dimensions
        
        if not self.api_key:
            logger.error("No API key for embeddings (LITELLM_API_KEY or OPENAI_API_KEY)")
            return [0.0] * self.embedding_dimensions
        
        try:
            # Build kwargs for litellm.embedding
            kwargs = {
                "model": self.embedding_model,
                "input": [text],
            }
            
            # Add API base if using proxy
            if self.api_base:
                kwargs["api_base"] = self.api_base
                kwargs["api_key"] = self.api_key
            
            response = litellm.embedding(**kwargs)
            return response.data[0]["embedding"]
        except Exception as e:
            logger.error(f"LiteLLM embedding failed: {e}")
            return [0.0] * self.embedding_dimensions
    
    # =========================================================================
    # Public Tool Methods (exposed to agents)
    # =========================================================================
    
    def save_finding(
        self,
        content: str,
        source_url: str,
        source_title: Optional[str] = None,
        subtask_id: int = 0,
        worker_id: str = "unknown",
        verified: bool = False,
        search_type: str = "general",
    ) -> str:
        """
        Save a research finding to the knowledge base.
        
        IMPORTANT: Call this explicitly for each valuable finding you discover.
        EXTRACT COMPREHENSIVE CONTENT - aim for 500-2000 characters per finding.
        
        Args:
            content: COMPREHENSIVE finding text (500-2000 chars recommended).
                     Include ALL relevant details: stats, quotes, methodology,
                     comparisons, limitations, technical specs, outcomes.
                     More detail is ALWAYS better.
            source_url: URL where the finding was discovered (required)
            source_title: Title of the source page
            subtask_id: ID of the subtask that found this
            worker_id: ID of the worker agent
            verified: Whether the URL has been verified
            search_type: "academic" or "general"
        
        Returns:
            str: Confirmation message with finding ID
        
        Example:
            >>> tools.save_finding(
            ...     content="GPT-4 Technical Report (OpenAI, March 2023): GPT-4 achieved 86.4% on MMLU benchmark, "
            ...             "surpassing GPT-3.5's 70.0% and approaching human expert level of ~89%. The model was "
            ...             "evaluated across 57 subjects including STEM, humanities, and professional domains. "
            ...             "Notable results: 90th percentile on Uniform Bar Exam, 99th percentile on Biology Olympiad. "
            ...             "Training used RLHF with human feedback. Key limitation: knowledge cutoff September 2021. "
            ...             "The report notes the model still exhibits hallucinations and reasoning errors on complex tasks.",
            ...     source_url="https://arxiv.org/abs/2303.08774",
            ...     source_title="GPT-4 Technical Report",
            ...     search_type="academic",
            ...     verified=True
            ... )
        """
        logger.info(f"Saving finding from: {source_url}")
        
        try:
            # Generate unique ID
            finding_id = str(uuid.uuid4())[:8]
            
            # Truncate content to prevent context window overflow (max 1500 chars)
            MAX_CONTENT_LENGTH = 1500
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "..."
                logger.info(f"Truncated finding content to {MAX_CONTENT_LENGTH} chars")
            
            # Calculate quality score for this finding
            quality_score = _calculate_quality_score(content, search_type, verified)
            
            # Generate embedding
            embedding = self._get_embedding(content)
            
            # Prepare record
            record = {
                "id": finding_id,
                "content": content,
                "source_url": source_url,
                "source_title": source_title or "",
                "subtask_id": subtask_id,
                "worker_id": worker_id,
                "timestamp": datetime.utcnow().isoformat(),
                "verified": verified,
                "search_type": search_type,
                "quality_score": quality_score,
                "vector": embedding,
            }
            
            # Add to table
            self.table.add([record])
            
            return f"## Finding Saved\n\n**ID:** `{finding_id}`\n**Source:** {source_url}\n**Verified:** {'✅' if verified else '❌'}\n**Content preview:** {content[:200]}..."
            
        except Exception as e:
            logger.error(f"Failed to save finding: {e}")
            return f"## Save Error\n\n**Error:** {str(e)}"
    
    def search_knowledge(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_type_filter: Optional[str] = None,
        sort_by_quality: bool = True,
    ) -> str:
        """
        Semantic search over stored research findings.
        
        Args:
            query: Search query string
            top_k: Maximum number of results (default: 10)
            search_type_filter: Filter by "academic" or "general" (optional)
            sort_by_quality: If True, re-rank results by quality_score (default: True)
        
        Returns:
            str: Formatted search results with relevance and quality scores
        
        Example:
            >>> results = tools.search_knowledge("language model benchmarks")
            >>> results = tools.search_knowledge("battery energy density", search_type_filter="academic")
        """
        top_k = top_k or self.top_k_default
        logger.info(f"Searching knowledge base: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Perform vector search - fetch more to allow quality filtering
            search_results = (
                self.table
                .search(query_embedding)
                .limit(top_k * 3)  # Fetch 3x to allow quality-based reranking
                .to_pandas()
            )
            
            # Filter out initialization record
            search_results = search_results[search_results["id"] != "init"]
            
            # Apply search type filter if specified
            if search_type_filter:
                search_results = search_results[
                    search_results["search_type"] == search_type_filter
                ]
            
            # Sort by quality_score (descending) if requested
            if sort_by_quality and "quality_score" in search_results.columns:
                # Combine relevance (1 - distance) with quality for ranking
                # Formula: 0.6 * relevance + 0.4 * quality
                search_results["_combined_score"] = (
                    0.6 * (1 - search_results["_distance"]) + 
                    0.4 * search_results["quality_score"]
                )
                search_results = search_results.sort_values("_combined_score", ascending=False)
            
            # Limit results
            search_results = search_results.head(top_k)
            
            if len(search_results) == 0:
                return f"## Knowledge Search Results\n\n**Query:** {query}\n\nNo relevant findings found."
            
            # Format results
            output = [f"## Knowledge Search Results\n"]
            output.append(f"**Query:** {query}")
            output.append(f"**Found:** {len(search_results)} results (sorted by quality)\n")
            
            # Truncate content to prevent context overflow (max 800 chars per result)
            MAX_RESULT_CONTENT = 800
            
            for idx, (i, row) in enumerate(search_results.iterrows(), 1):
                distance = row.get("_distance", 0)
                quality = row.get("quality_score", 0.5)
                relevance = 1 - distance if isinstance(distance, float) else 0
                
                output.append(f"### {idx}. [{row['id']}] {row['source_title'] or 'Untitled'}")
                output.append(f"**Source:** {row['source_url']}")
                output.append(f"**Type:** {row['search_type']} | **Verified:** {'✅' if row['verified'] else '❌'}")
                output.append(f"**Quality:** {quality:.2f} | **Relevance:** {relevance:.2f}")
                # Truncate content for context efficiency
                content = row['content']
                if len(content) > MAX_RESULT_CONTENT:
                    content = content[:MAX_RESULT_CONTENT] + "..."
                output.append(f"**Content:** {content}")
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return f"## Search Error\n\n**Query:** {query}\n**Error:** {str(e)}"
    
    def list_sources(self, subtask_id: Optional[int] = None) -> str:
        """
        List all unique source URLs stored in the knowledge base.
        
        Useful for generating citations and reference lists.
        
        Args:
            subtask_id: Filter by specific subtask (optional)
        
        Returns:
            str: Formatted list of source URLs with titles
        """
        logger.info("Listing all sources")
        
        try:
            # Get all records
            df = self.table.to_pandas()
            
            # Filter out initialization record
            df = df[df["id"] != "init"]
            
            # Apply subtask filter if specified
            if subtask_id is not None:
                df = df[df["subtask_id"] == subtask_id]
            
            if len(df) == 0:
                return "## Sources\n\nNo sources found in knowledge base."
            
            # Get unique sources
            sources = df[["source_url", "source_title", "verified", "search_type"]].drop_duplicates()
            
            # Format output
            output = ["## Sources in Knowledge Base\n"]
            output.append(f"**Total unique sources:** {len(sources)}\n")
            
            # Group by search type
            for search_type in ["academic", "general"]:
                type_sources = sources[sources["search_type"] == search_type]
                if len(type_sources) > 0:
                    output.append(f"### {search_type.title()} Sources ({len(type_sources)})\n")
                    for _, row in type_sources.iterrows():
                        verified_icon = "✅" if row["verified"] else "❌"
                        title = row["source_title"] or "Untitled"
                        output.append(f"- {verified_icon} [{title}]({row['source_url']})")
                    output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"List sources failed: {e}")
            return f"## Error\n\n**Error:** {str(e)}"
    
    def get_finding(self, finding_id: str) -> str:
        """
        Retrieve a specific finding by its ID.
        
        Args:
            finding_id: The unique ID of the finding
        
        Returns:
            str: Full finding details
        """
        logger.info(f"Getting finding: {finding_id}")
        
        try:
            # Get all records and filter
            df = self.table.to_pandas()
            finding = df[df["id"] == finding_id]
            
            if len(finding) == 0:
                return f"## Finding Not Found\n\n**ID:** {finding_id}"
            
            row = finding.iloc[0]
            
            output = [f"## Finding: {finding_id}\n"]
            output.append(f"**Source:** [{row['source_title'] or 'Untitled'}]({row['source_url']})")
            output.append(f"**Type:** {row['search_type']} | **Verified:** {'✅' if row['verified'] else '❌'}")
            output.append(f"**Worker:** {row['worker_id']} | **Subtask:** {row['subtask_id']}")
            output.append(f"**Timestamp:** {row['timestamp']}")
            output.append(f"\n**Content:**\n{row['content']}")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Get finding failed: {e}")
            return f"## Error\n\n**Error:** {str(e)}"
    
    def get_findings_by_subtask(self, subtask_id: int) -> str:
        """
        Get all findings associated with a specific subtask.
        
        Args:
            subtask_id: The subtask ID to filter by
        
        Returns:
            str: All findings for the subtask
        """
        logger.info(f"Getting findings for subtask: {subtask_id}")
        
        try:
            # Get all records and filter
            df = self.table.to_pandas()
            findings = df[(df["subtask_id"] == subtask_id) & (df["id"] != "init")]
            
            if len(findings) == 0:
                return f"## Subtask {subtask_id} Findings\n\nNo findings for this subtask."
            
            output = [f"## Subtask {subtask_id} Findings\n"]
            output.append(f"**Total findings:** {len(findings)}\n")
            
            for _, row in findings.iterrows():
                output.append(f"### [{row['id']}] {row['source_title'] or 'Untitled'}")
                output.append(f"**Source:** {row['source_url']}")
                output.append(f"**Verified:** {'✅' if row['verified'] else '❌'}")
                output.append(f"**Content:** {row['content'][:300]}...")
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Get findings by subtask failed: {e}")
            return f"## Error\n\n**Error:** {str(e)}"
    
    def get_findings_index(self) -> str:
        """
        Get a compact index of all findings for report planning.
        
        Returns a summary with statistics, source list, and suggested search topics.
        Use this to understand what research is available before writing a report.
        Then use `search_knowledge(topic)` to get detailed findings for each section.
        
        Returns:
            str: Formatted index of all findings with statistics and topics
        
        Example:
            >>> index = tools.get_findings_index()
            >>> # Review index to plan report sections
            >>> # Then search for specific topics:
            >>> findings = tools.search_knowledge("multi-agent coordination", top_k=15)
        """
        logger.info("Generating findings index for report planning")
        
        try:
            df = self.table.to_pandas()
            df = df[df["id"] != "init"]
            
            if len(df) == 0:
                return "## Research Findings Index\n\nNo findings in knowledge base."
            
            # Calculate statistics
            academic_df = df[df["search_type"] == "academic"]
            general_df = df[df["search_type"] != "academic"]
            verified_count = len(df[df["verified"] == True])
            
            # Get unique sources
            sources = df[["source_title", "source_url", "search_type", "verified"]].drop_duplicates(subset=["source_url"])
            academic_sources = sources[sources["search_type"] == "academic"]
            general_sources = sources[sources["search_type"] != "academic"]
            
            # Build index
            output = [
                "## Research Findings Index\n",
                "Use this index to plan your report, then use `search_knowledge(topic)` to get detailed findings.\n",
                "---",
                "",
                "### Statistics",
                "",
                f"- **Total Findings:** {len(df)}",
                f"- **Academic Findings:** {len(academic_df)}",
                f"- **General Findings:** {len(general_df)}",
                f"- **Verified Sources:** {verified_count}",
                f"- **Unique Sources:** {len(sources)}",
                "",
                "---",
                "",
                "### Academic Sources ({} sources)\n".format(len(academic_sources)),
            ]
            
            # List academic sources
            for _, row in academic_sources.iterrows():
                verified_icon = "✅" if row["verified"] else "❌"
                title = row["source_title"] or "Untitled"
                output.append(f"- {verified_icon} **{title}**")
                output.append(f"  {row['source_url']}")
            
            output.append("")
            output.append("### General Sources ({} sources)\n".format(len(general_sources)))
            
            # List general sources
            for _, row in general_sources.iterrows():
                verified_icon = "✅" if row["verified"] else "❌"
                title = row["source_title"] or "Untitled"
                output.append(f"- {verified_icon} **{title}**")
                output.append(f"  {row['source_url']}")
            
            output.append("")
            output.append("---")
            output.append("")
            output.append("### Suggested Search Topics")
            output.append("")
            output.append("Use `search_knowledge(topic, top_k=15)` to retrieve findings for each section:")
            output.append("")
            output.append("1. **Agent Architectures**: `search_knowledge(\"AI agent architecture components\")`")
            output.append("2. **Multi-Agent Systems**: `search_knowledge(\"multi-agent coordination collaboration\")`")
            output.append("3. **Reasoning & Planning**: `search_knowledge(\"reasoning planning capabilities LLM\")`")
            output.append("4. **Tool Use**: `search_knowledge(\"tool use function calling agents\")`")
            output.append("5. **Memory Systems**: `search_knowledge(\"agent memory knowledge representation\")`")
            output.append("6. **Benchmarks**: `search_knowledge(\"benchmarks evaluation performance metrics\")`")
            output.append("7. **Applications**: `search_knowledge(\"applications real-world deployment\")`")
            output.append("8. **Frameworks**: `search_knowledge(\"frameworks libraries implementation\")`")
            output.append("")
            output.append("---")
            output.append("")
            output.append("### Sample Content Preview")
            output.append("")
            output.append("First 3 findings (use search to get more):")
            output.append("")
            
            # Show preview of first 3 findings
            for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
                output.append(f"**{i}. {row['source_title'] or 'Untitled'}** ({row['search_type']})")
                output.append(f"   {row['content'][:300]}...")
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Get findings index failed: {e}")
            return f"## Error\n\n**Error:** {str(e)}"

    def clear_database(self) -> str:
        """
        Clear all findings from the knowledge base.
        
        This removes all stored findings and resets the table to its initial state.
        Useful for cleaning up between research runs.
        
        Returns:
            str: Confirmation message with count of cleared findings
        """
        logger.info("Clearing knowledge base...")
        
        try:
            # Get count before clearing
            df = self.table.to_pandas()
            # Exclude init record from count
            findings_count = len(df[df["id"] != "init"])
            
            if findings_count == 0:
                logger.info("Knowledge base already empty")
                return "Knowledge base already empty."
            
            # Drop and recreate the table
            table_name = "findings"
            self.db.drop_table(table_name)
            self._table = None  # Reset cached table reference
            
            # Recreate with initial record
            initial_data = [{
                "id": "init",
                "content": "Initialization record",
                "source_url": "",
                "source_title": "",
                "subtask_id": 0,
                "worker_id": "system",
                "timestamp": datetime.utcnow().isoformat(),
                "verified": False,
                "search_type": "init",
                "quality_score": 0.0,
                "vector": [0.0] * self.embedding_dimensions,
            }]
            self._table = self.db.create_table(table_name, data=initial_data)
            
            logger.info(f"Cleared {findings_count} findings from knowledge base")
            return f"✅ Cleared {findings_count} findings from knowledge base."
            
        except Exception as e:
            logger.error(f"Clear database failed: {e}")
            return f"❌ Error clearing database: {str(e)}"


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Knowledge Tools Test ===\n")
    
    # Check API configuration
    litellm_base = os.getenv("LITELLM_API_BASE")
    litellm_key = os.getenv("LITELLM_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if litellm_base and litellm_key:
        print(f"✅ Using LiteLLM proxy: {litellm_base}")
    elif openai_key:
        print(f"✅ Using direct OpenAI API")
    else:
        print("❌ No embedding API key found")
        print("   Set LITELLM_API_KEY + LITELLM_API_BASE (recommended)")
        print("   Or set OPENAI_API_KEY for direct OpenAI")
        exit(1)
    
    # Initialize tools with test path
    tools = KnowledgeTools(db_path="./test_research_kb")
    print(f"✅ Embedding model: {tools.embedding_model}")
    print(f"✅ Dimensions: {tools.embedding_dimensions}")
    
    # Test save_finding
    print("\n--- Save Finding Test ---")
    result = tools.save_finding(
        content="LLMs have shown remarkable capability in zero-shot reasoning tasks, achieving human-level performance on many benchmarks.",
        source_url="https://arxiv.org/abs/2301.00001",
        source_title="Large Language Model Survey",
        search_type="academic",
        verified=True,
        subtask_id=1,
        worker_id="test_worker",
    )
    print(result)
    
    # Test search
    print("\n--- Search Test ---")
    result = tools.search_knowledge("language model reasoning")
    print(result)
    
    # Test list sources
    print("\n--- List Sources Test ---")
    result = tools.list_sources()
    print(result)

