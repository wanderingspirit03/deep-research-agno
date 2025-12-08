"""
Data Analyst Agent - Structures research findings into adoption data and computes metrics.

The Data Analyst Agent:
1. Parses unstructured research findings to extract agency adoption evidence
2. Maps findings to structured AgencyAdoption records
3. Computes penetration metrics against the agency universe
4. Identifies patterns in adoption data (geographic, temporal, by agency type)
5. Generates structured output for reporting
"""
import os
import csv
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from textwrap import dedent

import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.utils.log import logger

from pydantic import BaseModel, Field

from infrastructure.knowledge_tools import KnowledgeTools
from infrastructure.observability import observe
from .schemas import (
    AgencyAdoption,
    AdoptionEvidence,
    AdoptionStatus,
    AgencyType,
    EvidenceSourceType,
    PenetrationMetrics,
)


# =============================================================================
# Structured Output Schemas for LLM
# =============================================================================

class ExtractedEvidence(BaseModel):
    """Evidence extracted from a finding"""
    agency_name: str = Field(..., description="Name of the agency mentioned")
    state: str = Field(..., description="US state code (2 letters)")
    status: str = Field(
        default="no_data",
        description="One of: confirmed, probable, pilot, not_adopted, no_data"
    )
    evidence_type: str = Field(
        default="other",
        description="Type: police_report, procurement_contract, council_minutes, news_article, etc."
    )
    evidence_url: str = Field(default="", description="Source URL")
    evidence_excerpt: str = Field(default="", description="Relevant excerpt from source")
    has_ai_disclaimer: bool = Field(default=False, description="Contains AI/DraftOne disclaimer")
    disclaimer_text: Optional[str] = Field(default=None, description="Exact disclaimer text if found")
    confidence: float = Field(default=0.5, description="Confidence score 0-1")
    adoption_date: Optional[str] = Field(default=None, description="Date of adoption if mentioned")


class ExtractionResult(BaseModel):
    """Result of evidence extraction from multiple findings"""
    agencies: List[ExtractedEvidence] = Field(default_factory=list)
    extraction_notes: str = Field(default="", description="Notes about the extraction process")


# =============================================================================
# Agency Census Loader
# =============================================================================

def load_agency_census(csv_path: str = "./data/agency_census.csv") -> Dict[str, Dict[str, Any]]:
    """
    Load agency census data from CSV file.
    
    Returns a dict keyed by "{agency_name}|{state}" for easy lookup.
    """
    agencies = {}
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['agency_name'].strip()}|{row['state'].strip()}"
                agencies[key] = {
                    "agency_name": row["agency_name"].strip(),
                    "state": row["state"].strip(),
                    "agency_type": row.get("agency_type", "Municipal Police"),
                    "officer_count": int(row["officer_count"]) if row.get("officer_count") else None,
                    "population_served": int(row["population_served"]) if row.get("population_served") else None,
                }
        logger.info(f"Loaded {len(agencies)} agencies from census")
    except FileNotFoundError:
        logger.warning(f"Agency census file not found: {csv_path}")
    except Exception as e:
        logger.error(f"Error loading agency census: {e}")
    
    return agencies


# =============================================================================
# AI Disclaimer Detection Patterns
# =============================================================================

# Patterns that indicate AI-assisted report writing
AI_DISCLAIMER_PATTERNS = [
    # Axon DraftOne specific
    r"axon\s*draftone",
    r"draft\s*one",
    r"axon\s*draft\s*one",
    r"draftone\s*ai",
    
    # General AI report disclaimers
    r"drafted\s+using\s+(?:axon\s+)?(?:ai|artificial\s+intelligence)",
    r"ai[- ]assisted\s+report",
    r"ai[- ]powered\s+report",
    r"prepared\s+with\s+assistance\s+from\s+ai",
    r"ai[- ]generated\s+(?:draft|report)",
    r"automated\s+draft",
    r"artificial\s+intelligence\s+(?:assisted|generated|drafted)",
    
    # Officer responsibility disclaimers
    r"officer\s+(?:is\s+)?responsible\s+for\s+(?:verifying|accuracy)",
    r"reviewed\s+(?:and\s+)?(?:approved\s+)?by\s+(?:the\s+)?officer",
    
    # Axon AI Era Plan
    r"axon\s+ai\s+era\s+plan",
    r"ai\s+era\s+plan",
]

# Compile patterns for efficiency
DISCLAIMER_REGEX = re.compile(
    "|".join(AI_DISCLAIMER_PATTERNS),
    re.IGNORECASE
)


def detect_ai_disclaimer(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect AI disclaimer in text.
    
    Returns:
        Tuple of (has_disclaimer, matched_text)
    """
    match = DISCLAIMER_REGEX.search(text)
    if match:
        # Get surrounding context (50 chars before/after)
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end].strip()
        return True, context
    return False, None


# =============================================================================
# Data Analyst Agent
# =============================================================================

class DataAnalystAgent:
    """
    Data Analyst Agent - Extracts structured adoption data from research findings.
    
    Responsibilities:
    1. Parse unstructured findings to identify agency mentions
    2. Classify adoption status based on evidence
    3. Map to AgencyAdoption schema
    4. Compute penetration metrics
    5. Identify temporal patterns in adoption
    """
    
    def __init__(
        self,
        model_id: str = "openai/claude-opus-4-5-20251101",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,  # Low for consistent extraction
        knowledge_tools: Optional[KnowledgeTools] = None,
        census_path: str = "./data/agency_census.csv",
    ):
        """
        Initialize Data Analyst Agent.
        
        Args:
            model_id: LLM model ID
            api_base: LiteLLM API base URL
            api_key: LiteLLM API key
            temperature: Model temperature (lower = more consistent)
            knowledge_tools: Knowledge base toolkit
            census_path: Path to agency census CSV
        """
        self.model_id = model_id
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.temperature = temperature
        
        # Knowledge tools for accessing findings
        self.knowledge_tools = knowledge_tools or KnowledgeTools()
        
        # Load agency universe
        self.agency_census = load_agency_census(census_path)
        
        # Structured adoption records
        self.adoption_records: Dict[str, AgencyAdoption] = {}
        
        # Initialize records from census
        self._initialize_from_census()
        
        # Lazy-initialized agent
        self._agent: Optional[Agent] = None
    
    def _initialize_from_census(self):
        """Initialize adoption records from agency census"""
        for key, data in self.agency_census.items():
            agency_type = self._parse_agency_type(data.get("agency_type", ""))
            
            self.adoption_records[key] = AgencyAdoption(
                agency_name=data["agency_name"],
                state=data["state"],
                agency_type=agency_type,
                officer_count=data.get("officer_count"),
                population_served=data.get("population_served"),
                status=AdoptionStatus.NO_DATA,
            )
        
        logger.info(f"Initialized {len(self.adoption_records)} agency records from census")
    
    def _parse_agency_type(self, type_str: str) -> AgencyType:
        """Parse agency type string to enum"""
        type_map = {
            "municipal police": AgencyType.MUNICIPAL_POLICE,
            "county sheriff": AgencyType.COUNTY_SHERIFF,
            "sheriff's office": AgencyType.COUNTY_SHERIFF,
            "state police": AgencyType.STATE_POLICE,
            "highway patrol": AgencyType.HIGHWAY_PATROL,
            "federal": AgencyType.FEDERAL,
        }
        type_lower = type_str.lower()
        for key, value in type_map.items():
            if key in type_lower:
                return value
        return AgencyType.OTHER
    
    @property
    def agent(self) -> Agent:
        """Lazy initialization of Agno Agent"""
        if self._agent is None:
            if not self.api_base or not self.api_key:
                raise ValueError("LITELLM_API_BASE and LITELLM_API_KEY must be set")
            
            logger.info(f"Creating Data Analyst Agent with model: {self.model_id}")
            
            self._agent = Agent(
                name="Data Analyst",
                model=LiteLLM(
                    id=self.model_id,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    top_p=None,
                ),
                description="Expert data analyst that extracts structured adoption data from research findings.",
                instructions=self._get_instructions(),
                output_schema=ExtractionResult,
                markdown=True,
                debug_mode=True,
                debug_level=2,
            )
        
        return self._agent
    
    def _get_instructions(self) -> List[str]:
        """Get agent instructions for data extraction"""
        return [
            dedent("""
                You are an expert data analyst specializing in extracting structured adoption 
                data from unstructured research findings about law enforcement technology adoption.
                
                ## Your Mission
                
                Extract structured agency adoption records from research findings about 
                Axon DraftOne / AI Era Plan adoption in US law enforcement.
                
                ## What to Extract
                
                For each agency mentioned in the findings, extract:
                
                1. **Agency Name**: Full official name (e.g., "Seattle Police Department")
                2. **State**: 2-letter state code (e.g., "WA")
                3. **Adoption Status**: One of:
                   - `confirmed`: Clear, unambiguous evidence of active use
                     * Police reports with DraftOne disclaimers
                     * Official announcements of deployment
                     * Contracts specifying AI Era Plan
                   - `probable`: Strong evidence but not explicit
                     * AI disclaimers without Axon name
                     * Budget allocation for "AI report writing"
                   - `pilot`: Limited/trial deployment
                     * "Pilot program" or "trial" language
                     * Limited number of officers using
                   - `not_adopted`: Confirmed not using
                     * Explicit rejection statements
                     * Competing product adoption
                   - `no_data`: Insufficient evidence
                
                4. **Evidence Type**: Source of evidence
                   - `police_report`: Incident/arrest reports with disclaimers
                   - `procurement_contract`: RFPs, contracts, POs
                   - `council_minutes`: Council/board meeting records
                   - `budget_document`: Fiscal documents
                   - `press_release`: Official announcements
                   - `news_article`: News coverage
                   - `policy_memo`: Internal policies
                   - `axon_announcement`: Axon's press releases
                   - `other`: Other sources
                
                5. **AI Disclaimer**: If police report contains AI disclaimer
                   - Set `has_ai_disclaimer: true`
                   - Extract exact `disclaimer_text` if visible
                
                6. **Confidence Score**: 0.0 to 1.0
                   - 0.9-1.0: Direct, unambiguous evidence
                   - 0.7-0.9: Strong indirect evidence
                   - 0.5-0.7: Moderate evidence
                   - Below 0.5: Weak/uncertain
                
                7. **Adoption Date**: If mentioned (ISO format: YYYY-MM-DD)
                
                ## Classification Guidelines
                
                ### Confirmed (highest bar)
                - Must have explicit mention of "Axon DraftOne" or "AI Era Plan"
                - OR police reports with distinctive AI disclaimers
                - OR signed contracts/POs for AI Era Plan
                
                ### Probable
                - AI report disclaimers without specific product name
                - Budget items for "AI-assisted report writing" with Axon
                - News articles describing active use without official confirmation
                
                ### Pilot
                - Language like "testing", "trial", "pilot program"
                - Limited deployment (e.g., "select officers")
                - Time-limited mentions
                
                ### Not Adopted
                - Explicit statements of rejection
                - Adoption of competing product (e.g., Truleo)
                - City council vote against
                
                ## Important Notes
                
                - Be conservative: when in doubt, use lower-confidence classification
                - Don't infer agency names - only extract explicitly mentioned agencies
                - Include ALL agencies mentioned, even if evidence is weak
                - Preserve exact wording of disclaimers when found
                - Note if agency is NOT in the standard census list (new/small agencies)
            """).strip(),
        ]
    
    @observe(name="analyst.structure_findings")
    def structure_findings(
        self,
        findings: Optional[List[Dict[str, Any]]] = None,
        max_findings: int = 100,
    ) -> Dict[str, AgencyAdoption]:
        """
        Structure raw findings into AgencyAdoption records.
        
        Args:
            findings: List of finding dicts (if None, retrieves from KB)
            max_findings: Maximum findings to process
            
        Returns:
            Dict mapping agency keys to AgencyAdoption records
        """
        logger.info("Structuring findings into adoption records...")
        
        # Get findings if not provided
        if findings is None:
            findings = self._get_findings_from_kb(max_findings)
        
        if not findings:
            logger.warning("No findings to structure")
            return self.adoption_records
        
        logger.info(f"Processing {len(findings)} findings")
        
        # First pass: Pattern-based extraction (fast, no LLM)
        self._pattern_based_extraction(findings)
        
        # Second pass: LLM-based extraction (more thorough)
        self._llm_based_extraction(findings)
        
        # Update search status for agencies with evidence
        self._mark_searched_agencies()
        
        logger.info(f"Structured {len(self.adoption_records)} agency records")
        return self.adoption_records
    
    def _get_findings_from_kb(self, max_findings: int) -> List[Dict[str, Any]]:
        """Retrieve findings from knowledge base"""
        try:
            df = self.knowledge_tools.table.to_pandas()
            df = df[df["id"] != "init"]
            
            findings = []
            for _, row in df.head(max_findings).iterrows():
                findings.append({
                    "id": row.get("id", ""),
                    "content": row.get("content", ""),
                    "source_url": row.get("source_url", ""),
                    "source_title": row.get("source_title", ""),
                    "search_type": row.get("search_type", "general"),
                })
            
            return findings
        except Exception as e:
            logger.error(f"Error retrieving findings: {e}")
            return []
    
    def _pattern_based_extraction(self, findings: List[Dict[str, Any]]):
        """Fast pattern-based extraction of AI disclaimers"""
        logger.info("Running pattern-based extraction...")
        
        for finding in findings:
            content = finding.get("content", "")
            source_url = finding.get("source_url", "")
            source_title = finding.get("source_title", "")
            
            # Check for AI disclaimer
            has_disclaimer, disclaimer_text = detect_ai_disclaimer(content)
            
            if has_disclaimer:
                logger.info(f"AI disclaimer detected in: {source_title[:50]}...")
                
                # Try to extract agency name from context
                agency_match = self._extract_agency_from_context(content, source_title)
                
                if agency_match:
                    agency_name, state = agency_match
                    key = f"{agency_name}|{state}"
                    
                    # Update or create record
                    if key in self.adoption_records:
                        record = self.adoption_records[key]
                    else:
                        record = AgencyAdoption(
                            agency_name=agency_name,
                            state=state,
                            status=AdoptionStatus.NO_DATA,
                        )
                        self.adoption_records[key] = record
                    
                    # Add evidence
                    evidence = AdoptionEvidence(
                        evidence_type=EvidenceSourceType.POLICE_REPORT,
                        url=source_url,
                        title=source_title,
                        excerpt=content[:500],
                        has_ai_disclaimer=True,
                        disclaimer_text=disclaimer_text,
                        confidence_score=0.9 if "draftone" in content.lower() else 0.7,
                    )
                    record.add_evidence(evidence)
                    
                    # Update status
                    if "draftone" in content.lower() or "draft one" in content.lower():
                        record.status = AdoptionStatus.CONFIRMED
                        record.status_confidence = 0.95
                    else:
                        # AI disclaimer but not explicit DraftOne
                        if record.status in [AdoptionStatus.NO_DATA, AdoptionStatus.NOT_ADOPTED]:
                            record.status = AdoptionStatus.PROBABLE
                            record.status_confidence = 0.7
    
    def _extract_agency_from_context(self, content: str, title: str) -> Optional[Tuple[str, str]]:
        """Extract agency name and state from content context"""
        combined = f"{title} {content}"
        
        # Look for patterns like "City Police Department" + state
        agency_patterns = [
            r"([A-Z][a-zA-Z\s]+(?:Police Department|Sheriff'?s? Office|PD))",
            r"([A-Z][a-zA-Z\s]+(?:Police|Sheriff))",
        ]
        
        state_pattern = r"\b([A-Z]{2})\b"
        
        for pattern in agency_patterns:
            match = re.search(pattern, combined)
            if match:
                agency_name = match.group(1).strip()
                
                # Find state near agency mention
                state_matches = re.findall(state_pattern, combined)
                if state_matches:
                    # Use first valid state code
                    valid_states = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                                   "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                                   "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                                   "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                                   "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"}
                    for state in state_matches:
                        if state in valid_states:
                            return (agency_name, state)
        
        return None
    
    def _llm_based_extraction(self, findings: List[Dict[str, Any]]):
        """LLM-based extraction for complex cases"""
        logger.info("Running LLM-based extraction...")
        
        # Batch findings for efficiency
        batch_size = 10
        for i in range(0, len(findings), batch_size):
            batch = findings[i:i + batch_size]
            
            # Format findings for LLM
            findings_text = self._format_findings_for_extraction(batch)
            
            prompt = f"""
Extract structured agency adoption data from these research findings about 
Axon DraftOne / AI Era Plan adoption in US law enforcement.

**Research Findings:**
{findings_text}

---

Extract ALL agencies mentioned with their adoption status and evidence.
Be thorough - include every agency mentioned, even with weak evidence.
            """.strip()
            
            try:
                response = self.agent.run(prompt)
                
                if isinstance(response.content, ExtractionResult):
                    result = response.content
                elif isinstance(response.content, dict):
                    result = ExtractionResult(**response.content)
                else:
                    logger.warning("Could not parse extraction result")
                    continue
                
                # Process extracted agencies
                for extracted in result.agencies:
                    self._process_extracted_evidence(extracted)
                    
            except Exception as e:
                logger.error(f"LLM extraction error: {e}")
                continue
    
    def _format_findings_for_extraction(self, findings: List[Dict[str, Any]]) -> str:
        """Format findings for LLM extraction prompt"""
        lines = []
        for i, f in enumerate(findings, 1):
            lines.append(f"### Finding {i}")
            lines.append(f"**Source:** {f.get('source_title', 'Unknown')}")
            lines.append(f"**URL:** {f.get('source_url', '')}")
            lines.append(f"**Content:**")
            lines.append(f.get('content', '')[:2000])
            lines.append("")
        return "\n".join(lines)
    
    def _process_extracted_evidence(self, extracted: ExtractedEvidence):
        """Process extracted evidence and update adoption records"""
        key = f"{extracted.agency_name}|{extracted.state}"
        
        # Get or create record
        if key in self.adoption_records:
            record = self.adoption_records[key]
        else:
            record = AgencyAdoption(
                agency_name=extracted.agency_name,
                state=extracted.state,
                status=AdoptionStatus.NO_DATA,
            )
            self.adoption_records[key] = record
        
        # Parse evidence type
        try:
            evidence_type = EvidenceSourceType(extracted.evidence_type)
        except ValueError:
            evidence_type = EvidenceSourceType.OTHER
        
        # Add evidence
        evidence = AdoptionEvidence(
            evidence_type=evidence_type,
            url=extracted.evidence_url,
            title="",
            excerpt=extracted.evidence_excerpt,
            has_ai_disclaimer=extracted.has_ai_disclaimer,
            disclaimer_text=extracted.disclaimer_text,
            confidence_score=extracted.confidence,
            date_found=extracted.adoption_date,
        )
        record.add_evidence(evidence)
        
        # Update status based on extraction
        status_map = {
            "confirmed": AdoptionStatus.CONFIRMED,
            "probable": AdoptionStatus.PROBABLE,
            "pilot": AdoptionStatus.PILOT,
            "not_adopted": AdoptionStatus.NOT_ADOPTED,
            "no_data": AdoptionStatus.NO_DATA,
        }
        new_status = status_map.get(extracted.status.lower(), AdoptionStatus.NO_DATA)
        
        # Only upgrade status (don't downgrade)
        status_order = {
            AdoptionStatus.NO_DATA: 0,
            AdoptionStatus.NOT_ADOPTED: 1,
            AdoptionStatus.PILOT: 2,
            AdoptionStatus.PROBABLE: 3,
            AdoptionStatus.CONFIRMED: 4,
        }
        
        if status_order[new_status] > status_order[record.status]:
            record.status = new_status
            record.status_confidence = extracted.confidence
        
        # Update adoption date
        if extracted.adoption_date and not record.first_adoption_date:
            record.first_adoption_date = extracted.adoption_date
    
    def _mark_searched_agencies(self):
        """Mark agencies as searched based on evidence presence"""
        for record in self.adoption_records.values():
            if record.evidence_count > 0:
                record.search_completed = True
                record.last_verified_date = datetime.utcnow().isoformat()
    
    @observe(name="analyst.compute_penetration")
    def compute_penetration(self) -> PenetrationMetrics:
        """
        Compute penetration metrics from adoption records.
        
        Returns:
            PenetrationMetrics with aggregate statistics
        """
        logger.info("Computing penetration metrics...")
        
        metrics = PenetrationMetrics()
        
        for record in self.adoption_records.values():
            metrics.total_agencies += 1
            metrics.total_officers += record.officer_count or 0
            
            if record.status == AdoptionStatus.CONFIRMED:
                metrics.confirmed_count += 1
                metrics.officers_confirmed += record.officer_count or 0
            elif record.status == AdoptionStatus.PROBABLE:
                metrics.probable_count += 1
                metrics.officers_probable += record.officer_count or 0
            elif record.status == AdoptionStatus.PILOT:
                metrics.pilot_count += 1
            elif record.status == AdoptionStatus.NOT_ADOPTED:
                metrics.not_adopted_count += 1
            else:
                metrics.no_data_count += 1
        
        logger.info(f"Penetration: {metrics.penetration_confirmed:.1f}% confirmed, "
                   f"{metrics.penetration_confirmed_plus_probable:.1f}% including probable")
        
        return metrics
    
    def get_adoption_by_state(self) -> Dict[str, Dict[str, int]]:
        """Get adoption counts broken down by state"""
        state_data = {}
        
        for record in self.adoption_records.values():
            state = record.state
            if state not in state_data:
                state_data[state] = {
                    "total": 0,
                    "confirmed": 0,
                    "probable": 0,
                    "pilot": 0,
                    "not_adopted": 0,
                    "no_data": 0,
                }
            
            state_data[state]["total"] += 1
            state_data[state][record.status.value] += 1
        
        return state_data
    
    def get_adoption_by_type(self) -> Dict[str, Dict[str, int]]:
        """Get adoption counts broken down by agency type"""
        type_data = {}
        
        for record in self.adoption_records.values():
            agency_type = record.agency_type.value
            if agency_type not in type_data:
                type_data[agency_type] = {
                    "total": 0,
                    "confirmed": 0,
                    "probable": 0,
                    "pilot": 0,
                    "not_adopted": 0,
                    "no_data": 0,
                }
            
            type_data[agency_type]["total"] += 1
            type_data[agency_type][record.status.value] += 1
        
        return type_data
    
    def export_to_csv(self, output_path: str = "./data/adoption_results.csv"):
        """Export adoption records to CSV"""
        logger.info(f"Exporting adoption records to: {output_path}")
        
        rows = []
        for record in self.adoption_records.values():
            rows.append({
                "agency_name": record.agency_name,
                "state": record.state,
                "agency_type": record.agency_type.value,
                "officer_count": record.officer_count,
                "population_served": record.population_served,
                "status": record.status.value,
                "status_confidence": record.status_confidence,
                "evidence_count": record.evidence_count,
                "first_adoption_date": record.first_adoption_date,
                "last_verified_date": record.last_verified_date,
                "search_completed": record.search_completed,
                "best_evidence_url": record.best_evidence_url,
            })
        
        # Sort by state then agency name
        rows.sort(key=lambda x: (x["state"], x["agency_name"]))
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Exported {len(rows)} records to CSV")
        return output_path
    
    def generate_summary_report(self) -> str:
        """Generate a text summary of penetration metrics"""
        metrics = self.compute_penetration()
        state_data = self.get_adoption_by_state()
        type_data = self.get_adoption_by_type()
        
        lines = [
            "# Axon DraftOne / AI Era Plan Penetration Analysis",
            "",
            "## Executive Summary",
            "",
            f"- **Total Agencies Analyzed:** {metrics.total_agencies:,}",
            f"- **Total Officers:** {metrics.total_officers:,}",
            "",
            "### Adoption Status",
            "",
            f"| Status | Agencies | % of Total |",
            f"|--------|----------|------------|",
            f"| Confirmed | {metrics.confirmed_count:,} | {metrics.penetration_confirmed:.1f}% |",
            f"| Probable | {metrics.probable_count:,} | {(metrics.probable_count/metrics.total_agencies*100) if metrics.total_agencies else 0:.1f}% |",
            f"| Pilot/Trial | {metrics.pilot_count:,} | {(metrics.pilot_count/metrics.total_agencies*100) if metrics.total_agencies else 0:.1f}% |",
            f"| Not Adopted | {metrics.not_adopted_count:,} | {(metrics.not_adopted_count/metrics.total_agencies*100) if metrics.total_agencies else 0:.1f}% |",
            f"| No Data | {metrics.no_data_count:,} | {(metrics.no_data_count/metrics.total_agencies*100) if metrics.total_agencies else 0:.1f}% |",
            "",
            "### Key Metrics",
            "",
            f"- **Penetration (Confirmed):** {metrics.penetration_confirmed:.1f}%",
            f"- **Penetration (Confirmed + Probable):** {metrics.penetration_confirmed_plus_probable:.1f}%",
            f"- **Officer Penetration (Confirmed):** {metrics.officer_penetration_confirmed:.1f}%",
            f"- **Research Coverage:** {metrics.coverage_rate:.1f}%",
            "",
            "## Breakdown by State",
            "",
        ]
        
        # Add state breakdown
        sorted_states = sorted(state_data.items(), key=lambda x: x[1]["confirmed"], reverse=True)
        lines.append("| State | Total | Confirmed | Probable | Pilot | Not Adopted |")
        lines.append("|-------|-------|-----------|----------|-------|-------------|")
        for state, data in sorted_states[:20]:  # Top 20 states
            lines.append(
                f"| {state} | {data['total']} | {data['confirmed']} | "
                f"{data['probable']} | {data['pilot']} | {data['not_adopted']} |"
            )
        
        lines.extend([
            "",
            "## Breakdown by Agency Type",
            "",
        ])
        
        # Add type breakdown
        lines.append("| Type | Total | Confirmed | Probable | Pilot |")
        lines.append("|------|-------|-----------|----------|-------|")
        for agency_type, data in type_data.items():
            lines.append(
                f"| {agency_type} | {data['total']} | {data['confirmed']} | "
                f"{data['probable']} | {data['pilot']} |"
            )
        
        lines.extend([
            "",
            "---",
            "",
            f"*Analysis generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
        ])
        
        return "\n".join(lines)


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Data Analyst Agent Test ===\n")
    
    # Check API config
    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not api_base or not api_key:
        print("❌ LITELLM_API_BASE and LITELLM_API_KEY must be set")
        exit(1)
    
    print(f"✅ API Base: {api_base}")
    
    # Test disclaimer detection
    print("\n--- AI Disclaimer Detection Test ---")
    test_texts = [
        "This report was drafted using Axon DraftOne, an AI-powered report-writing tool.",
        "The officer is responsible for verifying the accuracy of this AI-generated report.",
        "Standard police report with no AI involvement.",
        "Report prepared with assistance from AI. Officer reviewed and approved.",
    ]
    
    for text in test_texts:
        has_disclaimer, matched = detect_ai_disclaimer(text)
        status = "✅ DETECTED" if has_disclaimer else "❌ Not detected"
        print(f"{status}: {text[:50]}...")
        if matched:
            print(f"   Matched: {matched[:100]}")
    
    # Initialize analyst
    print("\n--- Analyst Initialization ---")
    analyst = DataAnalystAgent()
    print(f"✅ Loaded {len(analyst.adoption_records)} agencies from census")
    
    # Compute initial metrics (should all be no_data)
    metrics = analyst.compute_penetration()
    print(f"\n--- Initial Metrics ---")
    print(f"Total agencies: {metrics.total_agencies}")
    print(f"No data: {metrics.no_data_count}")
    print(f"Coverage: {metrics.coverage_rate:.1f}%")
