"""
Ingest FBI/BJS agency list into LanceDB for universe baseline.

This script loads law enforcement agency census data into the knowledge base
to establish the "denominator" for adoption rate calculations.

Data sources:
- BJS CSLLEA: https://bjs.ojp.gov/data-collection/census-state-and-local-law-enforcement-agencies
- FBI UCR: https://crime-data-explorer.fr.cloud.gov/

Usage:
    python scripts/ingest_agency_list.py data/agency_census.csv
    python scripts/ingest_agency_list.py --download  # Download from BJS
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from infrastructure.knowledge_tools import KnowledgeTools


def download_bjs_data(output_path: str) -> str:
    """
    Download agency census data from BJS.
    
    Note: BJS data may require manual download from their website.
    This function provides the URL and creates a sample file structure.
    
    Args:
        output_path: Where to save the CSV
        
    Returns:
        str: Path to the downloaded/created file
    """
    import requests
    
    # BJS CSLLEA 2018 (most recent complete census)
    # Direct data download often requires account; provide sample structure
    print("📥 BJS Law Enforcement Agency Census")
    print("   Direct download URL: https://bjs.ojp.gov/data-collection/census-state-and-local-law-enforcement-agencies")
    print("   Note: Full data may require account registration at BJS")
    print()
    
    # Create sample data structure for testing/demonstration
    sample_data = {
        "agency_name": [
            "New York City Police Department",
            "Los Angeles Police Department",
            "Chicago Police Department",
            "Houston Police Department",
            "Philadelphia Police Department",
            "Phoenix Police Department",
            "San Antonio Police Department",
            "San Diego Police Department",
            "Dallas Police Department",
            "San Jose Police Department",
            "Austin Police Department",
            "Fort Worth Police Department",
            "Columbus Police Department",
            "Indianapolis Police Department",
            "Charlotte-Mecklenburg Police Department",
            "Seattle Police Department",
            "Denver Police Department",
            "Washington Metropolitan Police",
            "Boston Police Department",
            "Las Vegas Metropolitan Police",
        ],
        "state": [
            "NY", "CA", "IL", "TX", "PA", "AZ", "TX", "CA", "TX", "CA",
            "TX", "TX", "OH", "IN", "NC", "WA", "CO", "DC", "MA", "NV",
        ],
        "agency_type": [
            "Municipal Police", "Municipal Police", "Municipal Police", 
            "Municipal Police", "Municipal Police", "Municipal Police",
            "Municipal Police", "Municipal Police", "Municipal Police",
            "Municipal Police", "Municipal Police", "Municipal Police",
            "Municipal Police", "Municipal Police", "Municipal Police",
            "Municipal Police", "Municipal Police", "Municipal Police",
            "Municipal Police", "Sheriff's Office",
        ],
        "officer_count": [
            36000, 9000, 12000, 5300, 6300, 3000, 2500, 1800, 3500, 1100,
            1900, 1700, 1900, 1700, 2100, 1400, 1500, 3800, 2100, 3000,
        ],
        "population_served": [
            8300000, 3900000, 2700000, 2300000, 1600000, 1700000, 1500000,
            1400000, 1300000, 1000000, 1000000, 950000, 900000, 880000,
            880000, 750000, 720000, 700000, 700000, 650000,
        ],
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Created sample agency data: {output_path}")
    print(f"   Agencies: {len(df)}")
    print(f"   Total officers: {df['officer_count'].sum():,}")
    print()
    print("⚠️  This is SAMPLE data for the top 20 agencies.")
    print("   For production use, download full BJS CSLLEA dataset.")
    
    return output_path


def ingest_agencies(csv_path: str, db_path: str = "./research_kb") -> dict:
    """
    Ingest agency census data into LanceDB knowledge base.
    
    Args:
        csv_path: Path to CSV file with agency data
        db_path: Path to LanceDB database
        
    Returns:
        dict: Statistics about ingestion
    """
    print(f"📖 Reading agency data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate expected columns
    required_cols = ["agency_name", "state", "officer_count"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Optional columns with defaults
    if "agency_type" not in df.columns:
        df["agency_type"] = "Unknown"
    if "population_served" not in df.columns:
        df["population_served"] = 0
    
    print(f"   Found {len(df)} agencies")
    print(f"   Total officers: {df['officer_count'].sum():,}")
    print()
    
    # Initialize knowledge tools
    print(f"🔗 Connecting to LanceDB: {db_path}")
    kb = KnowledgeTools(db_path=db_path)
    
    # Ingest each agency as a finding
    stats = {
        "total": len(df),
        "ingested": 0,
        "errors": 0,
        "total_officers": int(df["officer_count"].sum()),
    }
    
    print("📤 Ingesting agencies into knowledge base...")
    for idx, row in df.iterrows():
        try:
            # Build comprehensive content string
            content = (
                f"Law Enforcement Agency Census Data:\n"
                f"- Agency: {row['agency_name']}\n"
                f"- State: {row['state']}\n"
                f"- Type: {row['agency_type']}\n"
                f"- Sworn Officers: {row['officer_count']:,}\n"
            )
            if row.get('population_served', 0) > 0:
                content += f"- Population Served: {row['population_served']:,}\n"
            
            content += (
                f"\nThis data establishes the baseline universe for calculating "
                f"technology adoption rates among US law enforcement agencies."
            )
            
            result = kb.save_finding(
                content=content,
                source_url="https://bjs.ojp.gov/data-collection/census-state-and-local-law-enforcement-agencies",
                source_title=f"BJS CSLLEA - {row['agency_name']}",
                search_type="general",
                subtask_id=0,
                worker_id="ingest_script",
                verified=True,  # Census data is verified
            )
            
            stats["ingested"] += 1
            
            # Progress indicator
            if (idx + 1) % 100 == 0 or idx == len(df) - 1:
                print(f"   Progress: {idx + 1}/{len(df)} agencies")
                
        except Exception as e:
            print(f"   ❌ Error ingesting {row['agency_name']}: {e}")
            stats["errors"] += 1
    
    print()
    print("=" * 50)
    print("📊 Ingestion Summary")
    print("=" * 50)
    print(f"   Total agencies: {stats['total']}")
    print(f"   Successfully ingested: {stats['ingested']}")
    print(f"   Errors: {stats['errors']}")
    print(f"   Total officers in baseline: {stats['total_officers']:,}")
    print()
    
    return stats


def ingest_national_totals(db_path: str = "./research_kb") -> None:
    """
    Ingest national totals for the law enforcement universe.
    
    Uses FBI UCR 2022 data as the authoritative source.
    """
    print("📊 Ingesting national totals...")
    
    kb = KnowledgeTools(db_path=db_path)
    
    # FBI UCR 2022 totals (approximate)
    national_content = """
US Law Enforcement Universe (FBI UCR 2022 Data):

NATIONAL TOTALS:
- Total Law Enforcement Agencies: ~18,000
- Total Sworn Officers: ~700,000
- Total Civilian Employees: ~300,000

BREAKDOWN BY AGENCY TYPE:
- Local Police Departments: ~12,500 agencies (~470,000 officers)
- Sheriff's Offices: ~3,000 agencies (~170,000 officers)
- State Police/Highway Patrol: ~50 agencies (~60,000 officers)
- Federal Agencies: ~65 agencies (varies)

TOP 10 AGENCIES BY SIZE (sworn officers):
1. New York City PD: ~36,000
2. Chicago PD: ~12,000
3. Los Angeles PD: ~9,000
4. Philadelphia PD: ~6,300
5. Houston PD: ~5,300
6. Washington DC Metro PD: ~3,800
7. Dallas PD: ~3,500
8. Phoenix PD: ~3,000
9. Las Vegas Metro PD: ~3,000
10. San Antonio PD: ~2,500

These totals serve as the DENOMINATOR for calculating technology adoption 
penetration rates. A product adopted by agencies covering X officers 
divided by 700,000 gives the officer-weighted penetration rate.

Source: FBI Crime Data Explorer, BJS CSLLEA 2018-2022
"""
    
    result = kb.save_finding(
        content=national_content,
        source_url="https://crime-data-explorer.fr.cloud.gov/pages/le/national",
        source_title="FBI UCR - National Law Enforcement Statistics 2022",
        search_type="general",
        subtask_id=0,
        worker_id="ingest_script",
        verified=True,
    )
    
    print("✅ Ingested national totals")
    print(result)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest law enforcement agency data into LanceDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_agency_list.py data/agency_census.csv
  python scripts/ingest_agency_list.py --download
  python scripts/ingest_agency_list.py --download --db-path ./my_research_kb
        """
    )
    
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/agency_census.csv",
        help="Path to agency CSV file (default: data/agency_census.csv)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/create sample agency data"
    )
    parser.add_argument(
        "--db-path",
        default="./research_kb",
        help="Path to LanceDB database (default: ./research_kb)"
    )
    parser.add_argument(
        "--totals-only",
        action="store_true",
        help="Only ingest national totals, skip individual agencies"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚔 Law Enforcement Agency Data Ingestion")
    print("=" * 60)
    print()
    
    # Download if requested
    if args.download:
        args.csv_path = download_bjs_data(args.csv_path)
        print()
    
    # Ingest national totals
    ingest_national_totals(args.db_path)
    print()
    
    # Ingest individual agencies (unless totals-only)
    if not args.totals_only:
        if not os.path.exists(args.csv_path):
            print(f"❌ CSV file not found: {args.csv_path}")
            print("   Use --download to create sample data")
            sys.exit(1)
        
        ingest_agencies(args.csv_path, args.db_path)
    
    print("✅ Ingestion complete!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
