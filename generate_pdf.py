"""
Generate Beautiful PDF from Research Report

Converts markdown research reports to professionally-styled PDFs
with proper typography, headers, table of contents, and citations.

Primary method: weasyprint (requires system dependencies)
Fallback: fpdf2 (pure Python, no system deps)

For weasyprint:
    Install: pip install weasyprint markdown pygments
    System deps (macOS): brew install pango libffi glib
    
For fallback (simpler):
    Install: pip install fpdf2 markdown
"""
import os
import sys
import re
import argparse
from datetime import datetime

try:
    import markdown
    from markdown.extensions.toc import TocExtension
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Suppress weasyprint's stderr messages when it can't load system libs
import io as _io
_old_stderr = sys.stderr
sys.stderr = _io.StringIO()
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError, Exception):
    WEASYPRINT_AVAILABLE = False
    HTML = None
    CSS = None
finally:
    sys.stderr = _old_stderr
del _io, _old_stderr


# =============================================================================
# Professional CSS Styling for PDF
# =============================================================================

PDF_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+Pro:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;
    
    @top-center {
        content: string(chapter-title);
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 9pt;
        color: #666;
    }
    
    @bottom-center {
        content: counter(page);
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 10pt;
        color: #333;
    }
    
    @bottom-right {
        content: "Deep Research Swarm";
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 8pt;
        color: #888;
    }
}

@page :first {
    @top-center { content: none; }
    @bottom-right { content: none; }
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
}

/* Title Page Styling */
h1:first-of-type {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 28pt;
    font-weight: 700;
    color: #1a365d;
    text-align: center;
    margin-top: 3cm;
    margin-bottom: 0.5cm;
    line-height: 1.2;
    page-break-after: avoid;
    string-set: chapter-title content();
}

/* Subtitle/Tagline */
h1:first-of-type + p {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 14pt;
    font-style: italic;
    text-align: center;
    color: #4a5568;
    margin-bottom: 2cm;
}

/* Section Headers */
h2 {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #2c5282;
    margin-top: 1.5cm;
    margin-bottom: 0.5cm;
    padding-bottom: 0.3cm;
    border-bottom: 2px solid #bee3f8;
    page-break-after: avoid;
    string-set: chapter-title content();
}

h3 {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 14pt;
    font-weight: 600;
    color: #2d3748;
    margin-top: 1cm;
    margin-bottom: 0.4cm;
    page-break-after: avoid;
}

h4 {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 12pt;
    font-weight: 600;
    color: #4a5568;
    margin-top: 0.8cm;
    margin-bottom: 0.3cm;
}

/* Paragraphs */
p {
    margin-bottom: 0.4cm;
    text-indent: 0;
    orphans: 3;
    widows: 3;
}

/* Abstract Box */
h2#abstract + p,
h2:first-of-type + p {
    background: linear-gradient(135deg, #ebf8ff 0%, #f0fff4 100%);
    padding: 1cm;
    border-left: 4px solid #3182ce;
    font-size: 10.5pt;
    text-align: justify;
    margin: 0.5cm 0 1cm 0;
}

/* Table of Contents */
h2#table-of-contents + ol,
h2#table-of-contents + ul {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 11pt;
    column-count: 1;
    list-style: none;
    padding-left: 0;
}

h2#table-of-contents + ol li,
h2#table-of-contents + ul li {
    margin-bottom: 0.3cm;
    padding-left: 0;
}

h2#table-of-contents + ol ol,
h2#table-of-contents + ul ul {
    margin-left: 1cm;
    margin-top: 0.2cm;
}

/* Lists */
ul, ol {
    margin-left: 0.8cm;
    margin-bottom: 0.5cm;
}

li {
    margin-bottom: 0.2cm;
}

/* Citations in brackets */
p em:last-child,
li em:last-child {
    font-size: 9pt;
    color: #718096;
}

/* Inline Citations [Source] */
a[href^="#"] {
    color: #3182ce;
    text-decoration: none;
}

/* Horizontal Rules */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1cm 0;
}

/* Blockquotes */
blockquote {
    margin: 0.5cm 0 0.5cm 1cm;
    padding: 0.5cm 1cm;
    border-left: 3px solid #90cdf4;
    background: #f7fafc;
    font-style: italic;
    color: #4a5568;
}

/* Code Blocks */
pre, code {
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 9pt;
    background: #f7fafc;
    border-radius: 4px;
}

code {
    padding: 0.1cm 0.2cm;
    color: #d53f8c;
}

pre {
    padding: 0.5cm;
    overflow-x: auto;
    border: 1px solid #e2e8f0;
    margin: 0.5cm 0;
}

pre code {
    padding: 0;
    background: none;
    color: #1a202c;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.5cm 0;
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 10pt;
}

th {
    background: #2c5282;
    color: white;
    font-weight: 600;
    padding: 0.3cm 0.4cm;
    text-align: left;
    border: 1px solid #2c5282;
}

td {
    padding: 0.3cm 0.4cm;
    border: 1px solid #e2e8f0;
    vertical-align: top;
}

tr:nth-child(even) {
    background: #f7fafc;
}

/* Strong/Bold */
strong {
    font-weight: 700;
    color: #1a202c;
}

/* Emphasis */
em {
    font-style: italic;
}

/* Links */
a {
    color: #3182ce;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Reference Section */
h2#references + p {
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 12pt;
    font-weight: 600;
    color: #2d3748;
}

h3 + ol {
    font-size: 9.5pt;
    line-height: 1.5;
}

h3 + ol li {
    margin-bottom: 0.4cm;
    padding-left: 0.3cm;
}

/* Appendix Styling */
h2#appendix-research-statistics + table {
    width: auto;
    margin: 0 auto;
}

/* Page Breaks */
h2 {
    page-break-before: auto;
}

h2#introduction {
    page-break-before: always;
}

h2#references {
    page-break-before: always;
}

/* Print Optimizations */
@media print {
    body {
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
    }
    
    a[href]::after {
        content: none !important;
    }
}

/* Key Insights Numbered List */
h3#key-insights + p + ol li strong:first-child,
h3 + ol li strong:first-child {
    color: #2c5282;
}

/* Conclusion Key Takeaways */
h2#conclusions + p + p + ol li {
    margin-bottom: 0.5cm;
}

/* Cover Page Metadata */
.cover-meta {
    text-align: center;
    margin-top: 4cm;
    font-family: 'Source Sans Pro', sans-serif;
    color: #718096;
    font-size: 10pt;
}

.cover-meta p {
    margin: 0.2cm 0;
    text-indent: 0;
    text-align: center;
}
"""


def preprocess_markdown(md_content: str) -> str:
    """
    Preprocess markdown for better PDF rendering.
    - Fix citation formatting
    - Add IDs to headers for TOC
    - Clean up formatting issues
    """
    
    # Fix the title if it's a question (remove question marks from headers)
    lines = md_content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Add anchor IDs to headers for TOC linking
        if line.startswith('## '):
            header_text = line[3:].strip()
            anchor = header_text.lower()
            anchor = re.sub(r'[^a-z0-9\s-]', '', anchor)
            anchor = re.sub(r'\s+', '-', anchor)
            anchor = anchor.strip('-')
            processed_lines.append(f'## {header_text} {{#{anchor}}}')
        elif line.startswith('### '):
            header_text = line[4:].strip()
            anchor = header_text.lower()
            anchor = re.sub(r'[^a-z0-9\s-]', '', anchor)
            anchor = re.sub(r'\s+', '-', anchor)
            anchor = anchor.strip('-')
            processed_lines.append(f'### {header_text} {{#{anchor}}}')
        else:
            processed_lines.append(line)
    
    content = '\n'.join(processed_lines)
    
    # Convert [Source Title] citations to styled format
    content = re.sub(
        r'\[([^\]]+)\](?!\()',
        r'<em class="citation">[\1]</em>',
        content
    )
    
    return content


def markdown_to_html(md_content: str, title: str = "Research Report") -> str:
    """Convert markdown to styled HTML"""
    
    if not MARKDOWN_AVAILABLE:
        raise ImportError("markdown package not installed. Run: pip install markdown")
    
    # Preprocess
    md_content = preprocess_markdown(md_content)
    
    # Convert to HTML
    md = markdown.Markdown(
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'attr_list',
            TocExtension(permalink=False, toc_depth=3),
        ]
    )
    
    html_content = md.convert(md_content)
    
    # Build full HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{PDF_STYLES}
    </style>
</head>
<body>
    <article>
{html_content}
    </article>
    
    <footer class="cover-meta">
        <p>Generated by Deep Research Swarm</p>
        <p>{datetime.now().strftime('%B %d, %Y')}</p>
    </footer>
</body>
</html>
"""
    
    return html


def generate_pdf(markdown_path: str, output_path: str = None, title: str = None) -> str:
    """
    Generate PDF from markdown file.
    
    Args:
        markdown_path: Path to markdown file
        output_path: Output PDF path (default: same name with .pdf)
        title: Document title (extracted from markdown if not provided)
    
    Returns:
        str: Path to generated PDF
    """
    
    if not WEASYPRINT_AVAILABLE:
        raise ImportError(
            "weasyprint package not installed.\n"
            "Install with: pip install weasyprint\n"
            "Note: weasyprint requires additional system dependencies.\n"
            "See: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
        )
    
    # Read markdown
    print(f"Reading markdown from: {markdown_path}")
    with open(markdown_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Extract title from first H1 if not provided
    if not title:
        title_match = re.search(r'^#\s+(.+)$', md_content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = "Research Report"
    
    # Set output path
    if not output_path:
        output_path = os.path.splitext(markdown_path)[0] + '.pdf'
    
    # Convert to HTML
    print("Converting markdown to HTML...")
    html_content = markdown_to_html(md_content, title)
    
    # Save HTML for debugging (optional)
    html_debug_path = os.path.splitext(output_path)[0] + '_debug.html'
    with open(html_debug_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Debug HTML saved to: {html_debug_path}")
    
    # Generate PDF
    print("Generating PDF...")
    html = HTML(string=html_content, base_url=os.path.dirname(os.path.abspath(markdown_path)))
    html.write_pdf(output_path)
    
    print(f"✅ PDF generated: {output_path}")
    
    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   Size: {size_mb:.2f} MB")
    
    return output_path


def check_dependencies():
    """Check if required dependencies are installed"""
    issues = []
    
    if not MARKDOWN_AVAILABLE:
        issues.append("markdown: pip install markdown")
    
    if not WEASYPRINT_AVAILABLE:
        issues.append("weasyprint: pip install weasyprint")
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Generate beautiful PDF from markdown research report"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input markdown file path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output PDF path (default: same name with .pdf extension)"
    )
    parser.add_argument(
        "-t", "--title",
        help="Document title (default: extracted from markdown)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies and exit"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple PDF generator (fpdf2) instead of weasyprint"
    )
    
    args = parser.parse_args()
    
    # Check if should use simple version
    if args.simple or not WEASYPRINT_AVAILABLE:
        if args.input:
            try:
                from generate_pdf_simple import generate_pdf as generate_pdf_simple
                print("Using simple PDF generator (fpdf2)...")
                output_path = generate_pdf_simple(
                    markdown_path=args.input,
                    output_path=args.output
                )
                print(f"\n🎉 Success! Open your PDF:\n   {output_path}")
                return 0
            except ImportError:
                print("❌ Simple PDF generator requires fpdf2.")
                print("   Install with: pip install fpdf2")
                return 1
            except Exception as e:
                print(f"\n❌ Error generating PDF: {e}")
                import traceback
                traceback.print_exc()
                return 1
        elif not WEASYPRINT_AVAILABLE:
            print("=" * 50)
            print("PDF Generator - Dependency Check")
            print("=" * 50)
            print("\n⚠️  weasyprint not available (missing system dependencies)")
            print("\nOptions:")
            print("   1. Use simple generator: python generate_pdf_simple.py <file>")
            print("   2. Install weasyprint deps:")
            print("      macOS: brew install pango libffi glib")
            print("      Ubuntu: apt-get install libpango-1.0-0 libpangocairo-1.0-0")
            return 1
    
    # Check dependencies for weasyprint
    issues = check_dependencies()
    
    if args.check or issues:
        print("=" * 50)
        print("PDF Generator - Dependency Check")
        print("=" * 50)
        
        if issues:
            print("\n❌ Missing dependencies:")
            for issue in issues:
                print(f"   - {issue}")
            print("\nInstall all with:")
            print("   pip install markdown weasyprint pygments")
            print("\nNote: weasyprint requires system dependencies:")
            print("   macOS: brew install pango libffi glib")
            print("   Ubuntu: apt-get install libpango-1.0-0 libpangocairo-1.0-0")
            print("\n💡 Tip: Use --simple flag or generate_pdf_simple.py for no system deps")
            return 1
        else:
            print("\n✅ All dependencies installed!")
            if args.check:
                return 0
    
    # Generate PDF
    if not args.input:
        parser.print_help()
        return 1
        
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return 1
    
    try:
        output_path = generate_pdf(
            markdown_path=args.input,
            output_path=args.output,
            title=args.title
        )
        print(f"\n🎉 Success! Open your PDF:\n   {output_path}")
        return 0
    except Exception as e:
        print(f"\n❌ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

