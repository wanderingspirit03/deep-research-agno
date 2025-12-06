"""
Daytona Sandbox Tools - Custom Agno Toolkit for secure code execution

Uses Daytona SDK for:
- Secure sandboxed code execution (Python)
- URL verification via HTTP requests
- Web content scraping
"""
import os
from typing import Optional, Dict, Any

from agno.tools import Toolkit
from agno.utils.log import logger

try:
    from daytona import Daytona
    DAYTONA_AVAILABLE = True
except ImportError:
    DAYTONA_AVAILABLE = False
    logger.warning("daytona package not installed. Run: pip install daytona")


class DaytonaSandboxTools(Toolkit):
    """
    Custom Agno Toolkit for Daytona sandbox operations.
    
    Features:
    - Secure Python code execution
    - URL verification and scraping
    - Automatic sandbox cleanup
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        target: str = "us",
        auto_cleanup: bool = True,
        timeout: int = 120,
    ):
        """
        Initialize Daytona Sandbox Tools.
        
        Args:
            api_key: Daytona API key (defaults to DAYTONA_API_KEY env var)
            target: Daytona target region (e.g., "us", "eu")
            auto_cleanup: Whether to automatically delete sandbox after use
            timeout: Default timeout for operations in seconds
        """
        self.api_key = api_key or os.getenv("DAYTONA_API_KEY")
        self.target = target or os.getenv("DAYTONA_TARGET", "us")
        self.auto_cleanup = auto_cleanup
        self.timeout = timeout
        
        # Lazy-initialized clients
        self._daytona: Optional[Daytona] = None
        self._sandbox = None
        
        # Register tools with Toolkit
        tools = [
            self.run_code,
            self.verify_url,
            self.scrape_content,
            self.cleanup,
        ]
        
        super().__init__(name="daytona_sandbox", tools=tools)
    
    @property
    def daytona(self) -> "Daytona":
        """Lazy initialization of Daytona client"""
        if not DAYTONA_AVAILABLE:
            raise ImportError("daytona package not installed. Run: pip install daytona")
        
        if self._daytona is None:
            if not self.api_key:
                raise ValueError("DAYTONA_API_KEY not set")
            
            # Set env var for Daytona SDK (it reads from environment)
            os.environ["DAYTONA_API_KEY"] = self.api_key
            if self.target:
                os.environ["DAYTONA_TARGET"] = self.target
            
            self._daytona = Daytona()
        
        return self._daytona
    
    def _get_sandbox(self):
        """Get or create a sandbox instance"""
        if self._sandbox is None:
            logger.info("Creating Daytona sandbox...")
            self._sandbox = self.daytona.create()
            sandbox_id = getattr(self._sandbox, 'id', 'unknown')
            logger.info(f"Sandbox created: {sandbox_id}")
        return self._sandbox
    
    def run_code(self, code: str, timeout: Optional[int] = None) -> str:
        """
        Execute Python code in a secure Daytona sandbox.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (default: 120)
        
        Returns:
            str: JSON-formatted result with stdout, stderr, and exit_code
        
        Example:
            >>> result = tools.run_code('print("Hello, World!")')
            >>> # Returns: {"stdout": "Hello, World!\\n", "stderr": "", "exit_code": 0}
        """
        timeout = timeout or self.timeout
        logger.info(f"Executing code in sandbox (timeout={timeout}s)")
        
        try:
            sandbox = self._get_sandbox()
            response = sandbox.process.code_run(code, timeout=timeout)
            
            result = {
                "stdout": getattr(response, "result", ""),
                "stderr": getattr(response, "stderr", ""),
                "exit_code": getattr(response, "exit_code", 0),
            }
            
            # Format as readable output
            output = []
            output.append("## Code Execution Result\n")
            
            if result["exit_code"] == 0:
                output.append("**Status:** ✅ Success")
            else:
                output.append(f"**Status:** ❌ Failed (exit code: {result['exit_code']})")
            
            if result["stdout"]:
                output.append(f"\n**Output:**\n```\n{result['stdout']}\n```")
            
            if result["stderr"]:
                output.append(f"\n**Errors:**\n```\n{result['stderr']}\n```")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"## Code Execution Error\n\n**Error:** {str(e)}"
    
    def verify_url(self, url: str, timeout: Optional[int] = None) -> str:
        """
        Verify that a URL is accessible by making an HTTP request from the sandbox.
        
        Args:
            url: The URL to verify
            timeout: Request timeout in seconds (default: 30)
        
        Returns:
            str: Verification result with URL, status, and page title
        
        Example:
            >>> result = tools.verify_url("https://arxiv.org/abs/2301.00001")
            >>> # Returns verification status, HTTP code, and page title
        """
        timeout = timeout or 30
        logger.info(f"Verifying URL: {url}")
        
        # Code to run in sandbox for URL verification
        verification_code = f'''
import urllib.request
import urllib.error
import re
import json

url = "{url}"
result = {{"url": url, "exists": False, "status_code": None, "title": None, "error": None}}

try:
    req = urllib.request.Request(url, headers={{"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}})
    with urllib.request.urlopen(req, timeout={timeout}) as response:
        result["exists"] = True
        result["status_code"] = response.status
        
        # Try to extract title
        content = response.read(10000).decode("utf-8", errors="ignore")
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1).strip()

except urllib.error.HTTPError as e:
    result["status_code"] = e.code
    result["error"] = f"HTTP {{e.code}}: {{e.reason}}"
except urllib.error.URLError as e:
    result["error"] = f"URL Error: {{e.reason}}"
except Exception as e:
    result["error"] = str(e)

print(json.dumps(result))
'''
        
        try:
            sandbox = self._get_sandbox()
            response = sandbox.process.code_run(verification_code, timeout=timeout + 10)
            
            # Parse the JSON result
            import json
            stdout = getattr(response, "result", "")
            
            try:
                result = json.loads(stdout.strip())
            except json.JSONDecodeError:
                return f"## URL Verification Error\n\nCould not parse result: {stdout}"
            
            # Format output
            output = [f"## URL Verification: {url}\n"]
            
            if result.get("exists"):
                output.append(f"**Status:** ✅ Accessible (HTTP {result.get('status_code')})")
                if result.get("title"):
                    output.append(f"**Title:** {result['title']}")
            else:
                output.append(f"**Status:** ❌ Not accessible")
                if result.get("error"):
                    output.append(f"**Error:** {result['error']}")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"URL verification failed: {e}")
            return f"## URL Verification Error\n\n**URL:** {url}\n**Error:** {str(e)}"
    
    def scrape_content(self, url: str, max_chars: int = 5000) -> str:
        """
        Scrape and extract text content from a URL.
        
        Args:
            url: The URL to scrape
            max_chars: Maximum characters to return (default: 5000)
        
        Returns:
            str: Extracted text content from the page
        
        Example:
            >>> content = tools.scrape_content("https://arxiv.org/abs/2301.00001")
            >>> # Returns cleaned text content from the page
        """
        logger.info(f"Scraping content from: {url}")
        
        # Code to run in sandbox for content scraping
        scrape_code = f'''
import urllib.request
import urllib.error
import re
import json

url = "{url}"
max_chars = {max_chars}

result = {{"url": url, "content": None, "error": None, "metadata": {{}}}}

try:
    req = urllib.request.Request(url, headers={{"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}})
    with urllib.request.urlopen(req, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")
        
        # Extract title
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if title_match:
            result["metadata"]["title"] = title_match.group(1).strip()
        
        # Remove script and style elements
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        
        # Clean up whitespace
        text = re.sub(r"\\s+", " ", text).strip()
        
        # Truncate to max_chars
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        result["content"] = text

except urllib.error.HTTPError as e:
    result["error"] = f"HTTP {{e.code}}: {{e.reason}}"
except urllib.error.URLError as e:
    result["error"] = f"URL Error: {{e.reason}}"
except Exception as e:
    result["error"] = str(e)

print(json.dumps(result))
'''
        
        try:
            sandbox = self._get_sandbox()
            response = sandbox.process.code_run(scrape_code, timeout=45)
            
            # Parse the JSON result
            import json
            stdout = getattr(response, "result", "")
            
            try:
                result = json.loads(stdout.strip())
            except json.JSONDecodeError:
                return f"## Scraping Error\n\nCould not parse result: {stdout}"
            
            # Format output
            output = [f"## Scraped Content from: {url}\n"]
            
            if result.get("error"):
                output.append(f"**Error:** {result['error']}")
                return "\n".join(output)
            
            if result.get("metadata", {}).get("title"):
                output.append(f"**Title:** {result['metadata']['title']}\n")
            
            if result.get("content"):
                output.append(f"**Content:**\n{result['content']}")
            else:
                output.append("**Content:** (empty)")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Content scraping failed: {e}")
            return f"## Scraping Error\n\n**URL:** {url}\n**Error:** {str(e)}"
    
    def cleanup(self) -> str:
        """
        Clean up and delete the Daytona sandbox.
        
        Should be called when research is complete to free resources.
        
        Returns:
            str: Cleanup status message
        """
        if self._sandbox is None:
            return "## Cleanup\n\nNo sandbox to clean up."
        
        try:
            sandbox_id = getattr(self._sandbox, 'id', 'unknown')
            self.daytona.delete(self._sandbox)
            self._sandbox = None
            logger.info(f"Sandbox {sandbox_id} deleted")
            return f"## Cleanup\n\n✅ Sandbox `{sandbox_id}` deleted successfully."
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return f"## Cleanup Error\n\n**Error:** {str(e)}"
    
    def __del__(self):
        """Cleanup sandbox on deletion if auto_cleanup is enabled"""
        if self.auto_cleanup and self._sandbox is not None:
            try:
                self.daytona.delete(self._sandbox)
            except Exception:
                pass


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Daytona Sandbox Tools Test ===\n")
    
    # Check API key
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        print("❌ DAYTONA_API_KEY not set")
        print("   Set it in your .env file or environment")
        exit(1)
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    # Initialize tools
    tools = DaytonaSandboxTools()
    
    # Test code execution
    print("\n--- Code Execution Test ---")
    result = tools.run_code('print("Hello from Daytona sandbox!")')
    print(result)
    
    # Cleanup
    print("\n--- Cleanup ---")
    result = tools.cleanup()
    print(result)

