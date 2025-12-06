"""
Docker Sandbox Tools - Local code execution using Docker containers

Alternative to Daytona for local development/testing:
- Secure containerized code execution (Python)
- URL verification via HTTP requests
- Web content scraping
"""
import os
import json
import tempfile
from typing import Optional

from agno.tools import Toolkit
from agno.utils.log import logger

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    logger.warning("docker package not installed. Run: pip install docker")


class DockerSandboxTools(Toolkit):
    """
    Custom Agno Toolkit for Docker-based sandbox operations.
    
    Features:
    - Secure Python code execution in containers
    - URL verification and scraping
    - No external API dependencies (runs locally)
    """
    
    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 120,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        network_enabled: bool = True,
    ):
        """
        Initialize Docker Sandbox Tools.
        
        Args:
            image: Docker image to use for execution
            timeout: Default timeout for operations in seconds
            memory_limit: Container memory limit (e.g., "512m", "1g")
            cpu_limit: CPU limit (1.0 = 1 CPU core)
            network_enabled: Whether to enable network access in container
        """
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_enabled = network_enabled
        
        # Lazy-initialized Docker client
        self._client: Optional[docker.DockerClient] = None
        
        # Register tools with Toolkit
        tools = [
            self.run_code,
            self.verify_url,
            self.scrape_content,
        ]
        
        super().__init__(name="docker_sandbox", tools=tools)
    
    @property
    def client(self) -> "docker.DockerClient":
        """Lazy initialization of Docker client"""
        if not DOCKER_AVAILABLE:
            raise ImportError("docker package not installed. Run: pip install docker")
        
        if self._client is None:
            try:
                self._client = docker.from_env()
                # Test connection
                self._client.ping()
                logger.info("Docker client connected")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to Docker: {e}. Is Docker running?")
        
        return self._client
    
    def _ensure_image(self):
        """Pull the Docker image if not present"""
        try:
            self.client.images.get(self.image)
            logger.debug(f"Image {self.image} found locally")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image: {self.image}")
            self.client.images.pull(self.image)
    
    def _run_in_container(self, code: str, timeout: Optional[int] = None) -> dict:
        """Execute code in a Docker container"""
        timeout = timeout or self.timeout
        
        self._ensure_image()
        
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name
        
        try:
            # Run container
            container = self.client.containers.run(
                self.image,
                command=["python", "/code/script.py"],
                volumes={code_file: {"bind": "/code/script.py", "mode": "ro"}},
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=int(100000 * self.cpu_limit),
                network_mode="bridge" if self.network_enabled else "none",
                detach=True,
                remove=False,
            )
            
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            
            # Get output
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
            exit_code = result.get('StatusCode', -1)
            
            # Cleanup container
            container.remove(force=True)
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }
            
        except docker.errors.ContainerError as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": e.exit_status,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
            }
        finally:
            # Cleanup temp file
            try:
                os.unlink(code_file)
            except Exception:
                pass
    
    def run_code(self, code: str, timeout: Optional[int] = None) -> str:
        """
        Execute Python code in a secure Docker container.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (default: 120)
        
        Returns:
            str: Formatted result with stdout, stderr, and exit_code
        
        Example:
            >>> result = tools.run_code('print("Hello, World!")')
        """
        timeout = timeout or self.timeout
        logger.info(f"Executing code in Docker container (timeout={timeout}s)")
        
        try:
            result = self._run_in_container(code, timeout)
            
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
        Verify that a URL is accessible by making an HTTP request from the container.
        
        Args:
            url: The URL to verify
            timeout: Request timeout in seconds (default: 30)
        
        Returns:
            str: Verification result with URL, status, and page title
        """
        timeout = timeout or 30
        logger.info(f"Verifying URL: {url}")
        
        # Code to run in container for URL verification
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
            result = self._run_in_container(verification_code, timeout + 10)
            
            # Parse the JSON result
            stdout = result.get("stdout", "").strip()
            
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                return f"## URL Verification Error\n\nCould not parse result: {stdout}"
            
            # Format output
            output = [f"## URL Verification: {url}\n"]
            
            if parsed.get("exists"):
                output.append(f"**Status:** ✅ Accessible (HTTP {parsed.get('status_code')})")
                if parsed.get("title"):
                    output.append(f"**Title:** {parsed['title']}")
            else:
                output.append(f"**Status:** ❌ Not accessible")
                if parsed.get("error"):
                    output.append(f"**Error:** {parsed['error']}")
            
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
        """
        logger.info(f"Scraping content from: {url}")
        
        # Code to run in container for content scraping
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
            result = self._run_in_container(scrape_code, 45)
            
            # Parse the JSON result
            stdout = result.get("stdout", "").strip()
            
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                return f"## Scraping Error\n\nCould not parse result: {stdout}"
            
            # Format output
            output = [f"## Scraped Content from: {url}\n"]
            
            if parsed.get("error"):
                output.append(f"**Error:** {parsed['error']}")
                return "\n".join(output)
            
            if parsed.get("metadata", {}).get("title"):
                output.append(f"**Title:** {parsed['metadata']['title']}\n")
            
            if parsed.get("content"):
                output.append(f"**Content:**\n{parsed['content']}")
            else:
                output.append("**Content:** (empty)")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Content scraping failed: {e}")
            return f"## Scraping Error\n\n**URL:** {url}\n**Error:** {str(e)}"


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=== Docker Sandbox Tools Test ===\n")
    
    # Check Docker availability
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("✅ Docker is running")
    except Exception as e:
        print(f"❌ Docker not available: {e}")
        print("   Make sure Docker Desktop is running")
        exit(1)
    
    # Initialize tools
    tools = DockerSandboxTools()
    
    # Test code execution
    print("\n--- Code Execution Test ---")
    result = tools.run_code('print("Hello from Docker sandbox!")')
    print(result)
    
    # Test URL verification
    print("\n--- URL Verification Test ---")
    result = tools.verify_url("https://example.com")
    print(result)
    
    # Test content scraping
    print("\n--- Content Scraping Test ---")
    result = tools.scrape_content("https://example.com", max_chars=500)
    print(result)
    
    print("\n✅ All tests completed!")


