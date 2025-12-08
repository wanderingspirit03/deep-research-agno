from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add root directory to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import DeepResearchSwarm

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    query: str

@app.post("/api/research")
def research(request: ResearchRequest):
    """
    Execute deep research.
    Note: defined as sync function so FastAPI runs it in a threadpool,
    preventing the long-running research from blocking the event loop.
    """
    try:
        print(f"Starting research for: {request.query}")
        # Initialize swarm with reasonable defaults for web
        swarm = DeepResearchSwarm(
            max_workers=5,
            max_subtasks=5,
            max_iterations=1,  # Single iteration for faster response
            quality_threshold=80
        )
        
        result = swarm.deep_research(request.query)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or "Research failed")
            
        return {
            "report": result.report,
            "success": result.success,
            "summary": result.summary()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
