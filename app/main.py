from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from workflow_graph import run_workflow
from workflow_graph_phi3_model import run_workflow_phi3_mini

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search/v1.1")
async def search_products(request: QueryRequest):
    try:
        output = run_workflow_phi3_mini(request.query)
        return {"status": "success", "products": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

