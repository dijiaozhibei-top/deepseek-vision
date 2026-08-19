"""GET /v1/models."""
from fastapi import APIRouter, Depends
from app.auth import require_auth
from app.router import MODEL_REGISTRY

router = APIRouter()


@router.get("/models")
async def list_models(api_key: str = Depends(require_auth)):
    models = [
        {"id": model_id, "type": "model", "display_name": model_id}
        for model_id in MODEL_REGISTRY
    ]
    return {
        "data": models,
        "has_more": False,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
    }
