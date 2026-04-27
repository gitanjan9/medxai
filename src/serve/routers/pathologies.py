"""POST /v1/pathologies — 18-class multi-label pathology detection."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from src.serve.dependencies import get_request_id
from src.serve.schemas.response import PathologyResponse
from src.serve.services import pathology_detector as pd

router = APIRouter(prefix="/v1", tags=["pathologies"])


@router.post("/pathologies", response_model=PathologyResponse)
async def detect_pathologies(
    file: UploadFile,
    request_id: str = Depends(get_request_id),
) -> PathologyResponse:
    """Detect 18 chest pathologies from a CXR image.

    Uses a pretrained DenseNet-121 (torchxrayvision, trained on MIMIC-CXR +
    CheXpert + NIH ChestX-ray14 + PadChest) — no fine-tuning required.

    Returns sigmoid scores for each pathology class plus a list of findings
    above the detection threshold (0.30).
    """
    if not pd.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pathology detector not loaded. Set MEDXAI_PATHOLOGY_ENABLED=true.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    result = pd.get_pathology_detector().predict(image_bytes)

    return PathologyResponse(
        scores=result.scores,
        findings=result.findings,
        top_finding=result.top_finding,
        top_score=result.top_score,
        no_finding_score=result.no_finding_score,
        request_id=request_id,
    )
