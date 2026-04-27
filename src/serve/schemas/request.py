"""Request schemas – thin; actual image data arrives as multipart UploadFile."""
from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Placeholder for future JSON body options (e.g. return_all_scores flag)."""
    return_all_scores: bool = True


class ExplainRequest(BaseModel):
    """Placeholder for future explainability options."""
    target_class: int | None = None   # None → use predicted class
    output_size: int = 320            # heatmap resize target
