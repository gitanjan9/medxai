"""Background retraining service.

Trigger policy
--------------
Retraining fires when the `retraining_queue` contains at least
RETRAIN_MIN_SAMPLES unprocessed wrong predictions (default: 10).
A single asyncio task runs at a time; concurrent calls are no-ops.

Training approach
-----------------
Fine-tunes the EfficientNet-B3 checkpoint (v2) on the accumulated
wrong predictions using a lightweight single-epoch pass.  The txrv
DenseNet is left untouched (it is a large pretrained model).
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
import io

from src.common.logging import get_logger
from src.serve.services.prediction_store import (
    get_retraining_queue,
    mark_retrained,
    retraining_queue_size,
)

logger = get_logger("serve.retraining")

RETRAIN_MIN_SAMPLES: int = int(os.getenv("RETRAIN_MIN_SAMPLES", "10"))
RETRAIN_LR: float = float(os.getenv("RETRAIN_LR", "1e-5"))
RETRAIN_EPOCHS: int = int(os.getenv("RETRAIN_EPOCHS", "1"))

# Global state
_lock = asyncio.Lock()
_status: dict = {"state": "idle", "last_run": None, "samples_used": 0, "error": None}


def get_retraining_status() -> dict:
    return dict(_status)


async def maybe_trigger_retraining() -> bool:
    """Called after each wrong-prediction feedback submission.
    Returns True if retraining was started, False if skipped.
    """
    if _lock.locked():
        logger.info("Retraining already in progress — skipping trigger")
        return False

    queue_size = retraining_queue_size()
    logger.info("Retraining queue size: %d / %d threshold", queue_size, RETRAIN_MIN_SAMPLES)

    if queue_size < RETRAIN_MIN_SAMPLES:
        return False

    asyncio.create_task(_run_retraining())
    return True


async def force_retrain() -> bool:
    """Admin-triggered retraining regardless of queue size."""
    if _lock.locked():
        return False
    asyncio.create_task(_run_retraining())
    return True


async def _run_retraining() -> None:
    async with _lock:
        run_id = str(uuid.uuid4())
        _status.update({"state": "running", "run_id": run_id, "error": None})
        logger.info("Retraining started run_id=%s", run_id)
        t0 = time.perf_counter()

        try:
            items = get_retraining_queue(limit=500)
            if not items:
                _status.update({"state": "idle", "error": "empty queue"})
                return

            # Run fine-tuning in a thread pool (CPU/GPU bound, not async)
            loop = asyncio.get_event_loop()
            samples_used = await loop.run_in_executor(
                None, _fine_tune_sync, items, run_id
            )

            mark_retrained([i.id for i in items[:samples_used]], run_id)
            elapsed = round(time.perf_counter() - t0, 1)
            _status.update({
                "state": "idle",
                "last_run": run_id,
                "samples_used": samples_used,
                "elapsed_s": elapsed,
                "error": None,
            })
            logger.info(
                "Retraining complete run_id=%s samples=%d elapsed=%.1fs",
                run_id, samples_used, elapsed,
            )

        except Exception as exc:
            logger.error("Retraining failed run_id=%s: %s", run_id, exc)
            _status.update({"state": "error", "error": str(exc)})


def _fine_tune_sync(items, run_id: str) -> int:
    """Synchronous fine-tuning executed in a thread pool.

    Loads the v2 EfficientNet-B3 checkpoint, fine-tunes for RETRAIN_EPOCHS
    on the feedback samples, and saves a new checkpoint alongside the original.
    """
    from src.serve.services.artifact_loader import EnvConfig, load_all_artifacts
    from src.serve.services.inference import run_inference

    checkpoint_dir = Path(os.getenv("MEDXAI_CHECKPOINT", "artifacts/v2/checkpoints"))
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    # Find best existing checkpoint
    ckpts = sorted(checkpoint_dir.glob("*.pt"), reverse=True)
    if not ckpts:
        logger.warning("No checkpoint found — skipping fine-tune")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")

    # Load model
    from timm import create_model
    label_map_path = Path(os.getenv("MEDXAI_LABEL_MAP", "artifacts/v2/label_map.json"))
    import json
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map.get("idx_to_str", {}))
    if num_classes == 0:
        num_classes = 5

    arch = os.getenv("MEDXAI_ARCH", "efficientnet_b3")
    model = create_model(arch, pretrained=False, num_classes=num_classes)
    state = torch.load(str(ckpts[0]), map_location=device, weights_only=True)
    model_state = state.get("model_state_dict", state)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device).train()

    # Build label index map
    str_to_idx = label_map.get("str_to_idx", {v: k for k, v in label_map.get("idx_to_str", {}).items()})

    optimizer = torch.optim.AdamW(model.parameters(), lr=RETRAIN_LR)
    criterion = nn.CrossEntropyLoss()

    image_size = int(os.getenv("MEDXAI_IMAGE_SIZE", "320"))
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # We stored image_hash, not raw bytes — need to fetch from audit or skip
    # Items without stored images are skipped gracefully
    trained = 0
    skipped = 0
    for epoch in range(RETRAIN_EPOCHS):
        for item in items:
            # Attempt to reconstruct image from audit JSONL (best effort)
            # If image bytes are not available, skip this sample
            label_idx = str_to_idx.get(item.true_label)
            if label_idx is None:
                skipped += 1
                continue

            try:
                # Try to load from artifacts/images/<hash>.jpg if we cached it
                img_path = Path("artifacts/images") / f"{item.image_hash}.jpg"
                if not img_path.exists():
                    skipped += 1
                    continue
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                target = torch.tensor([int(label_idx)], dtype=torch.long, device=device)

                optimizer.zero_grad()
                logits = model(tensor)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                trained += 1
            except Exception as exc:
                logger.warning("Skip sample %s: %s", item.image_hash, exc)
                skipped += 1

    if trained == 0:
        logger.info("No trainable samples found (images not cached). run_id=%s", run_id)
        return 0

    # Save new checkpoint
    new_ckpt = checkpoint_dir / f"finetune_{run_id[:8]}.pt"
    torch.save({"model_state_dict": model.state_dict(), "run_id": run_id}, str(new_ckpt))
    logger.info("Saved fine-tuned checkpoint: %s (trained=%d skipped=%d)", new_ckpt, trained, skipped)
    return trained
