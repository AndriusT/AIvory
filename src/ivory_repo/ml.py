from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# Heavy ML imports are deferred so this module can be loaded without fastai
# in environments where it is not installed. The functions that require
# fastai (load_model, load_data, create_model, new_show_predictions) will
# raise ImportError at call time if fastai is missing.

def _require_fastai():
    try:
        import fastai  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "fastai is required for ML utilities. "
            "Install it with: pip install fastai"
        ) from e

CODES = ["_background_", "Ivory", "no_ivory"]
MASK_VALS = {0: 0, 1: 38, 2: 75}
MODEL_NAMES = [
    "sel_resize_noaug_bs8_stage2_new",
    "justbones_resize_aug_bs8_stage2",
    "justbones_rrc_aug_bs8_stage2_new",
    "justbones_rrc_aug_huesat_bs8_stage2_new",
]


def get_msk(fn: Path, vals, path_lbl: Path):
    """Grab a mask from filename and remap pixels for fastai."""
    _require_fastai()
    from fastai.vision.core import PILMask
    fn = path_lbl / f"{fn.stem}.png"
    msk = np.array(PILMask.create(fn))
    for i, val in enumerate(vals):
        msk[msk == vals[i]] = val
    return PILMask.create(msk)


def foreground_acc(inp, targ, bkg_idx: int = 0, axis: int = 1):
    """Compute non-background accuracy for multiclass segmentation."""
    _require_fastai()
    from fastai.torch_core import TensorBase, cast
    targ = cast(targ.squeeze(1), TensorBase)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


def load_data(path: Path, path_lbl: Path, batch_size: int, item_tfms, batch_tfms):
    _require_fastai()
    from fastai.data.block import DataBlock
    from fastai.data.transforms import RandomSplitter, get_image_files
    from fastai.vision.data import ImageBlock, MaskBlock
    get_y = lambda o: get_msk(o, MASK_VALS, path_lbl)
    ivory = DataBlock(
        blocks=(ImageBlock, MaskBlock(CODES)),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=get_y,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    return ivory.dataloaders(path / "bones_and_ivory", bs=batch_size)


def create_model(dataloader, path: Path):
    _require_fastai()
    from fastai.layers import Mish
    from fastai.losses import CrossEntropyLossFlat
    from fastai.metrics import DiceMulti
    from fastai.optimizer import ranger
    from fastai.vision.all import unet_learner
    from torchvision.models.resnet import resnet34
    return unet_learner(
        dataloader,
        resnet34,
        loss_func=CrossEntropyLossFlat(axis=1),
        metrics=[foreground_acc, DiceMulti],
        self_attention=True,
        act_cls=Mish,
        opt_func=ranger,
        path=path,
    )


def load_model(model_index: int, train_path: Path, path_lbl: Path):
    _require_fastai()
    from fastai.vision.all import (
        Hue, Saturation, Resize, RandomResizedCrop,
        Normalize, aug_transforms, imagenet_stats,
    )
    xtra_tfms = [Hue(max_hue=0.25), Saturation(max_lighting=0.5)]
    item_tfms = [
        Resize(400),
        Resize(400),
        RandomResizedCrop(400, min_scale=0.3),
        RandomResizedCrop(400, min_scale=0.3),
    ]
    batch_tfms = [
        [Normalize.from_stats(*imagenet_stats)],
        [*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
        [*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
        [*aug_transforms(xtra_tfms=xtra_tfms), Normalize.from_stats(*imagenet_stats)],
    ]
    dls = load_data(train_path, path_lbl, 8, item_tfms[model_index], batch_tfms[model_index])
    learn = create_model(dls, train_path)
    learn.load(MODEL_NAMES[model_index], weights_only=False)
    return learn


def new_show_predictions(
    ipath: Path,
    preds,
    npics: Optional[int] = None,
    thr: float = 0.3,
    sz: int = 8,
    outname: str = "preds.png",
):
    _require_fastai()
    from fastai.data.transforms import get_image_files
    from fastai.vision.core import PILImage

    if npics is None:
        npics = len(preds[0])

    fig, axs = plt.subplots(nrows=npics, ncols=2, figsize=(2 * sz, npics * sz))

    classes = []
    ivory_frac = []
    avg_preds = []

    for i, (pred, ax) in enumerate(zip(preds[0][:npics], axs)):
        pred_arx = pred.argmax(axis=0)
        ivory = np.where(pred_arx > 0, pred[1], np.nan)
        no_ivory = np.where(pred_arx > 0, pred[2], np.nan)
        excess = ivory - no_ivory / (ivory + no_ivory)
        flat = excess[~np.isnan(excess)]

        ivory_frac.append(sum(flat > 0) / len(flat))
        avg_pred = sum(flat) / len(flat)
        avg_preds.append(avg_pred)

        if avg_pred > thr:
            classes.append("ivory")
        elif avg_pred < -thr:
            classes.append("bone")
        else:
            classes.append("uncertain")

        fnames = get_image_files(ipath)
        img_fn = fnames[i]
        img = PILImage.create(img_fn)
        ax[0].imshow(img)
        ax[0].axis("off")
        ax[1].annotate(
            "Fraction of ivory pixels: {:.2f}\nAverage prediction: {:.2f}\nPredicted class: {}".format(
                ivory_frac[-1], avg_pred, classes[-1]
            ),
            xy=(1, 1),
            xycoords="data",
            xytext=(0.98, 0.98),
            textcoords="axes fraction",
            bbox=dict(boxstyle="round4", fc="w"),
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=14,
        )
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax[1].imshow(excess, cmap="inferno", vmin=-1, vmax=1)
        fig.colorbar(im, cax=cax, orientation="vertical")

    plt.savefig(f"{outname}")
    return {"ivory_frac": ivory_frac, "avg_pred": avg_preds, "class": classes}


def classic_predict_from_pred(pred, thr: float = 0.3):
    """Convert segmentation logits into ivory/bone/uncertain class decision."""
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()

    pred_arx = pred.argmax(axis=0)
    ivory = np.where(pred_arx > 0, pred[1], np.nan)
    bone = np.where(pred_arx > 0, pred[2], np.nan)
    denom = ivory + bone

    excess = np.where(np.isfinite(denom) & (denom != 0), (ivory - bone) / denom, np.nan)
    flat = excess[~np.isnan(excess)]

    if flat.size == 0:
        return {
            "ivory_frac": np.nan,
            "avg_pred": np.nan,
            "class": "uncertain",
            "excess": excess,
        }

    ivory_frac = float(np.mean(flat > 0))
    avg_pred = float(np.mean(flat))

    if avg_pred > thr:
        klass = "ivory"
    elif avg_pred < -thr:
        klass = "bone"
    else:
        klass = "uncertain"

    return {
        "ivory_frac": ivory_frac,
        "avg_pred": avg_pred,
        "class": klass,
        "excess": excess,
    }


def compare_classic_vs_gemini(
    learn,
    client,
    model_id: str,
    image_paths,
    gemini_classify_image_fn,
    thr: float = 0.3,
    show_examples: bool = False,
):
    """Run side-by-side classic segmentation classification and Gemini classification."""
    results = []

    for i, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        print(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")

        try:
            dl = learn.dls.test_dl([image_path])
            preds = learn.get_preds(dl=dl)
            classic_pred = classic_predict_from_pred(preds[0][0], thr=thr)
        except Exception as e:
            print(f"Classic model failed: {e}")
            classic_pred = {
                "class": "error",
                "ivory_frac": np.nan,
                "avg_pred": np.nan,
                "excess": None,
            }

        try:
            gemini_pred = gemini_classify_image_fn(client, model_id, image_path)
        except Exception as e:
            print(f"Gemini failed: {e}")
            gemini_pred = {
                "class": "error",
                "confidence": None,
                "reasoning": str(e),
                "raw_text": "",
            }

        results.append(
            {
                "filename": image_path.name,
                "path": str(image_path),
                "classic_class": classic_pred["class"],
                "classic_ivory_frac": classic_pred["ivory_frac"],
                "classic_avg_pred": classic_pred["avg_pred"],
                "gemini_class": gemini_pred["class"],
                "gemini_confidence": gemini_pred.get("confidence", None),
                "gemini_reasoning": gemini_pred.get("reasoning", ""),
                "gemini_raw_text": gemini_pred.get("raw_text", ""),
            }
        )

        if show_examples:
            img = Image.open(image_path).convert("RGB")
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(
                f"{image_path.name}\n"
                f"Classic: {classic_pred['class']} | Gemini: {gemini_pred['class']}"
            )
            plt.axis("off")
            plt.show()

    return pd.DataFrame(results)
