#!/usr/bin/env python3
"""
The main training script for the ivory/bone image segmentation model. 

The script reproduces the key results presented in publication: "Automated identification of ivory carvings machine learning: A proof-of-concept approach to detecting the illegal wildlife trade".

This includes the trainig routines for the different image augmentation strategies discussed in the paper,
along with the calculation of the key statistics presented in Tables 1 and 2 in the paper. 

"""

import fastbook
fastbook.setup_book()


# Import all required libraries
from fastcore.xtras import Path

from fastai.callback.hook import summary
from fastai.callback.progress import ShowGraphCallback
from fastai.callback.schedule import lr_find, fit_flat_cos

from fastai.data.block import DataBlock, TransformBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files, FuncSplitter, Normalize, RandomSplitter, IntToFloatTensor

from fastai.layers import Mish
from fastai.losses import CrossEntropyLossFlat
from fastai.optimizer import ranger
from fastai.metrics import DiceMulti
from fastai.interpret import SegmentationInterpretation

from fastai.torch_core import cast, TensorBase

from fastai.vision.augment import *
from fastai.vision.core import PILImage, PILMask
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.all import unet_learner

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math

from torchvision.models.resnet import resnet34

from sklearn.metrics import (roc_curve, auc, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)


# Training data path: 
path = Path("/mnt/vstor/CAS_PHY_GDS6/axt807/data/Ivory_data/Ivory/")
path_lbl = path / "binary_masks"
path_im = path / "bones_and_ivory"


# Test data paths for inference:
test_path_ivory = Path("/mnt/vstor/CAS_PHY_GDS6/axt807/data/Ivory_data/Ivory/Public_domain_ivory_bone_images/ivory_clr")
test_path_bone = Path("/mnt/vstor/CAS_PHY_GDS6/axt807/data/Ivory_data/Ivory/Public_domain_ivory_bone_images/bone_clr")

# Output path for inference results:
outpath = Path("/mnt/vstor/CAS_PHY_GDS6/axt807/data/Ivory_data/Ivory/Public_domain_ivory_bone_images/inference")
outpath.mkdir(parents=True, exist_ok=True)

# Verify paths exist:
print(f"Data path exists: {path.exists()}")
print(f"Labels path exists: {path_lbl.exists()}")
print(f"Images path exists: {path_im.exists()}")


# Define class codes and mask values
codes = ['_background_', 'Ivory', 'no_ivory']
vals = dict()
for code, value in zip([0, 1, 2], [0, 38, 75]):
    vals[code] = value

def get_msk(fn, vals):
    """Grab a mask from a filename and adjust the pixels"""
    fn = path_lbl / f'{fn.stem}.png'
    msk = np.array(PILMask.create(fn))
    for i, val in enumerate(vals):
        msk[msk == vals[i]] = val
    return PILMask.create(msk)

get_y = lambda o: get_msk(o, vals)


# Define accuracy function for segmentation:
def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    """Computes non-background accuracy for multiclass segmentation"""
    targ = cast(targ.squeeze(1), TensorBase)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


def load_data(data_path, batch_size, item_tfms, batch_tfms):
    """Load and prepare data for training"""
    ivory = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                      get_items=get_image_files,
                      splitter=RandomSplitter(valid_pct=0.2, seed=42),
                      get_y=get_y,
                      item_tfms=item_tfms,
                      batch_tfms=batch_tfms)
    return ivory.dataloaders(path_im, bs=batch_size)

def create_model(dataloader):
    """Create UNet model with ResNet34 backbone"""
    return unet_learner(dataloader, resnet34, loss_func=CrossEntropyLossFlat(axis=1),
                       metrics=[foreground_acc, DiceMulti],
                       self_attention=True, act_cls=Mish, opt_func=ranger,
                       path=path)


from PIL import Image, ExifTags, ImageOps
from exif import Image as ExifImage
from glob import glob
import os.path


# Define transformation functions for different EXIF orientations:
transformation_funcs = {
    6: lambda img: img.rotate(-90, resample=Image.BICUBIC, expand=True),
    8: lambda img: img.rotate(90, resample=Image.BICUBIC, expand=True),
    3: lambda img: img.rotate(180, resample=Image.BICUBIC, expand=True),
    2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    5: lambda img: img.rotate(-90, resample=Image.BICUBIC, expand=True).transpose(Image.FLIP_LEFT_RIGHT),
    7: lambda img: img.rotate(90, resample=Image.BICUBIC, expand=True).transpose(Image.FLIP_LEFT_RIGHT),
    4: lambda img: img.rotate(180, resample=Image.BICUBIC, expand=True).transpose(Image.FLIP_LEFT_RIGHT),
}

# Process all JPG files in the image directory:
for img_path in glob(os.path.join(str(path_im), '*.JPG')):
    with open(img_path, 'rb') as image_file:
        my_image = ExifImage(image_file)
    try:
        orientation = my_image.orientation.value
    except:
        print('  N/A skipping')
        continue
    if orientation == 1:
        print('  1 skipping')
        continue
    else:
        print(img_path)
        print(' ', orientation, 'transforming')

    my_image.orientation = 1
    with open(img_path, 'wb') as image_file:
        image_file.write(my_image.get_file())

    img = Image.open(img_path)
    img = transformation_funcs[orientation](img)
    img.save(img_path)


item_tfms = Resize(400)
batch_tfms = [Normalize.from_stats(*imagenet_stats)]
dls = load_data(path, 8, item_tfms, batch_tfms)


learn = unet_learner(dls, resnet34, loss_func=CrossEntropyLossFlat(axis=1),
                     metrics=[foreground_acc, DiceMulti],
                     self_attention=True, act_cls=Mish, opt_func=ranger,
                     path=path)


learn.model
learn.summary()

####################################################################
## The training routines:                                         ##
####################################################################

# Training and saving the different models with various augmentation strategies as described in the paper.
learn.lr_find()
lr = 2.e-4
learn.fit_flat_cos(10, slice(lr), cbs=[ShowGraphCallback()])

# Save the model: 
learn.save("resize_noaug_bs8_stage1")

learn.unfreeze()
lrs = slice(lr/400, lr/4)
learn.fit_flat_cos(8, lrs, cbs=[ShowGraphCallback()])


learn.save("resize_noaug_bs8_stage2")
interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(6)


item_tfms = Resize(400)
batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
dls = load_data(path, 8, item_tfms, batch_tfms)

learn = create_model(dls)
learn.lr_find()


lr = 5.e-4
learn.fit_flat_cos(12, slice(lr), cbs=[ShowGraphCallback()])
learn.save("resize_aug_bs8_stage1")


interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(6)


learn.unfreeze()
lrs = slice(lr/400, lr/4)
learn.fit_flat_cos(10, lrs, cbs=[ShowGraphCallback()])
learn.save("resize_aug_bs8_stage2")


interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(6)


item_tfms = RandomResizedCrop(400, min_scale=0.3)
batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
dls = load_data(path, 8, item_tfms, batch_tfms)


learn = create_model(dls)


learn.lr_find()


lr = 5.e-4
learn.fit_flat_cos(12, slice(lr), cbs=[ShowGraphCallback()])
learn.save("rrc_aug_bs8_stage1")


interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(6)


learn.load("rrc_aug_bs8_stage1")
learn.unfreeze()
lr = 1.6e-4
lrs = slice(lr/400, lr/4)
learn.fit_flat_cos(10, lrs, cbs=[ShowGraphCallback()])
learn.save("rrc_aug_bs8_stage2")


interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(6)


item_tfms = RandomResizedCrop(400, min_scale=0.2)
xtra_tfms = [Hue(max_hue=0.25), Saturation(max_lighting=0.5)]
batch_tfms = [*aug_transforms(xtra_tfms=xtra_tfms), Normalize.from_stats(*imagenet_stats)]
dls = load_data(path, 8, item_tfms, batch_tfms)


dls.show_batch()


learn = create_model(dls)


lr = learn.lr_find()


slice(lr[0])


learn.fit_flat_cos(12, slice(lr[0]), cbs=[ShowGraphCallback()])
learn.save("rrc_aug_huesat_bs8_stage1")


interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(6)


learn.unfreeze()
lrs = slice(lr[0]/400, lr[0]/4)
learn.fit_flat_cos(15, lrs, cbs=[ShowGraphCallback()])
learn.save("rrc_aug_huesat_bs8_stage2")


interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(8)



################################################################
## Statistical analysis (post-training):                      ##
################################################################

def get_dl(model, direc):
    """Get dataloader for inference on a directory"""
    fnames = get_image_files(direc)
    return model.dls.test_dl(fnames)


def load_pretrained_model(model_index, train_path, return_model=False):
    """
    Load a pretrained model with specified augmentation configuration.
    
    Args:
        model_index: 0-3 selecting which model configuration to load
        train_path: Path to training data
        return_model: If True, return both model name and learner
    
    Returns:
        Learner with loaded weights, or (model_name, learner) tuple
    """
    xtra_tfms = [Hue(max_hue=0.25), Saturation(max_lighting=0.5)]
    item_tfms = [Resize(400),
                 Resize(400),
                 RandomResizedCrop(400, min_scale=0.3),
                 RandomResizedCrop(400, min_scale=0.3)]
    batch_tfms = [[Normalize.from_stats(*imagenet_stats)],
                  [*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
                  [*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
                  [*aug_transforms(xtra_tfms=xtra_tfms),
                   Normalize.from_stats(*imagenet_stats)]
                  ]

    # Models I, II, III, IV from the paper: 
    model_names = ["sel_resize_noaug_bs8_stage2",
                   "justbones_resize_aug_bs8_stage2",
                   "justbones_rrc_aug_bs8_stage2",
                   "justbones_rrc_aug_huesat_bs8_stage2"
                   ]
    
    dls = load_data(train_path, 8, item_tfms[model_index], batch_tfms[model_index])
    learn = create_model(dls)
    if return_model:
        return model_names[model_index], learn.load(model_names[model_index])
    else:
        return learn.load(model_names[model_index])


def new_show_predictions(ipath, preds, npics=None, thr=0.3, sz=8,
                         cmap='inferno', outname="preds.png", plot=True):
    if npics is None:
        npics = len(preds[0])

    classes = []
    ivory_frac = []
    avg_preds = []
    fnames = get_image_files(ipath)  # moved outside loop

    if plot:
        fig, axs = plt.subplots(nrows=npics, ncols=2, figsize=(2*sz, npics*sz))
    else:
        axs = [None] * npics  # dummy to keep loop structure

    for i, (pred, ax) in enumerate(zip(preds[0][:npics], axs)):
        pred_arx = pred.argmax(axis=0)
        ivory = np.where(pred_arx > 0, pred[1], np.nan)
        no_ivory = np.where(pred_arx > 0, pred[2], np.nan)
        excess = ivory - no_ivory / (ivory + no_ivory)
        #excess = (ivory - no_ivory) / (ivory + no_ivory)
        flat = excess[~np.isnan(excess)]
        ivory_frac.append(sum(flat > 0) / len(flat))
        avg_pred = sum(flat) / len(flat)

        if avg_pred > thr:
            classes.append("ivory")
        elif avg_pred < -thr:
            classes.append("bone")
        else:
            classes.append("uncertain")
        avg_preds.append(avg_pred)

        if plot:
            new_pred = np.where(pred_arx > 1, -1, pred_arx)
            img = PILImage.create(fnames[i])
            ax[0].imshow(img)
            ax[0].axis("off")
            ax[1].annotate(
                "Fraction of ivory pixels: {:.2f} \n Average prediction: {:.2f} \n Predicted class: {}".format(
                    ivory_frac[-1], avg_pred, classes[-1]),
                xy=(1, 1), xycoords='data', xytext=(0.98, 0.98), textcoords='axes fraction',
                bbox=dict(boxstyle="round4", fc="w"),
                horizontalalignment='right', verticalalignment='top', fontsize=14)
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax[1].imshow(excess, cmap='inferno', vmin=-1, vmax=1)
            fig.colorbar(im, cax=cax, orientation='vertical')

    if plot:
        plt.savefig(f"{outname}")

    return {"ivory_frac": ivory_frac, "avg_pred": avg_preds, "class": classes}

## For the Youden J calculations: 
def youden_j(fpr, tpr, thresholds):
    """Returns (J statistic, optimal threshold, sensitivity, specificity) at the best cut-point."""
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    return j_scores[idx], thresholds[idx], tpr[idx], 1 - fpr[idx]

# Load one of the saved models (0 -> Model I, 1 -> Model II, 2 -> Model III, 3 -> Model IV):
learn = load_pretrained_model(3, path)


# Ivory predictions: 
fnames = get_image_files(test_path_ivory)
dl = learn.dls.test_dl(fnames)
preds = learn.get_preds(dl=dl)
ivory_pred = new_show_predictions(test_path_ivory, preds, npics=None,
                                  outname=str(outpath / "Ivory"), plot=False)
ivory_pred

# Bone predictions:
fnames = get_image_files(test_path_bone)
dl = learn.dls.test_dl(fnames)
preds = learn.get_preds(dl=dl)
bone_pred = new_show_predictions(test_path_bone, preds, npics=None,
                                 outname=str(outpath / "Bone"), plot=False)
bone_pred


## Ivory prediction probabilities:
true_labels_ivory = np.ones(len(ivory_pred['class']), dtype=int)    # label 1 = ivory
predicted_probs_ivory = (np.array(ivory_pred['avg_pred']) + 1) / 2  # scale [-1,1] -> [0,1]
print(f"Ivory samples: {len(true_labels_ivory)}")
print(f"Predicted probs (ivory): {predicted_probs_ivory}")


## Bone prediction probabilities:
true_labels_bone = np.zeros(len(bone_pred['class']), dtype=int)      # label 0 = bone
predicted_probs_bones = (np.array(bone_pred['avg_pred']) + 1) / 2    # scale [-1,1] -> [0,1]
print(f"Bone samples: {len(true_labels_bone)}")
print(f"Predicted probs (bone): {predicted_probs_bones}")

## Combine and compute ROC:
true_labels = np.concatenate((true_labels_ivory, true_labels_bone))
predicted_probs = np.concatenate((predicted_probs_ivory, predicted_probs_bones))


fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)
j, opt_thr, sens, spec = youden_j(fpr, tpr, thresholds)
print(f"\nYouden's J : {j:.4f}  (opt threshold={opt_thr:.4f}, sensitivity={sens:.4f}, specificity={spec:.4f})")


## Plot:
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(str(outpath / "ROC_curve_model_3_clr.png"))
#plt.show()
print(f"AUC: {roc_auc:.4f}")


# ── Precision, Recall, F1-score ──────────────────────────────────────────────
# Convert predicted probabilities to binary class labels using threshold = 0.5
predicted_classes = (predicted_probs >= 0.5).astype(int)

precision = precision_score(true_labels, predicted_classes)
recall    = recall_score(true_labels, predicted_classes)
f1        = f1_score(true_labels, predicted_classes)

print("\n=== Classification Metrics ===")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {roc_auc:.4f}")

# ── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(true_labels, predicted_classes, normalize='true')
print("\nConfusion Matrix (rows=true, cols=predicted):")
print(f"  Labels: 0=bone, 1=ivory")
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bone", "ivory"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(str(outpath / "confusion_matrix.png"))
#plt.show()
print(f"Confusion matrix saved to {outpath / 'confusion_matrix_model_3_clr.png'}")


## Plot with AUC, precission, Recall etc printed: 
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')

textstr = '\n'.join([
    f'AUC: {roc_auc:.4f}',
    f'Precision: {precision:.4f}',
    f'Recall: {recall:.4f}',
    f'F1-score: {f1:.4f}',
    f'Youden\'s J: {j:.4f}'
])

ax.text(
    0.63, 0.15, textstr,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

ax.legend(loc='lower right')
plt.savefig(str(outpath / "ROC_curve_with_stats_model_3_clr.png"))
#plt.show()



############################################
### Discarding "uncertain" predictions:   ##
############################################

# Combine avg_preds and true labels
all_avg_preds = np.concatenate((ivory_pred['avg_pred'], bone_pred['avg_pred']))
thr = 0.3

# Mask out uncertain cases (where |avg_pred| < thr, i.e., prob between 0.35 and 0.65)
confident_mask = np.abs(all_avg_preds) >= thr

true_labels_confident = true_labels[confident_mask]
predicted_probs_confident = predicted_probs[confident_mask]
predicted_classes_confident = (predicted_probs_confident >= 0.5).astype(int)

print(f"Confident samples: {confident_mask.sum()} / {len(true_labels)} total")

#fpr_c, tpr_c, _ = roc_curve(true_labels_confident, predicted_probs_confident)
fpr_c, tpr_c, thr_c = roc_curve(true_labels_confident, predicted_probs_confident)
j_ens_c, opt_thr_ens_c, sens_ens_c, spec_ens_c = youden_j(fpr_c, tpr_c, thr_c)
print(f"Youden's J (confident): {j_ens_c:.4f}  (opt threshold={opt_thr_ens_c:.4f}, sensitivity={sens_ens_c:.4f}, specificity={spec_ens_c:.4f})")

print(f"AUC (confident): {auc(fpr_c, tpr_c):.4f}")
print(f"Precision: {precision_score(true_labels_confident, predicted_classes_confident):.4f}")
print(f"Recall:    {recall_score(true_labels_confident, predicted_classes_confident):.4f}")
print(f"F1:        {f1_score(true_labels_confident, predicted_classes_confident):.4f}")

cm_confident = confusion_matrix(true_labels_confident, predicted_classes_confident, normalize='true')
print("\nConfusion Matrix (confident only, rows=true, cols=predicted):")
print(f"  Labels: 0=bone, 1=ivory")
print(cm_confident)

## ROC plots: 
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_c, tpr_c, color='darkorange', lw=2,
        label=f'ROC curve (AUC = {auc(fpr_c, tpr_c):.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve (confident predictions only)')
#ax.legend(loc='lower right')

textstr = '\n'.join([
    f'AUC: {auc(fpr_c, tpr_c):.4f}',
    f'Precision: {precision_score(true_labels_confident, predicted_classes_confident):.4f}',
    f'Recall: {recall_score(true_labels_confident, predicted_classes_confident):.4f}',
    f'F1-score: {f1_score(true_labels_confident, predicted_classes_confident):.4f}',
    f'Youden\'s J: {j_ens_c:.4f}'
])
ax.text(
    0.63, 0.25, textstr,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

fig.tight_layout()
plt.savefig(str(outpath / "ROC_curve_confident_only_model_3_clr.png"))


## CM plot: 
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_confident, display_labels=["bone", "ivory"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix (confident predictions only)")
plt.tight_layout()
plt.savefig(str(outpath / "confusion_matrix_confident_only_model_3_clr.png"))

############################################
### Ensemble: average predictions over   ##
### selected models (e.g. models 2 & 3)  ##
############################################

ENSEMBLE_MODELS = [2, 3]  # indices into load_pretrained_model

ivory_avg_preds_list = []
bone_avg_preds_list  = []

for model_idx in ENSEMBLE_MODELS:
    learn_ens = load_pretrained_model(model_idx, path)

    fnames_iv = get_image_files(test_path_ivory)
    dl_iv = learn_ens.dls.test_dl(fnames_iv)
    preds_iv = learn_ens.get_preds(dl=dl_iv)
    res_iv = new_show_predictions(test_path_ivory, preds_iv, npics=None, plot=False)

    fnames_bn = get_image_files(test_path_bone)
    dl_bn = learn_ens.dls.test_dl(fnames_bn)
    preds_bn = learn_ens.get_preds(dl=dl_bn)
    res_bn = new_show_predictions(test_path_bone, preds_bn, npics=None, plot=False)

    ivory_avg_preds_list.append(np.array(res_iv['avg_pred']))
    bone_avg_preds_list.append(np.array(res_bn['avg_pred']))

# Average the per-image scores across models
ensemble_ivory_avg = np.mean(ivory_avg_preds_list, axis=0)
ensemble_bone_avg  = np.mean(bone_avg_preds_list,  axis=0)

true_labels_ens  = np.array([1] * len(ensemble_ivory_avg) + [0] * len(ensemble_bone_avg))
all_avg_ens      = np.concatenate([ensemble_ivory_avg, ensemble_bone_avg])
predicted_probs_ens  = (all_avg_ens + 1) / 2
predicted_classes_ens = (predicted_probs_ens >= 0.5).astype(int)

fpr_ens, tpr_ens, _ = roc_curve(true_labels_ens, predicted_probs_ens)
roc_auc_ens = auc(fpr_ens, tpr_ens)

print(f"\n=== Ensemble (models {ENSEMBLE_MODELS}) — All images ===")
print(f"Precision : {precision_score(true_labels_ens, predicted_classes_ens):.4f}")
print(f"Recall    : {recall_score(true_labels_ens, predicted_classes_ens):.4f}")
print(f"F1-score  : {f1_score(true_labels_ens, predicted_classes_ens):.4f}")
print(f"AUC       : {roc_auc_ens:.4f}")

cm_ens = confusion_matrix(true_labels_ens, predicted_classes_ens, normalize='true')
print("\nConfusion Matrix (rows=true, cols=predicted, 0=bone, 1=ivory):")
print(cm_ens.round(3))

# Confident-only subset
thr_ens = 0.3
conf_mask_ens = np.abs(all_avg_ens) >= thr_ens
yt_c = true_labels_ens[conf_mask_ens]
ypr_c = predicted_probs_ens[conf_mask_ens]
yp_c  = (ypr_c >= 0.5).astype(int)
fpr_c, tpr_c, _ = roc_curve(yt_c, ypr_c)
print(f"\n=== Ensemble (models {ENSEMBLE_MODELS}) — Confident only ({conf_mask_ens.sum()}/{len(true_labels_ens)}, thr={thr_ens}) ===")
print(f"Precision : {precision_score(yt_c, yp_c):.4f}")
print(f"Recall    : {recall_score(yt_c, yp_c):.4f}")
print(f"F1-score  : {f1_score(yt_c, yp_c):.4f}")
print(f"AUC       : {auc(fpr_c, tpr_c):.4f}")

cm_ens_conf = confusion_matrix(yt_c, yp_c, normalize='true')
print("\nConfusion Matrix (confident only, rows=true, cols=predicted, 0=bone, 1=ivory):")
print(cm_ens_conf.round(3))


fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_ens, tpr_ens, color='darkorange', lw=2,
        label=f'Ensemble models {ENSEMBLE_MODELS} (AUC = {roc_auc_ens:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC — Ensemble')
ax.legend(loc='lower right')
fig.tight_layout()
plt.savefig(str(outpath / f"ROC_ensemble_{'_'.join(map(str, ENSEMBLE_MODELS))}.png"))
