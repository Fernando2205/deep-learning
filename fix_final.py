# -*- coding: utf-8 -*-
"""
DEFINITIVE fix. Strategy:
1. Remove --save_segmentation_images from both PatchCore commands
2. Rewrite heatmap cell: loads saved PatchCore model + generates real heatmaps
3. Rewrite ROC cell: loads saved model + generates real ROC from scores
No fragile string replacements in run_patchcore.py needed.
"""
import json, glob
from pathlib import Path

nb_path = Path(glob.glob("sam_patchcore_final_corregido*.ipynb")[0])
print(f"Processing: {nb_path}")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ============================================================================
# 1. REMOVE --save_segmentation_images from baseline and SAM+ROI commands
# ============================================================================
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "BASELINE_OUT = DATA_ROOT" in src or "ROI_OUT = DATA_ROOT" in src:
        new_source = []
        for line in cell["source"]:
            if "--save_segmentation_images" in line:
                # Remove that flag
                line = line.replace("'--save_segmentation_images', ", "")
                line = line.replace(", '--save_segmentation_images'", "")
                line = line.replace("'--save_segmentation_images'", "")
            new_source.append(line)
        cell["source"] = new_source
        print(f"  Removed --save_segmentation_images from cell {cell.get('id','?')}")

# ============================================================================
# 2. DELETE the patch_transform_std_fix cell (no longer needed)
# ============================================================================
nb["cells"] = [c for c in nb["cells"] if c.get("id") != "patch_transform_std_fix"]
print("  Deleted patch_transform_std_fix cell (no longer needed)")

# ============================================================================
# 3. REWRITE HEATMAP CELL
# ============================================================================
heatmap_source = [
    "# ===== MAPAS DE CALOR desde modelo PatchCore guardado =====\n",
    "from pathlib import Path\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import torch\n",
    "from PIL import Image\n",
    "import os, random\n",
    "\n",
    "# Import PatchCore internals\n",
    "from patchcore.patchcore import PatchCore\n",
    "from patchcore.common import FaissNN\n",
    "from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit\n",
    "\n",
    "def _find_model_path(results_dir):\n",
    '    """Busca el directorio donde PatchCore guardo el modelo."""\n',
    "    results_dir = Path(results_dir)\n",
    '    candidates = list(results_dir.rglob(\"patchcore_params.pkl\"))\n',
    "    if not candidates:\n",
    "        return None\n",
    "    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)\n",
    "    return candidates[0].parent\n",
    "\n",
    "def _predict_with_model(model_path, data_root, category):\n",
    '    """Carga modelo PatchCore y ejecuta prediccion sobre test set."""\n',
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "    pc = PatchCore(device)\n",
    "    pc.load_from_path(\n",
    "        load_path=str(model_path),\n",
    "        device=device,\n",
    "        nn_method=FaissNN(on_gpu=False, num_workers=4),\n",
    "    )\n",
    "    test_dataset = MVTecDataset(\n",
    "        str(data_root),\n",
    "        classname=category,\n",
    "        resize=256,\n",
    "        imagesize=224,\n",
    "        split=DatasetSplit.TEST,\n",
    "        seed=0,\n",
    "    )\n",
    "    test_loader = torch.utils.data.DataLoader(\n",
    "        test_dataset,\n",
    "        batch_size=1,\n",
    "        shuffle=False,\n",
    "        num_workers=2,\n",
    "        pin_memory=True,\n",
    "    )\n",
    "    scores, segmentations, labels, masks = pc.predict(test_loader)\n",
    "    image_paths = [x[2] for x in test_dataset.data_to_iterate]\n",
    "    return image_paths, scores, segmentations, labels\n",
    "\n",
    "def _find_seg_by_path(image_paths, segmentations, scores, stem):\n",
    "    for i, p in enumerate(image_paths):\n",
    "        if stem in str(p):\n",
    "            return segmentations[i], scores[i]\n",
    "    return None, None\n",
    "\n",
    "def show_anomaly_heatmaps(mvtec_root, roi_root, baseline_out, roi_out,\n",
    "                          category, defect_type=None, n=3, seed=1):\n",
    '    """Heatmaps reales cargando modelo PatchCore guardado."""\n',
    "    random.seed(seed)\n",
    '    test_root = mvtec_root / category / \"test\"\n',
    "    defect_types = (\n",
    "        [defect_type] if defect_type\n",
    '        else [d for d in sorted(os.listdir(test_root)) if d != \"good\"]\n',
    "    )\n",
    "\n",
    "    # Cargar modelos y predecir\n",
    '    print(\"Cargando modelo Baseline...\")\n',
    "    b_path = _find_model_path(baseline_out)\n",
    "    b_data = _predict_with_model(b_path, mvtec_root, category) if b_path else None\n",
    "\n",
    '    print(\"Cargando modelo SAM+ROI...\")\n',
    "    r_path = _find_model_path(roi_out)\n",
    "    r_data = _predict_with_model(r_path, roi_root, category) if r_path else None\n",
    "\n",
    "    for dtype in defect_types[:3]:\n",
    '        imgs = sorted((test_root / dtype).glob(\"*.png\"))\n',
    "        if not imgs:\n",
    "            continue\n",
    "        selected = random.sample(imgs, min(n, len(imgs)))\n",
    "        ncols = 5\n",
    "        fig, axes = plt.subplots(len(selected), ncols,\n",
    "                                 figsize=(ncols * 3.2, len(selected) * 3.2))\n",
    "        if len(selected) == 1:\n",
    "            axes = [axes]\n",
    "\n",
    "        for row, p in enumerate(selected):\n",
    '            orig_img = Image.open(p).convert(\"RGB\")\n',
    '            axes[row][0].imshow(orig_img)\n',
    '            axes[row][0].set_title(\"Original\", fontsize=8)\n',
    '            axes[row][0].axis(\"off\")\n',
    "\n",
    '            roi_img_path = roi_root / category / \"test\" / dtype / p.name\n',
    "            roi_img = None\n",
    "            if roi_img_path.exists():\n",
    '                roi_img = Image.open(roi_img_path).convert(\"RGB\")\n',
    "                axes[row][1].imshow(roi_img)\n",
    "            else:\n",
    '                axes[row][1].text(0.5, 0.5, \"N/D\", ha=\"center\", va=\"center\")\n',
    '            axes[row][1].set_title(\"ROI (SAM 1)\", fontsize=8)\n',
    '            axes[row][1].axis(\"off\")\n',
    "\n",
    "            # Baseline heatmap\n",
    "            if b_data is not None:\n",
    "                seg_b, score_b = _find_seg_by_path(b_data[0], b_data[1], b_data[2], p.stem)\n",
    "                if seg_b is not None:\n",
    "                    seg_resized = np.array(Image.fromarray(seg_b).resize(orig_img.size, Image.NEAREST))\n",
    '                    axes[row][2].imshow(orig_img, cmap=\"gray\", alpha=0.4)\n',
    '                    im = axes[row][2].imshow(seg_resized, cmap=\"jet\", alpha=0.6, vmin=0, vmax=seg_b.max())\n',
    "                    plt.colorbar(im, ax=axes[row][2], fraction=0.046, pad=0.04)\n",
    '                    axes[row][2].set_title(\"Baseline (score=\" + str(round(score_b,3)) + \")\" if score_b else \"Baseline\", fontsize=8)\n',
    "                else:\n",
    '                    axes[row][2].text(0.5, 0.5, \"N/D\", ha=\"center\", va=\"center\")\n',
    '                    axes[row][2].set_title(\"Baseline\", fontsize=8)\n',
    "            else:\n",
    '                axes[row][2].text(0.5, 0.5, \"N/D\", ha=\"center\", va=\"center\")\n',
    '                axes[row][2].set_title(\"Baseline\", fontsize=8)\n',
    '            axes[row][2].axis(\"off\")\n',
    "\n",
    "            # SAM+ROI heatmap\n",
    "            if r_data is not None:\n",
    "                seg_r, score_r = _find_seg_by_path(r_data[0], r_data[1], r_data[2], p.stem)\n",
    "                if seg_r is not None:\n",
    "                    ref_img = roi_img if roi_img is not None else orig_img\n",
    "                    seg_resized_r = np.array(Image.fromarray(seg_r).resize(ref_img.size, Image.NEAREST))\n",
    '                    axes[row][3].imshow(ref_img, cmap=\"gray\", alpha=0.4)\n',
    '                    im2 = axes[row][3].imshow(seg_resized_r, cmap=\"jet\", alpha=0.6, vmin=0, vmax=seg_r.max())\n',
    "                    plt.colorbar(im2, ax=axes[row][3], fraction=0.046, pad=0.04)\n",
    '                    axes[row][3].set_title(\"SAM+PC (score=\" + str(round(score_r,3)) + \")\" if score_r else \"SAM+PC\", fontsize=8)\n',
    "                else:\n",
    '                    axes[row][3].text(0.5, 0.5, \"N/D\", ha=\"center\", va=\"center\")\n',
    '                    axes[row][3].set_title(\"SAM+PatchCore\", fontsize=8)\n',
    "            else:\n",
    '                axes[row][3].text(0.5, 0.5, \"N/D\", ha=\"center\", va=\"center\")\n',
    '                axes[row][3].set_title(\"SAM+PatchCore\", fontsize=8)\n',
    '            axes[row][3].axis(\"off\")\n',
    "\n",
    '            gt_path = mvtec_root / category / \"ground_truth\" / dtype / (p.stem + \"_mask.png\")\n',
    "            if gt_path.exists():\n",
    '                gt = Image.open(gt_path).convert(\"L\")\n',
    "                gt_arr = np.array(gt).astype(float)\n",
    '                axes[row][4].imshow(gt_arr, cmap=\"Reds\", vmin=0, vmax=255)\n',
    "            else:\n",
    '                axes[row][4].text(0.5, 0.5, \"N/D\", ha=\"center\", va=\"center\")\n',
    '            axes[row][4].set_title(\"Ground Truth\", fontsize=8)\n',
    '            axes[row][4].axis(\"off\")\n',
    "\n",
    '        plt.suptitle(\"Mapas de calor -- \" + category + \" / \" + dtype, fontsize=11, fontweight=\"bold\")\n',
    "        plt.tight_layout()\n",
    '        save_path = \"/content/heatmaps_\" + category + \"_\" + dtype + \".png\"\n',
    '        plt.savefig(save_path, dpi=120, bbox_inches=\"tight\")\n',
    "        plt.show()\n",
    '        print(\"Figura guardada en \" + save_path)\n',
    "\n",
    "show_anomaly_heatmaps(MVTEC_ROOT, ROI_ROOT, BASELINE_OUT, ROI_OUT, CATEGORY)\n",
]

# ============================================================================
# 4. REWRITE ROC CELL
# ============================================================================
roc_source = [
    "# ===== CURVAS ROC desde modelo PatchCore guardado =====\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import roc_curve, auc\n",
    "import torch\n",
    "\n",
    "from patchcore.patchcore import PatchCore\n",
    "from patchcore.common import FaissNN\n",
    "from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit\n",
    "\n",
    "def _find_model_path_roc(results_dir):\n",
    "    results_dir = Path(results_dir)\n",
    '    candidates = list(results_dir.rglob(\"patchcore_params.pkl\"))\n',
    "    if not candidates:\n",
    "        return None\n",
    "    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)\n",
    "    return candidates[0].parent\n",
    "\n",
    "def _get_scores_and_labels(model_path, data_root, category):\n",
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "    pc = PatchCore(device)\n",
    "    pc.load_from_path(\n",
    "        load_path=str(model_path),\n",
    "        device=device,\n",
    "        nn_method=FaissNN(on_gpu=False, num_workers=4),\n",
    "    )\n",
    "    test_dataset = MVTecDataset(\n",
    "        str(data_root),\n",
    "        classname=category,\n",
    "        resize=256,\n",
    "        imagesize=224,\n",
    "        split=DatasetSplit.TEST,\n",
    "        seed=0,\n",
    "    )\n",
    "    test_loader = torch.utils.data.DataLoader(\n",
    "        test_dataset,\n",
    "        batch_size=1,\n",
    "        shuffle=False,\n",
    "        num_workers=2,\n",
    "        pin_memory=True,\n",
    "    )\n",
    "    scores, segmentations, labels, masks = pc.predict(test_loader)\n",
    '    return np.array(scores, dtype=float), np.array(labels, dtype=int)\n',
    "\n",
    "def plot_roc_curves(baseline_out, roi_out, category):\n",
    "    b_path = _find_model_path_roc(baseline_out)\n",
    "    r_path = _find_model_path_roc(roi_out)\n",
    "\n",
    "    if b_path is None or r_path is None:\n",
    "        fig, ax = plt.subplots(figsize=(10, 4))\n",
    '        ax.axis(\"off\")\n',
    '        ax.text(0.5, 0.5, \"No se encontro el modelo guardado. Ejecute PatchCore primero.\", ha=\"center\", va=\"center\")\n',
    "        plt.tight_layout()\n",
    '        plt.savefig(\"/content/roc_curves.png\", dpi=150, bbox_inches=\"tight\")\n',
    "        plt.show()\n",
    "        return\n",
    "\n",
    '    print(\"Cargando modelo Baseline para ROC...\")\n',
    "    b_scores, b_labels = _get_scores_and_labels(b_path, MVTEC_ROOT, category)\n",
    '    print(\"Cargando modelo SAM+ROI para ROC...\")\n',
    "    r_scores, r_labels = _get_scores_and_labels(r_path, ROI_ROOT, category)\n",
    "\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    '    for ax, scores, labels, ttl in [(axes[0], b_scores, b_labels, \"Curva ROC -- Baseline\"),\n',
    '                                    (axes[1], r_scores, r_labels, \"Curva ROC -- SAM 1 + PatchCore\")]:\n',
    "        valid = np.isfinite(scores)\n",
    "        scores, labels = scores[valid], labels[valid]\n",
    "        if len(np.unique(labels)) < 2:\n",
    '            ax.text(0.5, 0.5, \"Solo una clase presente\", ha=\"center\", va=\"center\")\n',
    "            ax.set_title(ttl)\n",
    "            continue\n",
    "        fpr, tpr, _ = roc_curve(labels, scores)\n",
    "        roc_auc = auc(fpr, tpr)\n",
    '        ax.plot(fpr, tpr, lw=2, label=\"AUC = \" + str(round(roc_auc, 4)))\n',
    '        ax.plot([0, 1], [0, 1], \"k--\", lw=1, label=\"Random\")\n',
    '        ax.set_xlabel(\"False Positive Rate\")\n',
    '        ax.set_ylabel(\"True Positive Rate\")\n',
    "        ax.set_title(ttl)\n",
    "        ax.legend()\n",
    "        ax.grid(True, alpha=0.3)\n",
    "\n",
    '    plt.suptitle(\"Curvas ROC -- \" + category + \" (MVTec AD)\", fontsize=12, fontweight=\"bold\")\n',
    "    plt.tight_layout()\n",
    '    plt.savefig(\"/content/roc_curves.png\", dpi=150, bbox_inches=\"tight\")\n',
    "    plt.show()\n",
    '    print(\"Figura guardada en /content/roc_curves.png\")\n',
    "\n",
    "plot_roc_curves(BASELINE_OUT, ROI_OUT, CATEGORY)\n",
]

# Apply to the right cells
for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid == "wblQUPaMbSoW":
        cell["source"] = heatmap_source
        print("  Heatmap cell rewritten")
    elif cid == "6UdoDqJTbSoX":
        cell["source"] = roc_source
        print("  ROC cell rewritten")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"\nDone. Total cells: {len(nb['cells'])}")
