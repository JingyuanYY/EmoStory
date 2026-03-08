import cv2
import numpy as np
import os

from skimage.morphology import remove_small_objects
import torch


def top_k_center_shrink_delete_peak(
        heat,
        min_size=15,
):
    border = 3
    image = heat.copy()
    image[:border, :] = 0
    image[-border:, :] = 0
    image[:, :border] = 0
    image[:, -border:] = 0

    H, W = image.shape
    weighted = image

    # 二值化 + 去碎点
    new_thresh = np.quantile(weighted, 0.9)
    mask = weighted >= new_thresh
    mask = remove_small_objects(mask, min_size=min_size)

    return mask.astype(bool)


def mask_interpolation_save(mask, save_path, H=512, W=1024):
    mask = cv2.resize(
        mask,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(save_path, mask)
    return mask


# Save the attention mask corresponding to the tokens
def visualize_prompt_attention_by_idx_cv(
        attn_map, step, batch, token_idx, title, output_path="", h=32, w=64, normalize=True, cmap='inferno',
        is_ref=0
):
    title = title.replace(" ", "_")
    #show_step = [5, 10, 15, 20]

    step_file_path = f"{output_path}/step={step}"
    os.makedirs(step_file_path, exist_ok=True)

    attn_map = attn_map[batch].detach().cpu().float()

    heat = (
        attn_map[:, token_idx]
        .mean(dim=-1)
        .reshape(h, w)
        .numpy()
    )

    if normalize:
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    mask = top_k_center_shrink_delete_peak(heat)

    heat = cv2.resize(
        heat,
        (1024, 512),
        interpolation=cv2.INTER_LINEAR
    )

    COLORMAP_DICT = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "jet": cv2.COLORMAP_JET,
    }
    cv_cmap = COLORMAP_DICT.get(cmap, cv2.COLORMAP_INFERNO)

    heat_color = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_color, cv_cmap)

    image_path = os.path.join(
        step_file_path,
        f"cross_attn_batch={batch}_token={title}_step={step:02d}.png"
    )
    mask_interpolation_save(heat_color, image_path)
    print(f"[Saved] attention heatmap → {image_path}")

    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    mask_path = os.path.join(
        step_file_path,
        f"cross_attn_otzu_binarize_batch={batch}_token={title}_step={step:02d}.png"
    )
    mask_interpolation_save(mask_uint8, mask_path)
    print(f"[Saved] refined mask → {mask_path}")


# Save the mask shared by the value
def visualize_bottom_mask_and_matches_cv(
        bottom_labels: torch.Tensor,
        matched_top_indices: torch.Tensor,
        image_path_prefix: str,
        step: int,
        block: int,
        batch: int,
        H: int = 32,
        W: int = 64,
):
    os.makedirs(image_path_prefix, exist_ok=True)

    # ---------- bottom_labels ----------
    bottom_labels_np = bottom_labels.detach().cpu().numpy()

    if bottom_labels_np.ndim == 1:
        bottom_labels_np = bottom_labels_np.reshape(H, W)

    # 转成 0~255 的 uint8 灰度图
    bottom_img = (bottom_labels_np > 0).astype(np.uint8) * 255
    # n_bottom_img = (bottom_labels_np == 0).astype(np.uint8) * 255

    str_message = f"batch={batch}_step={step:02d}_block={block}"
    mask_path = os.path.join(
        image_path_prefix, f"bottom_labels_{str_message}.png"
    )
    # n_mask_path = os.path.join(
    #     image_path_prefix, f"n_bottom_labels_{str_message}.png"
    # )

    mask_interpolation_save(bottom_img, mask_path)

    # Inverted mask
    # mask_interpolation_save(n_bottom_img, n_mask_path)
    # cv2.imwrite(mask_path, bottom_img)
    # cv2.imwrite(n_mask_path, n_bottom_img)

    # ---------- matched_top_indices ----------
    match_mask = np.zeros((H * W,), dtype=np.uint8)
    matched_top_indices_np = (
        matched_top_indices.detach().cpu().numpy().astype(int)
    )

    match_mask[matched_top_indices_np] = 255
    match_mask = match_mask.reshape(H, W)

    match_path = os.path.join(
        image_path_prefix, f"matched_indices_{str_message}.png"
    )

    # cv2.imwrite(match_path, match_mask)
    mask_interpolation_save(match_mask, match_path)

    print(f"[Saved] {mask_path}")
    print(f"[Saved] {match_path}")