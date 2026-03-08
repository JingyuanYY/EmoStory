import cv2
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
from skimage.measure import label


def find_token_indices_via_offsets_batch(phrase, prompts, tokenizer):
    """
    prompts: List[str]
    phrase: str
    return: List[List[int] or None]，长度 == len(prompts)
    """
    def generate_phrase_variants(phrase: str):
        p = phrase.lower()
        variants = [p]

        # ---------- remove ----------
        if p.endswith("es"):
            variants.append(p[:-2])
        if p.endswith("s"):
            variants.append(p[:-1])

        # ---------- add ----------
        variants.append(p + "s")
        variants.append(p + "es")

        # 去重但保持顺序
        seen = set()
        ordered = []
        for v in variants:
            if v not in seen:
                ordered.append(v)
                seen.add(v)
        return ordered

    phrase_variants = generate_phrase_variants(phrase)

    results = []

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        offsets = enc.offset_mapping[0].tolist()
        prompt_lower = prompt.lower()

        # Try different forms of phrases one by one
        match_start = match_end = None
        for pv in phrase_variants:
            pos = prompt_lower.find(pv)
            if pos != -1:
                match_start = pos
                match_end = pos + len(pv)
                break

        # unfinded
        if match_start is None:
            results.append(None)
            continue

        # Match token indices based on char offset
        token_indices = []
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:  # padding
                continue
            # token 的 char span 与 phrase span 重叠
            if s < match_end and e > match_start:
                token_indices.append(i)

        results.append(token_indices if token_indices else None)

    return results


def get_element_tokens_mask(p_mask, emotion_token_indices):
    for element in emotion_token_indices["elements"]:
        idx_list = emotion_token_indices["elements"][element]
        for b, idx in enumerate(idx_list):
            if idx is not None:
                p_mask[b, idx] = 1  # p_mask -> [B, Tk]
    p_mask = p_mask[:, None, None, :]  # [B,1,1,Tk]
    return p_mask


# debug Check the tokens corresponding subscripts
def debug_recover_all(all_token_indices, prompts, tokenizer):
    def recover_tokens_from_indices(prompt, token_indices, tokenizer):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0].tolist()

        if token_indices is None:
            return None

        tokens = [encoded[i] for i in token_indices]
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        return text.strip()

    for group_name, group in all_token_indices.items():
        print(f"\n===== {group_name.upper()} =====")

        for phrase, index_groups in group.items():
            print(f"\n Phrase: {phrase}")

            for i, idxs in enumerate(index_groups):
                recovered = recover_tokens_from_indices(
                    prompts[i],
                    idxs,
                    tokenizer
                )
                print(f"=== Prompt[{i}] idx={idxs} -> '{recovered}'")


def get_cc_watershed(image, min_size=9):
    """
    Watershed + remove small components (default: removes clusters smaller than 3x3).

    Args:
        image (np.ndarray): Input binary or grayscale image
        min_size (int): Minimum number of pixels per component to keep

    Returns:
        labels (np.ndarray): Labeled connected components after watershed
    """
    # We ignore a small margin on the borders of the image, as for Flux it usually is
    # decorative and does not contain image content. Without this, Reciprocal Attention can
    # be more noisy.
    margin = 3
    image[:margin, :] = 0
    image[-margin:, :] = 0
    image[:, :margin] = 0
    image[:, -margin:] = 0

    binary, _ = otzu_binarize(image)
    # Otsu Binarization: Automatically finds the bright/dark threshold to separate the attention heatmap into "focused" and "unfocused" regions
    # remove_small_objects: Remove overly small noise regions
    # Watershed: If there are multiple focus clusters, segment them into independent connected regions
    binary = remove_small_objects(binary, min_size=min_size)

    return binary


def count_clusters(mask):
    """
    统计输入二值 mask 中的连通簇数量。
    Args:
        mask (np.ndarray): 二值化后的图像（0/1 或 bool）
    Returns:
        num_clusters (int): 连通区域的数量
    """
    # 确保输入为 bool 类型
    mask = mask.astype(bool)
    # 连通区域标记，connectivity=2 表示8邻域（更适合图像）
    labeled_mask = label(mask, connectivity=2)
    # 最大 label 值即为簇数量（背景为0）
    num_clusters = labeled_mask.max()

    return num_clusters


def otzu_binarize(image):
    # Normalize image between 0 and 1
    image = (image - image.min()) / (image.max() - image.min())
    # Threshold the image to separate clumps
    thresh = threshold_otsu(image)
    _, binary = cv2.threshold(image, thresh, 1, cv2.THRESH_BINARY)
    binary = (binary * 255).astype(bool)
    return binary, thresh


def topk_binarize(image, topk=0.05):
    """ 取图像中最亮 topk 百分比作为前景 (e.g. topk=0.05 表示取前 5%) """
    image = (image - image.min()) / (image.max() - image.min())
    thresh = np.quantile(image, 1 - topk)
    binary = image >= thresh
    return binary, thresh


def center_weight_shrink(image, binary, center, strength=0.5):
    """
    根据中心位置，对越远区域衰减亮度
    center: (y, x)
    """
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    dist = dist / dist.max()
    weighted = image * np.exp(-dist * strength)
    return weighted

