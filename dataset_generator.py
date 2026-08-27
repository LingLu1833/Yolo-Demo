from config import *

import os
import random
import math

import numpy as np

from PIL import Image, ImageEnhance, ImageFilter

_background_image_cache = None


def _get_background_image():
    global _background_image_cache
    if _background_image_cache is None:
        path = os.path.join(IMAGES_DIR, "background.png")
        if os.path.exists(path):
            _background_image_cache = Image.open(path).convert("RGB")
        else:
            _background_image_cache = False
    return _background_image_cache if _background_image_cache is not False else None


def _load_template_image(filepath):
    img = Image.open(filepath)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    return img


def load_templates(templates_dir):
    templates_dict = {}
    class_names = []

    png_files = sorted([
        f for f in os.listdir(templates_dir)
        if f.lower().endswith(".png")
    ])

    for class_id, filename in enumerate(png_files):
        class_name = os.path.splitext(filename)[0]
        class_names.append(class_name)

        filepath = os.path.join(templates_dir, filename)
        img = _load_template_image(filepath)
        templates_dict[class_id] = img

    return templates_dict, class_names


def _foreground_color_augment(img):
    if img.mode == "RGBA":
        alpha = img.split()[-1]
    else:
        alpha = None

    rgb_img = img.convert("RGB")
    dh, ds, dv = FOREGROUND_HSV_SHIFT
    shift_h = random.uniform(-dh, dh)
    shift_s = random.uniform(-ds, ds)
    shift_v = random.uniform(-dv, dv)

    import colorsys
    r, g, b = rgb_img.split()
    hsv_pixels = []
    for rv, gv, bv in zip(r.getdata(), g.getdata(), b.getdata()):
        h, s, v = colorsys.rgb_to_hsv(rv / 255.0, gv / 255.0, bv / 255.0)
        h = (h + shift_h) % 1.0
        s = max(0.0, min(1.0, s + shift_s))
        v = max(0.0, min(1.0, v + shift_v))
        nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
        hsv_pixels.append((int(nr * 255), int(ng * 255), int(nb * 255)))

    result = Image.new("RGB", rgb_img.size)
    result.putdata(hsv_pixels)

    brightness_factor = random.uniform(*FOREGROUND_BRIGHTNESS_RANGE)
    result = ImageEnhance.Brightness(result).enhance(brightness_factor)

    contrast_factor = random.uniform(*FOREGROUND_CONTRAST_RANGE)
    result = ImageEnhance.Contrast(result).enhance(contrast_factor)

    if alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(alpha)

    return result


def random_transform(foreground_img):
    angle = random.uniform(*ROTATION_RANGE)
    rotated = foreground_img.rotate(angle, expand=True, resample=Image.BICUBIC)

    scale = random.uniform(*SCALE_RANGE)
    new_w = max(1, int(rotated.width * scale))
    new_h = max(1, int(rotated.height * scale))
    scaled = rotated.resize((new_w, new_h), Image.LANCZOS)

    if FOREGROUND_COLOR_AUGMENT:
        scaled = _foreground_color_augment(scaled)

    return scaled


def _crop_to_content(img):
    if img.mode != "RGBA":
        return img

    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return img

    return img.crop(bbox)


def _rotate_background(array, angle):
    img = Image.fromarray(array, "RGB")
    rotated = img.rotate(math.degrees(angle), expand=False, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    rw, rh = rotated.size
    ow, oh = array.shape[1], array.shape[0]
    left = (rw - ow) // 2
    top = (rh - oh) // 2
    crop = rotated.crop((left, top, left + ow, top + oh))

    result = np.array(crop)
    mask = (result[:, :, 0] == 0) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
    result[mask] = array[mask]
    return result


def random_background():
    bg_type = random.choices(
        population=list(BACKGROUND_TYPE_WEIGHTS.keys()),
        weights=list(BACKGROUND_TYPE_WEIGHTS.values()),
        k=1
    )[0]

    w, h = IMAGE_SIZE
    array = np.zeros((h, w, 3), dtype=np.uint8)

    if bg_type == "solid":
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
        array[:, :] = color

    elif bg_type == "linear_gradient":
        n_colors = random.randint(2, 3)
        colors = np.array([np.random.randint(0, 256, 3) for _ in range(n_colors)], dtype=np.float64)

        angle = random.uniform(0, 2 * math.pi)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        xs = np.arange(w)
        ys = np.arange(h)
        xx, yy = np.meshgrid(xs, ys)

        proj = cos_a * xx + sin_a * yy
        proj_min = proj.min()
        proj_max = proj.max()
        t = (proj - proj_min) / max(proj_max - proj_min, 1e-8)

        if n_colors == 2:
            array = ((1 - t[:, :, None]) * colors[0] + t[:, :, None] * colors[1]).astype(np.uint8)
        else:
            mask1 = t < 0.5
            mask2 = ~mask1
            t1 = t / 0.5
            t2 = (t - 0.5) / 0.5
            result = np.zeros((h, w, 3), dtype=np.float64)
            result[mask1] = ((1 - t1[mask1, None]) * colors[0] + t1[mask1, None] * colors[1])
            result[mask2] = ((1 - t2[mask2, None]) * colors[1] + t2[mask2, None] * colors[2])
            array = result.astype(np.uint8)

    elif bg_type == "radial_gradient":
        n_colors = random.randint(2, 3)
        colors = np.array([np.random.randint(0, 256, 3) for _ in range(n_colors)], dtype=np.float64)

        cx = random.uniform(0, w - 1)
        cy = random.uniform(0, h - 1)

        xs = np.arange(w)
        ys = np.arange(h)
        xx, yy = np.meshgrid(xs, ys)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        dist_min = dist.min()
        dist_max = dist.max()
        t = (dist - dist_min) / max(dist_max - dist_min, 1e-8)

        if n_colors == 2:
            array = ((1 - t[:, :, None]) * colors[0] + t[:, :, None] * colors[1]).astype(np.uint8)
        else:
            mask1 = t < 0.5
            mask2 = ~mask1
            t1 = t / 0.5
            t2 = (t - 0.5) / 0.5
            result = np.zeros((h, w, 3), dtype=np.float64)
            result[mask1] = ((1 - t1[mask1, None]) * colors[0] + t1[mask1, None] * colors[1])
            result[mask2] = ((1 - t2[mask2, None]) * colors[1] + t2[mask2, None] * colors[2])
            array = result.astype(np.uint8)

    elif bg_type == "noise":
        block_size = random.randint(8, 32)
        bw = max(1, w // block_size)
        bh = max(1, h // block_size)
        low_res = np.random.randint(48, 208, (bh, bw, 3), dtype=np.uint8)

        upscaled = np.repeat(np.repeat(low_res, block_size, axis=0), block_size, axis=1)
        uh, uw = upscaled.shape[:2]

        if uh < h:
            upscaled = np.pad(upscaled, ((0, h - uh), (0, 0), (0, 0)), mode="edge")
        else:
            upscaled = upscaled[:h, :, :]
        if uw < w:
            upscaled = np.pad(upscaled, ((0, 0), (0, w - uw), (0, 0)), mode="edge")
        else:
            upscaled = upscaled[:, :w, :]

        noise = np.random.randint(-20, 21, (h, w, 3), dtype=np.int16)
        array = np.clip(upscaled.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif bg_type == "grid":
        bg_color = np.random.randint(30, 180, 3, dtype=np.uint8)
        line_color = np.random.randint(0, 256, 3, dtype=np.uint8)

        array[:, :] = bg_color

        grid_spacing = random.randint(20, 80)
        line_width = random.randint(1, 4)

        for x in range(0, w, grid_spacing):
            array[:, x:x + line_width] = line_color
        for y in range(0, h, grid_spacing):
            array[y:y + line_width, :] = line_color

        angle = random.uniform(0, 2 * math.pi)
        array = _rotate_background(array, angle)
        array = Image.fromarray(array, "RGB")
        return array

    elif bg_type == "checker":
        color_a = np.random.randint(30, 180, 3, dtype=np.uint8)
        color_b = np.random.randint(30, 180, 3, dtype=np.uint8)

        cell_size = random.randint(16, 64)
        for cy in range(0, h, cell_size):
            for cx in range(0, w, cell_size):
                is_a = ((cy // cell_size) + (cx // cell_size)) % 2 == 0
                y_end = min(cy + cell_size, h)
                x_end = min(cx + cell_size, w)
                array[cy:y_end, cx:x_end] = color_a if is_a else color_b

        angle = random.uniform(0, 2 * math.pi)
        array = _rotate_background(array, angle)
        array = Image.fromarray(array, "RGB")
        return array

    elif bg_type == "from_image":
        bg_img = _get_background_image()
        if bg_img is not None and bg_img.width >= w and bg_img.height >= h:
            max_x = bg_img.width - w
            max_y = bg_img.height - h
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)
            return bg_img.crop((crop_x, crop_y, crop_x + w, crop_y + h))
        # fall through to return the default array

    return Image.fromarray(array, "RGB")


def _bbox_iou(bbox_a, bbox_b):
    ax1 = bbox_a[0] - bbox_a[2] / 2.0
    ay1 = bbox_a[1] - bbox_a[3] / 2.0
    ax2 = bbox_a[0] + bbox_a[2] / 2.0
    ay2 = bbox_a[1] + bbox_a[3] / 2.0

    bx1 = bbox_b[0] - bbox_b[2] / 2.0
    by1 = bbox_b[1] - bbox_b[3] / 2.0
    bx2 = bbox_b[0] + bbox_b[2] / 2.0
    by2 = bbox_b[1] + bbox_b[3] / 2.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = bbox_a[2] * bbox_a[3]
    area_b = bbox_b[2] * bbox_b[3]
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def get_random_position(fg_w, fg_h, bg_w, bg_h, existing_bboxes, max_overlap_ratio):
    yolo_bbox = None
    paste_x = 0
    paste_y = 0

    max_x = bg_w - fg_w
    max_y = bg_h - fg_h

    if max_x < 0 or max_y < 0:
        return None, None

    for _ in range(50):
        paste_x = random.randint(0, max_x)
        paste_y = random.randint(0, max_y)

        x_center = (paste_x + fg_w / 2.0) / bg_w
        y_center = (paste_y + fg_h / 2.0) / bg_h
        width = fg_w / bg_w
        height = fg_h / bg_h

        yolo_bbox = (x_center, y_center, width, height)

        overlap_ok = True
        for existing_bbox in existing_bboxes:
            iou = _bbox_iou(yolo_bbox, existing_bbox)
            if iou > max_overlap_ratio:
                overlap_ok = False
                break

        if overlap_ok:
            return yolo_bbox, (paste_x, paste_y)

    return None, None


def composite_image(background, foreground, paste_x, paste_y):
    bg = background.copy()
    if foreground.mode == "RGBA":
        bg.paste(foreground, (paste_x, paste_y), foreground)
    else:
        bg.paste(foreground, (paste_x, paste_y))
    return bg


def _post_process(img):
    if not POST_PROCESS_AUGMENT:
        return img

    blur_sigma = random.uniform(*POST_BLUR_SIGMA_RANGE)
    if blur_sigma > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    noise_std = random.uniform(*POST_NOISE_STD_RANGE)
    if noise_std > 0.5:
        arr = np.array(img, dtype=np.float32)
        gaussian_noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
        arr = np.clip(arr + gaussian_noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")

    jpeg_quality = random.randint(*POST_JPEG_QUALITY_RANGE)
    if jpeg_quality < 100:
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        img = Image.open(buf)
        img.load()

    return img


def _validate_bbox(bbox):
    x_center, y_center, width, height = bbox
    left = x_center - width / 2.0
    right = x_center + width / 2.0
    top = y_center - height / 2.0
    bottom = y_center + height / 2.0
    return left >= 0.0 and right <= 1.0 and top >= 0.0 and bottom <= 1.0


def generate_dataset():
    templates_dict, class_names = load_templates(TEMPLATES_DIR)
    num_classes = len(class_names)
    print(f"[INFO] Loaded {num_classes} templates: {class_names}")

    train_images_dir = os.path.join(OUTPUT_DIR, "train", "images")
    train_labels_dir = os.path.join(OUTPUT_DIR, "train", "labels")
    val_images_dir = os.path.join(OUTPUT_DIR, "val", "images")
    val_labels_dir = os.path.join(OUTPUT_DIR, "val", "labels")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    total_images = num_classes * NUM_IMAGES_PER_CLASS
    train_count = int(total_images * TRAIN_RATIO)
    global_idx = 0
    generated_in_class = 0

    for class_id in range(num_classes):
        class_name = class_names[class_id]
        print(f"\n[INFO] Generating images for class [{class_id + 1}/{num_classes}]: {class_name}")

        for i in range(NUM_IMAGES_PER_CLASS):
            bg_img = random_background()

            n_objects = random.randint(*OBJECTS_PER_IMAGE)

            obj_labels = []
            canvas = bg_img.copy()
            existing_bboxes = []

            for obj_idx in range(n_objects):
                if obj_idx == 0:
                    obj_class_id = class_id
                else:
                    obj_class_id = random.randint(0, num_classes - 1)

                foreground = templates_dict[obj_class_id]
                transformed_fg = random_transform(foreground)
                transformed_fg = _crop_to_content(transformed_fg)

                position_result = get_random_position(
                    transformed_fg.width,
                    transformed_fg.height,
                    IMAGE_SIZE[0],
                    IMAGE_SIZE[1],
                    existing_bboxes,
                    MAX_OVERLAP_RATIO,
                )

                if position_result[0] is None:
                    continue

                yolo_bbox, paste_pos = position_result

                if not _validate_bbox(yolo_bbox):
                    continue

                canvas = composite_image(canvas, transformed_fg, paste_pos[0], paste_pos[1])
                obj_labels.append((obj_class_id, yolo_bbox))
                existing_bboxes.append(yolo_bbox)

            if not obj_labels:
                continue

            is_train = global_idx < train_count
            subdir_images = train_images_dir if is_train else val_images_dir
            subdir_labels = train_labels_dir if is_train else val_labels_dir

            image_filename = f"{class_name}_{i:06d}.jpg"
            label_filename = f"{class_name}_{i:06d}.txt"

            image_path = os.path.join(subdir_images, image_filename)
            label_path = os.path.join(subdir_labels, label_filename)

            canvas = _post_process(canvas)
            canvas.save(image_path, "JPEG", quality=95)

            with open(label_path, "w", encoding="utf-8") as f:
                for obj_cls_id, bbox in obj_labels:
                    f.write(
                        f"{obj_cls_id} "
                        f"{bbox[0]:.6f} {bbox[1]:.6f} "
                        f"{bbox[2]:.6f} {bbox[3]:.6f}\n"
                    )

            global_idx += 1
            generated_in_class += 1

            if generated_in_class % 10 == 0 or generated_in_class == NUM_IMAGES_PER_CLASS:
                pct_class = generated_in_class / NUM_IMAGES_PER_CLASS * 100
                pct_total = global_idx / total_images * 100
                bar_len = 30
                filled = int(bar_len * generated_in_class / NUM_IMAGES_PER_CLASS)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"\r  [{bar}] {generated_in_class}/{NUM_IMAGES_PER_CLASS} ({pct_class:.1f}%) "
                    f"| Total: {global_idx}/{total_images} ({pct_total:.1f}%)",
                    end=""
                )

        generated_in_class = 0
        print()

    print(f"[INFO] Dataset generation complete. "
          f"Train: {min(global_idx, train_count)}, Val: {max(0, global_idx - train_count)}")

    generate_data_yaml(OUTPUT_DIR, class_names)


def generate_data_yaml(output_dir, class_names):
    yaml_path = os.path.join(output_dir, "data.yaml")
    names_str = ", ".join(f"'{name}'" for name in class_names)
    yaml_content = (
        f"path: {output_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names_str}]\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[INFO] Generated data.yaml at: {yaml_path}")