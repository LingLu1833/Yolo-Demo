"""
使用YOLO模型实时识别游戏窗口画面（截取中下方 640x640 区域）
按 ESC 退出
新增功能：检测到目标进入三个独立圆形区域时，分别自动按下对应按键（A/W/D）
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
from ultralytics import YOLO


# ─── 路径常量 ────────────────────────────────────────────
MODEL_PATH = Path("runs/detect/datasets/runs/train/weights/best.pt")
CONFIG_PATH = Path("window_config.json")

# ─── 颜色映射（为不同类别分配固定颜色） ─────────────────
CLASS_COLORS = [
    (0, 255, 0),     # 绿色
    (255, 0, 0),     # 蓝色
    (0, 0, 255),     # 红色
    (255, 255, 0),   # 青色
    (255, 0, 255),   # 品红
    (0, 255, 255),   # 黄色
    (128, 255, 0),   # 黄绿
    (255, 128, 0),   # 橙色
    (128, 0, 255),   # 紫色
    (0, 128, 255),   # 橙红
]

# 三个触发圆圈的颜色（BGR）
CIRCLE_COLORS = [
    (0, 255, 255),   # 黄
    (255, 255, 0),   # 青
    (255, 0, 255),   # 品红
]


def load_config(config_path):
    """加载窗口配置文件"""
    if not config_path.exists():
        print(f"[WARN] 配置文件未找到: {config_path.resolve()}", file=sys.stderr)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_game_window(window_title, window_class):
    """
    根据窗口标题和/或窗口类名查找游戏窗口句柄。
    支持部分标题匹配。
    """
    if not window_title and not window_class:
        print("[ERROR] window_title 和 window_class 均为空，请在配置文件中至少填写一个",
              file=sys.stderr)
        return None

    def enum_callback(hwnd, candidates):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        cls = win32gui.GetClassName(hwnd)

        title_match = (not window_title) or (window_title.lower() in title.lower())
        class_match = (not window_class) or (window_class.lower() == cls.lower())

        if title_match and class_match:
            candidates.append((hwnd, title, cls))
        return True

    candidates = []
    win32gui.EnumWindows(enum_callback, candidates)

    if not candidates:
        print(f"[ERROR] 未找到匹配的窗口 "
              f"(title='{window_title}', class='{window_class}')", file=sys.stderr)
        return None

    if len(candidates) > 1:
        print(f"[INFO] 找到多个匹配窗口，使用第一个:")
        for hwnd, t, c in candidates:
            print(f"       hwnd={hwnd}, title='{t}', class='{c}'")

    return candidates[0][0]


def capture_window(hwnd, crop_box=None):
    """
    截取指定窗口客户区的指定区域（或整个客户区，若 crop_box 为 None）。
    crop_box: (left, top, width, height) 相对于客户区坐标。
    返回 BGRA numpy 数组。
    """
    # 获取窗口客户区尺寸
    client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
    client_width = client_right - client_left
    client_height = client_bottom - client_top

    if client_width <= 0 or client_height <= 0:
        return None

    # 若未指定裁剪区域，则默认整个客户区
    if crop_box is None:
        crop_left, crop_top, crop_width, crop_height = 0, 0, client_width, client_height
    else:
        crop_left, crop_top, crop_width, crop_height = crop_box
        # 确保裁剪区域不超出客户区
        crop_left = max(0, min(crop_left, client_width - 1))
        crop_top = max(0, min(crop_top, client_height - 1))
        crop_width = max(1, min(crop_width, client_width - crop_left))
        crop_height = max(1, min(crop_height, client_height - crop_top))

    # 获取客户区 DC（坐标原点在客户区左上角）
    hwnd_dc = win32gui.GetDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, crop_width, crop_height)
    save_dc.SelectObject(bitmap)

    # 从客户区指定位置拷贝到 bitmap
    save_dc.BitBlt((0, 0), (crop_width, crop_height), mfc_dc,
                   (crop_left, crop_top), win32con.SRCCOPY)

    # 转为 numpy 数组
    bmp_info = bitmap.GetInfo()
    bmp_bits = bitmap.GetBitmapBits(True)
    img = np.frombuffer(bmp_bits, dtype=np.uint8)
    img = img.reshape((bmp_info["bmHeight"], bmp_info["bmWidth"], 4))

    # 清理 GDI 资源
    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return img


def draw_detections(frame, results, config):
    """在帧上绘制检测结果"""
    conf_threshold = config.get("confidence_threshold", 0.5)
    thickness = config.get("box_thickness", 2)
    font_scale = config.get("font_scale", 0.6)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, f"cls_{cls_id}")
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 构建标签文本
            parts = []
            if config.get("show_labels", True):
                parts.append(cls_name)
            if config.get("show_conf", True):
                parts.append(f"{conf:.2f}")
            label = " ".join(parts)

            if label:
                (lw, lh), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(thickness, 1))
                # 标签背景
                cv2.rectangle(frame,
                              (x1, y1 - lh - baseline - 4),
                              (x1 + lw, y1),
                              color, -1)
                cv2.putText(frame, label,
                            (x1, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255),
                            max(thickness - 1, 1))

    return frame


def press_key(key_char):
    """模拟按下并释放一个键"""
    vk_code = ord(key_char.upper())
    win32api.keybd_event(vk_code, 0, 0, 0)
    time.sleep(0.03)
    win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)


def is_point_in_circle(point, center, radius):
    """判断点是否在圆内"""
    px, py = point
    cx, cy = center
    return (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2


def main():
    # 加载配置
    config = load_config(CONFIG_PATH)

    window_title = config.get("window_title", "")
    window_class = config.get("window_class", "")

    # 查找游戏窗口
    hwnd = find_game_window(window_title, window_class)
    if hwnd is None:
        sys.exit(1)

    title = win32gui.GetWindowText(hwnd)
    print(f"[INFO] 已找到窗口: hwnd={hwnd}, title='{title}'")

    # 加载 YOLO 模型
    if not MODEL_PATH.exists():
        print(f"[ERROR] 模型文件未找到: {MODEL_PATH.resolve()}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] 加载模型: {MODEL_PATH.resolve()}")
    model = YOLO(str(MODEL_PATH))

    # ─── 裁剪区域设置 ─────────────────────────────────────
    crop_width = config.get("crop_width", 640)
    crop_height = config.get("crop_height", 640)
    crop_x_ratio = config.get("crop_x_ratio", 0.5)   # 水平居中
    crop_y_ratio = config.get("crop_y_ratio", 1.0)   # 垂直底部对齐（1.0=底部）

    # 获取客户区尺寸
    client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
    client_width = client_right - client_left
    client_height = client_bottom - client_top

    # 计算裁剪区域的左上角坐标（确保不超出客户区）
    crop_left = int((client_width - crop_width) * crop_x_ratio)
    crop_top = int((client_height - crop_height) * crop_y_ratio)
    crop_left = max(0, min(crop_left, client_width - crop_width))
    crop_top = max(0, min(crop_top, client_height - crop_height))
    crop_box = (crop_left, crop_top, crop_width, crop_height)

    print(f"[INFO] 截取区域: 左上角=({crop_left},{crop_top}) 尺寸={crop_width}x{crop_height}")

    # ─── 自动按键配置（三个独立圆圈） ─────────────────────
    action_enabled = config.get("action_enabled", False)
    action_circles_cfg = config.get("action_circles", [])
    # 确保是列表且元素个数不超过3（可自定义更多，但颜色循环）
    if not isinstance(action_circles_cfg, list):
        action_circles_cfg = []
    # 解析圆圈配置
    circles = []
    for i, circle_cfg in enumerate(action_circles_cfg):
        if not isinstance(circle_cfg, dict):
            continue
        cx = int(circle_cfg.get("center_x", crop_width // 2))
        cy = int(circle_cfg.get("center_y", crop_height // 2))
        r = int(circle_cfg.get("radius", 50))
        key = str(circle_cfg.get("key", "a"))
        circles.append({
            "center": (cx, cy),
            "radius": r,
            "key": key,
            "color": CIRCLE_COLORS[i % len(CIRCLE_COLORS)]
        })

    if action_enabled:
        print(f"[INFO] 自动按键已启用，共配置 {len(circles)} 个触发圆圈:")
        for i, c in enumerate(circles):
            print(f"       圆圈{i+1}: 圆心=({c['center'][0]},{c['center'][1]}), "
                  f"半径={c['radius']}, 按键='{c['key']}'")
    else:
        print("[INFO] 自动按键未启用（action_enabled=False）")

    # 为每个圆圈维护防抖状态
    prev_in_circle = [False] * len(circles)

    # 创建结果显示窗口
    display_scale = config.get("display_scale", 1.0)
    win_name = "YOLO Real-Time Detection (ESC=Exit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("[INFO] 开始实时检测，按 ESC 退出\n")

    # 性能统计
    frame_times = []
    fps_update_interval = 0.5  # 每 0.5 秒更新 FPS 显示
    last_fps_update = time.perf_counter()
    fps_display = 0.0
    window_sized = False  # 是否已根据画面设置过窗口大小

    try:
        while True:
            loop_start = time.perf_counter()

            # 截取窗口的指定区域
            frame_rgba = capture_window(hwnd, crop_box)
            if frame_rgba is None:
                print("[WARN] 截取窗口失败，重试中...")
                time.sleep(0.1)
                continue

            # BGRA → BGR（OpenCV 使用 BGR）
            frame = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

            # 首次获取画面后，根据实际尺寸设置显示窗口大小
            if not window_sized:
                h, w = frame.shape[:2]
                new_w = int(w * display_scale)
                new_h = int(h * display_scale)
                cv2.resizeWindow(win_name, new_w, new_h)
                window_sized = True

            # YOLO 检测（此时 frame 尺寸为裁剪后的尺寸，通常为 640x640）
            results = model(frame, verbose=False)

            # 绘制检测结果
            annotated = draw_detections(frame, results, config)

            # ─── 绘制所有触发圆圈 ─────────────────────────
            for circle in circles:
                color = circle["color"]
                center = circle["center"]
                radius = circle["radius"]
                key = circle["key"]
                cv2.circle(annotated, center, radius, color, 2)
                cv2.circle(annotated, center, 3, color, -1)
                # 在圆心旁标注按键
                cv2.putText(annotated, key.upper(),
                            (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ─── 检测目标是否进入各圆圈并触发对应按键 ─────
            if action_enabled and circles:
                # 收集所有检测框的中心点（置信度达标）
                target_centers = []
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf < config.get("confidence_threshold", 0.5):
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        target_centers.append((center_x, center_y))

                # 对每个圆圈独立判断
                for idx, circle in enumerate(circles):
                    current_in = False
                    for pt in target_centers:
                        if is_point_in_circle(pt, circle["center"], circle["radius"]):
                            current_in = True
                            break
                    # 从圈外进入圈内则触发
                    if current_in and not prev_in_circle[idx]:
                        print(f"[ACTION] 目标进入圆圈{idx+1}，按下按键 '{circle['key']}'")
                        press_key(circle["key"])
                    prev_in_circle[idx] = current_in

            # FPS 统计与显示
            frame_times.append(time.perf_counter() - loop_start)
            now = time.perf_counter()
            if now - last_fps_update >= fps_update_interval:
                last_fps_update = now
                if frame_times:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    fps_display = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                frame_times.clear()

            if config.get("show_fps", True):
                fps_text = f"FPS: {fps_display:.1f}"
                cv2.putText(annotated, fps_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 缩放显示
            if display_scale != 1.0:
                h, w = annotated.shape[:2]
                new_w, new_h = int(w * display_scale), int(h * display_scale)
                display_frame = cv2.resize(annotated, (new_w, new_h))
            else:
                display_frame = annotated

            cv2.imshow(win_name, display_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")

    finally:
        cv2.destroyAllWindows()
        # 确保窗口置前以便关闭
        for _ in range(5):
            cv2.waitKey(1)
        print("[DONE] 检测已停止")


if __name__ == "__main__":
    main()