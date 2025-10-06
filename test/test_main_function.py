#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal stateless two-stage shooter helpers for RoboMaster camera.
- Stage 0: scan newest frame to build (pos, ori, cs_list) + centers map
- Stage 1: coarse gimbal move using FOV/pixel mapping
- Stage 2: fine PID centering inside a small center ROI and fire
This module does NOT save any state; callers (e.g., test_lab1) can store results.
"""

import time
import math
import numpy as np
import cv2


# -------------------- PID --------------------
class PID:
    def __init__(self, kp=0.2, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=12.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integ_clip = float(integ_clip)
        self.out_clip = float(out_clip)
        self.integ = 0.0
        self.prev = 0.0

    def step(self, err, dt):
        dt = max(1e-3, float(dt))
        self.integ += err * dt
        self.integ = max(-self.integ_clip, min(self.integ, self.integ_clip))
        deriv = (err - self.prev) / dt
        self.prev = err
        out = self.kp * err + self.ki * self.integ + self.kd * deriv
        return max(-self.out_clip, min(self.out_clip, out))


# -------------------- internal helpers --------------------

def _to_lab1_shape(name: str) -> str:
    """Canonicalize rectangle names to match test_lab1.py outputs.
    Returns one of: 'circle' | 'square' | 'horizontal_rectangle' | 'vertical_rectangle'
    Accepts synonyms like rectangle_h/rectangle_v and passes through known names.
    """
    if not name:
        return name
    mapping = {
        'horizontal_rectangle': 'horizontal_rectangle',
        'vertical_rectangle': 'vertical_rectangle',
        'rectangle_h': 'horizontal_rectangle',
        'rectangle_v': 'vertical_rectangle',
        'square': 'square',
        'circle': 'circle',
    }
    return mapping.get(name, name)


def _build_mask(hsv, color_range):
    """Support single range or tuple-of-ranges (e.g., red)."""
    multi = (
        isinstance(color_range, (tuple, list)) and len(color_range) > 0
        and isinstance(color_range[0][0], (list, tuple))
    )
    if multi:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in color_range:
            lo = np.array(lo, np.uint8)
            hi = np.array(hi, np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return mask
    lo, hi = color_range
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def _classify_shape(cnt):
    """Lab1-like shape classification with area/vertices and ratios.
    Returns: 'circle' | 'square' | 'rectangle_h' | 'rectangle_v' | None
    """
    area = cv2.contourArea(cnt)
    if area < 150:  # lower threshold like lab1
        return None
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return None
    # circle by circularity or many vertices
    circ = 4.0 * math.pi * area / max(peri * peri, 1e-6)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # tighter epsilon
    if circ > 0.78 or len(approx) >= 8:
        return 'circle'
    v = len(approx)
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            return None
        r = w / float(h)
        if 0.9 <= r <= 1.1:
            return 'square'
        return 'rectangle_h' if r > 1.1 else 'rectangle_v'
    # ignore triangles or unknown
    return None


# -------------------- Stage 0 --------------------

def ts_scan_entry(ep_robot, colors_map, color_ranges, pos, ori, roi_max=(800, 450)):
    """
    Scan newest frame and return ((pos, ori, cs_list), centers_map).
    - cs_list: [(color, shape), ...] compatible with test_lab1
    - centers_map: { (color,shape): (cx,cy) }
    Stateless; no persistence.
    """
    try:
        img = ep_robot.camera.read_cv2_image(strategy="newest", timeout=1)
    except Exception:
        img = None
    if img is None:
        return (pos, ori, []), {}

    h, w = img.shape[:2]
    rw, rh = min(roi_max[0], w), min(roi_max[1], h)
    rx, ry = (w - rw) // 2, (h - rh) // 2
    roi = img[ry:ry + rh, rx:rx + rw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    reverse_color = {v: k for k, v in colors_map.items()}

    cs_list = []
    centers = {}
    for color_key, cr in color_ranges.items():
        mask = _build_mask(hsv, cr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            shp = _classify_shape(cnt)
            shp = _to_lab1_shape(shp)
            if not shp:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = rx + x + bw // 2, ry + y + bh // 2
            user_color = reverse_color.get(color_key, color_key)
            key = (user_color, shp)
            cs_list.append(key)
            centers.setdefault(key, (cx, cy))

    return (pos, ori, cs_list), centers


# -------------------- Stage 1 --------------------

def ts_stage1_move(ep_robot, approx_center, h_fov_deg=96.0, v_fov_deg=54.0, gain=0.95, extra_pitch_deg=0.0):
    """Coarse single gimbal move using FOV projection from pixel error."""
    if approx_center is None:
        return 0.0, 0.0
    try:
        ep_robot.gimbal.recenter().wait_for_completed()
        time.sleep(0.2)
    except Exception:
        pass
    try:
        img = ep_robot.camera.read_cv2_image(strategy="newest", timeout=1)
    except Exception:
        img = None
    if img is None:
        return 0.0, 0.0
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    tx, ty = approx_center
    dx = tx - cx
    dy = ty - cy
    yaw_deg = gain * (dx / float(w)) * h_fov_deg
    pitch_deg = -gain * (dy / float(h)) * v_fov_deg + extra_pitch_deg
    yaw_deg = max(-30, min(30, yaw_deg))
    pitch_deg = max(-20, min(20, pitch_deg))
    try:
        ep_robot.gimbal.move(yaw=yaw_deg, pitch=pitch_deg).wait_for_completed()
        time.sleep(0.2)
    except Exception:
        pass
    return yaw_deg, pitch_deg


# -------------------- Stage 2 --------------------

def ts_stage2_pid_fire(ep_robot, target_color, target_shape, color_ranges, colors_map,
                       timeout_s=6.0, roi_size=220,
                       pid_gains=(0.20, 0.00, 0.00), clips=(12.0, 10.0)):
    """Fine PID centering in a small center ROI, then fire. Returns True/False."""
    kp, ki, kd = pid_gains
    pid_yaw = PID(kp=kp, ki=ki, kd=kd, integ_clip=40.0, out_clip=clips[0])
    pid_pitch = PID(kp=kp, ki=ki, kd=kd, integ_clip=40.0, out_clip=clips[1])
    color_key = colors_map.get(target_color, target_color)
    t_end = time.time() + timeout_s
    last_t = time.monotonic()
    target_shape_norm = _to_lab1_shape(target_shape)
    while time.time() < t_end:
        try:
            img = ep_robot.camera.read_cv2_image(strategy="newest", timeout=1)
        except Exception:
            img = None
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        rw, rh = min(roi_size, w), min(roi_size, h)
        rx, ry = max(0, cx - rw // 2), max(0, cy - rh // 2)
        roi = img[ry:ry + rh, rx:rx + rw]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if color_key not in color_ranges:
            return False
        mask = _build_mask(hsv, color_ranges[color_key])
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_c = None
        for cnt in contours:
            shp = _classify_shape(cnt)
            shp = _to_lab1_shape(shp)
            if shp == target_shape_norm:
                x, y, bw, bh = cv2.boundingRect(cnt)
                target_c = (rx + x + bw // 2, ry + y + bh // 2)
                break
        now = time.monotonic()
        dt = max(1e-3, now - last_t)
        last_t = now
        if target_c:
            err_x = target_c[0] - cx
            err_y = target_c[1] - cy
            ycmd = pid_yaw.step(err_x, dt)
            pcmd = pid_pitch.step(err_y, dt)
            try:
                ep_robot.gimbal.drive_speed(pitch_speed=-pcmd, yaw_speed=ycmd)
            except Exception:
                pass
            if abs(err_x) < 12 and abs(err_y) < 10:
                try:
                    ep_robot.blaster.fire(times=3)
                    time.sleep(0.3)
                except Exception:
                    pass
                return True
        else:
            time.sleep(0.03)
    try:
        ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    except Exception:
        pass
    return False


# -------------------- High-level wrapper --------------------

def ts_two_stage_shoot(ep_robot, desired_targets, pos, ori, colors_map, color_ranges,
                       h_fov_deg=96.0, v_fov_deg=54.0, stage2_timeout=6.0):
    """Run scan→stage1→stage2 for each desired (color,shape).
    Returns { 'entry': (pos,ori,cs_list), 'fired': [(color,shape,ok), ...] }
    """
    entry, centers = ts_scan_entry(ep_robot, colors_map, color_ranges, pos, ori)
    results = []
    for (c, s) in desired_targets:
        s_norm = _to_lab1_shape(s)
        approx = centers.get((c, s_norm))
        if approx is None:
            results.append((c, s, False))
            continue
        ts_stage1_move(ep_robot, approx, h_fov_deg=h_fov_deg, v_fov_deg=v_fov_deg)
        ok = ts_stage2_pid_fire(ep_robot, c, s, color_ranges, colors_map, timeout_s=stage2_timeout)
        results.append((c, s, ok))
    return {"entry": entry, "fired": results}