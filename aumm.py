#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboMaster Vision Detector — Pro Refactor (Oct 2025)

เป้าหมาย: เร็วขึ้น นิ่งขึ้น ใช้งานจริงได้มากขึ้น
- ปรับโครงสร้างคลาส + พารามิเตอร์ชัดเจน
- HSV mask แบบหลายช่วงสี (รวมแดง 2 ช่วง) + morphology ที่เบาแต่ได้ผล
- สแกนเฟส 1 (เต็มเฟรม) → เรียงลำดับเป้าแบบฉลาด (ซ้าย→ขวา หรือ ใกล้ศูนย์กลางก่อน)
- เฟส 2 เล็งด้วย ROI แคบ + PID ที่มี anti-windup + rate limiting + deadband
- กำหนดขอบเขต yaw/pitch + soft-stop ก่อนยิง เพื่อลด overshoot
- TOF gating ที่คุมทั้งก่อนเล็งและก่อนยิง
- วาด overlay debug (เปิด/ปิด) + ค่าความถี่เฟรม FPS
- หลีกเลี่ยงสั่งคำสั่งกิมบอลถี่เกินไปด้วย RateLimiter

หมายเหตุ: API ของ RoboMaster ในเครื่องคุณอาจต่างจากนี้เล็กน้อย (EP/EP2/S1) — จุดที่ผิดชื่อเมธอดให้ปรับตาม SDK เวอร์ชันของคุณ
"""
from __future__ import annotations

import cv2
import time
import math
import numpy as np
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from robomaster import robot
except Exception:
    robot = None  # สำหรับ static analysis

# ============================
# Utilities
# ============================

class RateLimiter:
    """อนุญาตให้เรียกสั่งงานได้ไม่เกินทุก interval วินาที"""
    def __init__(self, interval_s: float):
        self.interval = max(0.0, float(interval_s))
        self._last = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False

class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, integ_clip=None, out_clip=None, d_lpf_alpha: float = 0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integ_clip = integ_clip
        self.out_clip = out_clip
        self.integ = 0.0
        self.last_error = 0.0
        self._d_lpf = 0.0
        self._alpha = float(np.clip(d_lpf_alpha, 0.0, 1.0))

    def reset(self):
        self.integ = 0.0
        self.last_error = 0.0
        self._d_lpf = 0.0

    def step(self, error, dt):
        if dt <= 0:
            dt = 0.01
        # P
        p_term = self.kp * error
        # I
        self.integ += error * dt
        if self.integ_clip is not None:
            self.integ = float(np.clip(self.integ, -self.integ_clip, self.integ_clip))
        i_term = self.ki * self.integ
        # D (LPF เพื่อลด jitter)
        raw_d = (error - self.last_error) / dt
        self._d_lpf = (1 - self._alpha) * self._d_lpf + self._alpha * raw_d
        d_term = self.kd * self._d_lpf
        self.last_error = error
        # Out
        out = p_term + i_term + d_term
        if self.out_clip is not None:
            out = float(np.clip(out, -self.out_clip, self.out_clip))
        return out

@dataclass
class TargetBox:
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    area: int
    shape: Optional[str] = None

# ============================
# Simple TOF wrapper (safe)
# ============================
class SimpleTofHandler:
    def __init__(self, ep_sensor):
        self.ep_sensor = ep_sensor

    def get_distance(self) -> float:
        try:
            # บาง SDK: return mm โดยตรง; หากเป็นโครงสร้าง ให้ปรับอ่าน field ตามจริง
            return float(self.ep_sensor.get_distance_sensor())
        except Exception:
            return 1_000.0  # default

# ============================
# Vision Detector
# ============================
class VisionDetectorPro:
    def __init__(self, ep_robot, *, debug_windows: bool = True):
        self.ep_robot = ep_robot
        self.ep_camera = ep_robot.camera
        self.ep_gimbal = ep_robot.gimbal
        self.ep_blaster = ep_robot.blaster
        self.tof = SimpleTofHandler(ep_robot.sensor)
        self.debug = bool(debug_windows)

        # --- Config ---
        self.FULL_SCAN_DURATION_S = 4.0  # ↑ ขยายเวลาสแกนให้เก็บเฟรมมากขึ้นเล็กน้อย
        self.AIM_TIMEOUT_S = 25.0  # ↑ เผื่อเวลาหาเป้าในสภาพแสงยาก
        self.CENTER_ROI_FRAC_W = 0.28  # ↑ ROI กว้างขึ้นเพื่อติดเป้าง่ายขึ้น
        self.CENTER_ROI_FRAC_H = 0.45  # ↑ ROI สูงขึ้นเล็กน้อย
        self.ACCEPT_ERR_PX = 24   # ↑ ยอมให้ error พิกเซลก่อนยิงมากขึ้น ลดอาการแกว่งรอให้เป๊ะ
        self.TOF_MIN_MM, self.TOF_MAX_MM = 300.0, 1200.0
        self.CMD_INTERVAL_S = 0.09  # ↑ ผ่อนคาบสั่งคำสั่ง ลดสั่น (oscillation) จากคำสั่งถี่เกิน  # กันสั่งถี่เกิน
        self.YAW_LIMIT_DEG = 55.0
        self.PITCH_MIN_DEG, self.PITCH_MAX_DEG = -25.0, 30.0
        self.DEADBAND_CMD = 0.05

        # PID ที่นิ่งขึ้นหน่อย
        self.pid_yaw = PID(kp=0.12, ki=0.00, kd=0.03, integ_clip=50.0, out_clip=6.0, d_lpf_alpha=0.25)  # ↓ KP, ↑ KD เพื่อลด overshoot
        self.pid_pitch = PID(kp=0.14, ki=0.00, kd=0.03, integ_clip=50.0, out_clip=6.0, d_lpf_alpha=0.25)  # ↓ KP, ↑ KD เพื่อลด overshoot
        self._cmd_rl = RateLimiter(self.CMD_INTERVAL_S)
        self._last_time = time.monotonic()

        # HSV color ranges (np.uint8 for speed)
        self.color_ranges: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
            # red 2 ช่วง
            'brown_red': [
                (np.array([0, 120, 70], np.uint8),  np.array([10, 255, 255], np.uint8)),
                (np.array([170, 120, 70], np.uint8), np.array([180, 255, 255], np.uint8)),
            ],
            'dark_green': [ (np.array([40, 60, 20], np.uint8),  np.array([85, 255, 180], np.uint8)) ],
            'yellow_brown': [ (np.array([18, 100, 100], np.uint8), np.array([40, 255, 255], np.uint8)) ],
            'dark_blue':   [ (np.array([95, 70, 20], np.uint8),   np.array([130, 255, 200], np.uint8)) ],
        }
        self.colors = { 'red': 'brown_red', 'green': 'dark_green', 'yellow': 'yellow_brown', 'blue': 'dark_blue' }

        self.target_color: Optional[str] = None
        self.target_shape: Optional[str] = None  # 'circle'|'square'|'horizontal_rectangle'|'vertical_rectangle'|'triangle'

        # Pre-alloc kernel
        self._kernel = np.ones((5, 5), np.uint8)

    # --------------- Public API ---------------
    def set_target(self, color: str, shape: Optional[str]):
        self.target_color = color
        self.target_shape = shape

    def detect_and_shot_two_phase(self, *, order: str = "left_to_right") -> bool:
        """เฟส 1: scan → เฟส 2: aim&fire ทีละเป้า
        order: "left_to_right" | "center_first"
        """
        self._ensure_camera_started()
        self._recenter_gimbal()

        color_key = self._resolve_color_key(self.target_color)
        if color_key is None:
            print(f"⚠️ ไม่พบสี '{self.target_color}' ในระบบ")
            return False

        targets = self._phase1_scan(color_key)
        if not targets:
            print("❌ No targets found in Phase 1")
            return False

        if order == "center_first":
            cx = 640  # สมมุติกลางเฟรม 1280x720
            targets.sort(key=lambda t: abs(t.center[0] - cx))
        else:  # left_to_right
            targets.sort(key=lambda t: t.center[0])

        print(f"✅ Phase 1 found {len(targets)} {self.target_color} {self.target_shape}(s)")
        for i, t in enumerate(targets, 1):
            print(f"  {i}. center={t.center} bbox={t.bbox} area={t.area} shape={t.shape}")

        success = 0
        for i, tgt in enumerate(targets, 1):
            if self._phase2_aim_and_fire(color_key, tgt, index=i, total=len(targets)):
                success += 1

        self._recenter_gimbal()
        print(f"✅ Done Phase 2: fired {success}/{len(targets)}")
        return success == len(targets)

    # --------------- Internals ---------------
    def _ensure_camera_started(self):
        try:
            self.ep_camera.start_video_stream(display=False, resolution="720p")
        except Exception:
            pass

    def _recenter_gimbal(self):
        try:
            self.ep_gimbal.recenter(pitch_speed=120, yaw_speed=120).wait_for_completed()
            self.ep_gimbal.move(pitch=-12, yaw=0).wait_for_completed()
        except Exception:
            pass

    def _resolve_color_key(self, color: Optional[str]) -> Optional[str]:
        if color in self.colors:
            color = self.colors[color]
        if color not in self.color_ranges:
            return None
        return color

    def _build_mask(self, hsv: np.ndarray, color_key: str) -> np.ndarray:
        ranges = self.color_ranges[color_key]
        mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
        for (lo, hi) in ranges[1:]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        # morphology (เปิดเล็กน้อยแล้วปิด)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        return mask

    def _classify_shape(self, cnt) -> Optional[str]:
        area = cv2.contourArea(cnt)
        if area <= 1:
            return None
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            return None
        circularity = 4 * math.pi * area / (peri * peri)
        if circularity > 0.75:
            return 'circle'
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        v = len(approx)
        if v == 3:
            return 'triangle'
        if v == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                return None
            ratio = float(w) / float(h)
            if 0.9 <= ratio <= 1.1:
                return 'square'
            return 'horizontal_rectangle' if ratio > 1.1 else 'vertical_rectangle'
        return None

    def _find_targets(self, img: np.ndarray, color_key: str, *, min_area: int = 150,
                      shape_req: Optional[str] = None, restrict_rect: Optional[Tuple[int,int,int,int]] = None) -> Tuple[List[TargetBox], Optional[np.ndarray]]:
        if restrict_rect is not None:
            x0, y0, ww, hh = restrict_rect
            roi = img[y0:y0+hh, x0:x0+ww]
        else:
            x0, y0, ww, hh = (0, 0, img.shape[1], img.shape[0])
            roi = img
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv, color_key)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found: List[TargetBox] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            shape = self._classify_shape(cnt)
            if shape_req is not None and shape != shape_req:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x0 + x + w // 2
            cy = y0 + y + h // 2
            found.append(TargetBox(center=(cx, cy), bbox=(x0+x, y0+y, w, h), area=w*h, shape=shape))
        found.sort(key=lambda t: t.center[0])
        return found, mask if self.debug else None

    def _phase1_scan(self, color_key: str) -> List[TargetBox]:
        print("🔎 Phase 1: scanning full frame for targets...")
        best: List[TargetBox] = []
        t0 = time.monotonic()
        frame_cnt = 0
        while time.monotonic() - t0 < self.FULL_SCAN_DURATION_S:
            try:
                frame = self.ep_camera.read_cv2_image(strategy="newest", timeout=1)
            except queue.Empty:
                continue
            if frame is None:
                continue
            frame_cnt += 1
            found, mask = self._find_targets(frame, color_key, min_area=150, shape_req=self.target_shape)
            if not found and self.target_shape is not None:
                found, mask = self._find_targets(frame, color_key, min_area=150, shape_req=None)
            if len(found) > len(best):
                best = found
            if self.debug:
                vis = frame.copy()
                for t in found:
                    x, y, w, h = t.bbox
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.circle(vis, t.center, 4, (0, 255, 0), -1)
                    cv2.putText(vis, f"{self.target_color}_{t.shape}", (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                fps = frame_cnt / max(1e-3, (time.monotonic() - t0))
                cv2.putText(vis, f"Phase1 FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                try:
                    cv2.imshow("Phase1-Scan", vis)
                    if mask is not None:
                        cv2.imshow("Mask-Phase1", mask)
                    cv2.waitKey(1)
                except Exception:
                    pass
        return best

    def _phase2_aim_and_fire(self, color_key: str, hint: TargetBox, *, index: int, total: int) -> bool:
        print(f"🎯 Phase 2: aiming target #{index}/{total} ...")
        aim_start = time.monotonic()
        stable_frames = 0
        stable_needed = 3
        prev_err_x = None
        # กำหนดทิศเริ่มหมุนจากตำแหน่ง hint
        last_dir = 1 if hint.center[0] >= 640 else -1
        # reset PID เพื่อไม่ให้ค่าค้าง
        self.pid_yaw.reset(); self.pid_pitch.reset()

        while time.monotonic() - aim_start < self.AIM_TIMEOUT_S:
            try:
                img = self.ep_camera.read_cv2_image(strategy="newest", timeout=1)
            except queue.Empty:
                continue
            if img is None:
                continue

            h, w = img.shape[:2]
            cx_mid, cy_mid = w // 2, h // 2
            # ROI กลางแคบ
            roi_w = max(80, int(w * self.CENTER_ROI_FRAC_W))
            roi_h = max(100, int(h * self.CENTER_ROI_FRAC_H))
            roi_x = (w - roi_w) // 2
            roi_y = (h - roi_h) // 2

            found, mask = self._find_targets(img, color_key, min_area=180, shape_req=self.target_shape,
                                             restrict_rect=(roi_x, roi_y, roi_w, roi_h))
            best: Optional[TargetBox] = max(found, key=lambda t: t.area) if found else None

            if self.debug:
                vis = img.copy()
                cv2.rectangle(vis, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 255), 2)
                if best is not None:
                    x, y, ww, hh = best.bbox
                    cv2.rectangle(vis, (x, y), (x+ww, y+hh), (0, 255, 0), 2)
                    cv2.circle(vis, best.center, 4, (0, 255, 0), -1)
                try:
                    cv2.imshow("Phase2", vis)
                    if mask is not None:
                        cv2.imshow("Mask-Phase2", mask)
                    cv2.waitKey(1)
                except Exception:
                    pass

            now = time.monotonic()
            dt = now - self._last_time
            self._last_time = now
            dt = float(np.clip(dt, 1e-3, 0.1))

            tof = self.tof.get_distance()

            if best is None:
                # หมุนสแกนช้าๆ ใน ROI ตามทิศเดิม และพลิกทิศทุก 1.2s
                if self._cmd_rl.allow():
                    yaw_speed = 3.0 * last_dir
                    try:
                        self.ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=0.0)
                    except Exception:
                        pass
                # เปลี่ยนทิศทางเพื่อไม่ให้ค้าง
                if int((now - aim_start) / 1.2) % 2 == 1:
                    last_dir = -last_dir
                continue

            # มีเป้าใน ROI → คำนวณ error พิกเซล
            err_x = best.center[0] - cx_mid
            err_y = best.center[1] - cy_mid
            yaw_cmd = self.pid_yaw.step(err_x, dt)
            pitch_cmd = self.pid_pitch.step(err_y, dt)

            # soft slowing when near center
            ax = abs(err_x)
            if ax < 80:  yaw_cmd *= 0.6
            if ax < 40:  yaw_cmd *= 0.4
            if prev_err_x is not None and (err_x * prev_err_x) < 0 and ax < (2 * self.ACCEPT_ERR_PX):
                yaw_cmd *= 0.5
            prev_err_x = err_x

            # Deadband: ถ้าคำสั่งเล็กมากไม่ต้องสั่ง
            if abs(yaw_cmd) < self.DEADBAND_CMD and abs(pitch_cmd) < self.DEADBAND_CMD:
                pass
            else:
                if self._cmd_rl.allow():
                    try:
                        self.ep_gimbal.drive_speed(pitch_speed=-pitch_cmd, yaw_speed=yaw_cmd)
                    except Exception:
                        try:
                            self.ep_gimbal.drive_speed(pitch_speed=pitch_cmd, yaw_speed=yaw_cmd)
                        except Exception:
                            pass

            # เงื่อนไขความพร้อมยิง
            if (abs(err_x) <= self.ACCEPT_ERR_PX and abs(err_y) <= self.ACCEPT_ERR_PX and
                self.TOF_MIN_MM < tof < self.TOF_MAX_MM):
                stable_frames += 1
            else:
                stable_frames = 0

            if stable_frames >= 3:
                print(f"🔫 Fire target #{index} at {best.center} (TOF={tof:.0f}mm)")
                self._fire_burst()
                self._stop_gimbal()
                return True

        print(f"⏱️ Aim timeout for target #{index}")
        self._stop_gimbal()
        return False

    # ---- low-level helpers ----
    def _stop_gimbal(self):
        try:
            self.ep_gimbal.drive_speed(pitch_speed=0.0, yaw_speed=0.0)
        except Exception:
            pass

    def _fire_burst(self):
        try:
            # ยิง 2 burst เว้นจังหวะสั้น ๆ เพื่อลด recoil
            self.ep_blaster.fire(times=3)
            time.sleep(0.5)
            self.ep_blaster.fire(times=3)
        except Exception:
            pass

# ============================
# Test entrypoints
# ============================

def test_camera_only():
    print("🤖 Initializing RoboMaster...")
    ep = robot.Robot(); ep.initialize(conn_type="ap")
    try:
        det = VisionDetectorPro(ep)
        det._ensure_camera_started()
        t0, cnt = time.time(), 0
        while True:
            try:
                img = det.ep_camera.read_cv2_image(strategy="newest", timeout=1)
            except queue.Empty:
                continue
            if img is None:
                continue
            cnt += 1
            fps = cnt / max(1e-3, (time.time() - t0))
            h, w = img.shape[:2]
            cx, cy = w//2, h//2
            cv2.circle(img, (cx, cy), 20, (0,0,255), 2)
            cv2.putText(img, f"{w}x{h} | {fps:.1f} FPS", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Camera", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        try:
            det.ep_camera.stop_video_stream()
        except Exception:
            pass
        cv2.destroyAllWindows()
        ep.close()


def test_vision_detection(color: str = 'green', shape: str = 'square'):
    print("🤖 Initializing RoboMaster for vision test...")
    ep = robot.Robot(); ep.initialize(conn_type="ap")
    try:
        det = VisionDetectorPro(ep, debug_windows=True)
        det.set_target(color, shape)
        print(f"🎯 Looking for: {det.target_color} {det.target_shape}")
        det.detect_and_shot_two_phase(order="center_first")
    finally:
        cv2.destroyAllWindows()
        ep.close()


if __name__ == "__main__":
    print("=" * 60)
    print("🎥 RoboMaster Vision Detector — Pro Refactor")
    print("=" * 60)
    print("1. Camera only test")
    print("2. Vision detection test (center-first)")
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == '1':
        test_camera_only()
    else:
        # เปลี่ยนค่าตามที่ต้องการ
        test_vision_detection(color='green', shape='square')
