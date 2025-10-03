# lab2_rules_bfs.py
"""
รอบ 2 (no fixed order): ยิงทุกเป้าที่ "ยิงได้" ตามกติกา ด้วย BFS (hop-count)
- โหลดกราฟ/objects จาก JSON รอบ 1
- ใช้ ShootRules เพื่อกำหนดว่า object ไหนยิงได้/ไม่ได้
- เลือกเป้าถัดไปแบบ greedy nearest-next จากสิ่งที่ยิงได้
- เดินด้วยเมธอดจาก test (execute_path / turn_pid / move_forward_pid)
- ปรับระยะกำแพงด้วย periodic_wall_clearance_adjust จาก test ตอนเริ่ม, ถึงเป้า และก่อนยิง
"""

# =============================================================================
# CONFIGURATION (แก้ได้ตรงนี้)
# =============================================================================
JSON_PATH = "maze_exploration_20251003_120000.json"  # <<< ชี้ไปไฟล์ JSON รอบ 1 ของคุณ

# เริ่มจากตำแหน่งไหนในรอบ 2:
# - "start" = metadata.start (ปกติ (0,0))
# - "final" = final_position ของรอบ 1
START_MODE = "final"   # "start" | "final"

# ระยะห่างกำแพงเป้าหมาย (เมตร) เพื่อจัดท่าก่อนยิง
CLEARANCE_M = 0.18

# กติกา: อนุญาต/แบนตามสี/รูป/ชนิด (ปล่อย set() คือไม่บังคับ)
ALLOWED_COLORS = {"red", "blue"}       # set() ถ้าไม่บังคับสี
ALLOWED_SHAPES = {"circle", "square"}  # set() ถ้าไม่บังคับรูป
ALLOWED_TYPES  = set()
BANNED_COLORS  = set()
BANNED_SHAPES  = set()
BANNED_TYPES   = {"blue_triangle"}
RESPECT_OBJ_FLAG = True  # ถ้า object ใน JSON มี can_shoot=False จะไม่ยิง

# =============================================================================
# IMPORTS
# =============================================================================
import json
import time
from collections import deque
from typing import Dict, List, Set, Tuple, Optional

import robomaster
from robomaster import robot

# ---- ปรับชื่อโมดูลให้ตรงกับไฟล์รอบ 1 ----
import test  # ต้องมี: MazeExplorer, TofDataHandler, GimbalDataHandler, PoseDataHandler, GRID_SIZE_M, START

Grid = Tuple[int, int]
# mapping เดียวกับ test: 0=N,1=E,2=S,3=W
HEADING_TO_DEG = {0: 0, 1: 90, 2: 180, 3: -90}

# =============================================================================
# Graph + BFS
# =============================================================================
def build_graph_from_edges(edges: List[List[int]]) -> Dict[Grid, Set[Grid]]:
    g: Dict[Grid, Set[Grid]] = {}
    for x1, y1, x2, y2 in edges:
        a, b = (x1, y1), (x2, y2)
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    return g

def bfs_shortest_path(graph: Dict[Grid, Set[Grid]], start: Grid, goal: Grid) -> Optional[List[Grid]]:
    if start == goal:
        return [start]
    q = deque([start])
    parent: Dict[Grid, Optional[Grid]] = {start: None}
    while q:
        u = q.popleft()
        for v in graph.get(u, []):
            if v not in parent:
                parent[v] = u
                if v == goal:
                    path = [v]
                    cur = v
                    while parent[cur] is not None:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                q.append(v)
    return None

# =============================================================================
# Object metadata helpers
# =============================================================================
def normalize_obj_meta(obj: dict) -> dict:
    color = (obj.get("color") or "").lower() or None
    shape = (obj.get("shape") or "").lower() or None
    typ   = obj.get("type")
    if (color is None or shape is None) and isinstance(typ, str):
        t = typ.lower()
        for sep in (":", "_", "-", "|", "/"):
            if sep in t:
                parts = t.split(sep)
                if len(parts) == 2:
                    color = color or parts[0]
                    shape = shape or parts[1]
                    break
    return {"color": color, "shape": shape, "type": typ}

# =============================================================================
# Shooting rules (allowed / banned)
# =============================================================================
class ShootRules:
    """
    - allowed_* แบบ whitelist (ถ้ามีค่า ต้องผ่าน)
    - banned_* แบบ blacklist (ถ้าติดรายการ จะโดนตัดสิทธิ์)
    - respect_obj_flag: ถ้าจริงและ object.can_shoot == False จะตัดสิทธิ์
    """
    def __init__(
        self,
        allowed_colors: Set[str] = frozenset(),
        allowed_shapes: Set[str] = frozenset(),
        allowed_types:  Set[str] = frozenset(),
        banned_colors:  Set[str] = frozenset(),
        banned_shapes:  Set[str] = frozenset(),
        banned_types:   Set[str] = frozenset(),
        respect_obj_flag: bool = True,
    ):
        self.allowed_colors = {c.lower() for c in allowed_colors}
        self.allowed_shapes = {s.lower() for s in allowed_shapes}
        self.allowed_types  = {t.lower() for t in allowed_types}
        self.banned_colors  = {c.lower() for c in banned_colors}
        self.banned_shapes  = {s.lower() for s in banned_shapes}
        self.banned_types   = {t.lower() for t in banned_types}
        self.respect_obj_flag = respect_obj_flag

    def shootable(self, obj: dict) -> bool:
        if self.respect_obj_flag and obj.get("can_shoot") is False:
            return False
        meta = normalize_obj_meta(obj)
        col, shp, typ = meta["color"], meta["shape"], (meta["type"].lower() if isinstance(meta["type"], str) else None)
        # blacklist
        if col and col in self.banned_colors: return False
        if shp and shp in self.banned_shapes: return False
        if typ and typ in self.banned_types:  return False
        # whitelist (ถ้ากำหนดต้องผ่าน)
        if self.allowed_colors and (col not in self.allowed_colors): return False
        if self.allowed_shapes and (shp not in self.allowed_shapes): return False
        if self.allowed_types  and (typ not in self.allowed_types):  return False
        return True

# =============================================================================
# Approach cell & selection
# =============================================================================
def approach_cells_for_object(obj: dict) -> Tuple[Set[Grid], Optional[int]]:
    """
    คืน (ชุด approach cells, desired_heading)
    - ถ้า wall == "N": ไปยืนที่ (x, y+1) แล้ว "หัน South(0)"
    - ถ้า wall == "E": ไปยืนที่ (x+1, y) แล้ว "หัน West(1)"
    - ถ้า wall == "S": ไปยืนที่ (x, y-1) แล้ว "หัน North(2)"
    - ถ้า wall == "W": ไปยืนที่ (x-1, y) แล้ว "หัน East(3)"
    - ถ้า wall ไม่มี: ยืนใน cell เดียวกัน ไม่บังคับ heading
    """
    x, y = obj["grid"]
    w = obj.get("wall")
    if w == "N":
        return {(x, y + 1)}, 0
    if w == "E":
        return {(x + 1, y)}, 1
    if w == "S":
        return {(x, y - 1)}, 2
    if w == "W":
        return {(x - 1, y)}, 3
    return {(x, y)}, None

def pick_next_target_greedy(graph: Dict[Grid, Set[Grid]],
                            cur: Grid,
                            remaining: List[dict],
                            rules: ShootRules) -> Tuple[Optional[List[Grid]], Optional[dict], Optional[int]]:
    best = (None, None, None, 10**9)
    for obj in remaining:
        if not rules.shootable(obj):
            continue
        ac_set, head = approach_cells_for_object(obj)
        for ac in ac_set:
            path = bfs_shortest_path(graph, cur, ac)
            if path:
                cost = len(path) - 1
                if cost < best[3]:
                    best = (path, obj, head, cost)
    return best[0], best[1], best[2]

# =============================================================================
# Action: Fire (ปรับตาม SDK จริง)
# =============================================================================
def try_fire(ep_robot: robot.Robot):
    try:
        # ตัวอย่าง: ep_robot.blaster.fire(times=1)
        print("[ACTION] Fire!")
    except Exception as e:
        print(f"[WARN] Fire API failed: {e}. Fallback to LED flash.")
        try:
            ep_robot.led.set_led(r=255, g=0, b=0, effect="flash")
            time.sleep(0.4)
            ep_robot.led.set_led(r=0, g=255, b=0, effect="on")
        except:
            pass

# =============================================================================
# Main (no fixed order)
# =============================================================================
def run_round2_no_order(json_path: str, rules: ShootRules):
    # 1) โหลดข้อมูล
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph_edges = data["graph_edges"]
    objects_all = data.get("objects", [])
    start_xy = tuple(data["metadata"].get("start", test.START))
    final_xy = tuple(data.get("final_position", start_xy))
    origin = final_xy if START_MODE.lower() == "final" else start_xy

    graph = build_graph_from_edges(graph_edges)

    # 2) เชื่อมต่อหุ่นและ subscriptions
    print("Connecting to RoboMaster robot...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    print("Robot connected.")

    tof_handler = test.TofDataHandler()
    gimbal_handler = test.GimbalDataHandler()
    pose_handler = test.PoseDataHandler()

    ep_robot.sensor.sub_distance(freq=10, callback=tof_handler.update)
    ep_robot.chassis.sub_position(freq=10, callback=pose_handler.update_position)
    ep_robot.chassis.sub_attitude(freq=10, callback=pose_handler.update_attitude)
    ep_robot.gimbal.sub_angle(freq=20, callback=gimbal_handler.update)

    try:
        # ใช้คลาสเดินจาก lap1
        explorer = test.MazeExplorer(ep_robot, tof_handler, gimbal_handler, pose_handler)

        # ตั้งสถานะเริ่ม
        cur = (int(origin[0]), int(origin[1]))
        explorer.current_position = cur
        explorer.visited_path = [cur]
        explorer.pose_handler.set_xy(cur[0] * test.GRID_SIZE_M, cur[1] * test.GRID_SIZE_M)
        explorer.pose_handler.set_yaw(0.0)

        # จัดท่าเริ่มงาน (กันไปชิดกำแพงเกิน)
        explorer.periodic_wall_clearance_adjust(target_clearance_m=CLEARANCE_M)

        remaining = [dict(o) for o in objects_all]
        shot_any = False

        while True:
            path, obj, heading = pick_next_target_greedy(graph, cur, remaining, rules)
            if not path or not obj:
                break

            print(f"[INFO] next target: {obj}")
            print(f"[INFO] path len={len(path)-1}: {path}")

            # เดินตาม path (ใช้ execute_path ของ lap1)
            explorer.execute_path(path)

            # ถึง approach cell แล้ว จัด clearance ให้อยู่ตำแหน่งยิงสบาย
            explorer.periodic_wall_clearance_adjust(target_clearance_m=CLEARANCE_M)

            # หมุนเข้าหาเป้าตามกำแพงถ้าระบุไว้
            if heading is not None:
                explorer.turn_pid(HEADING_TO_DEG[heading])

            # เผื่อเซ็นเซอร์เพิ่งนิ่ง ปรับระยะอีกครั้งก่อนยิง
            explorer.periodic_wall_clearance_adjust(target_clearance_m=CLEARANCE_M)

            # ยิง
            try_fire(ep_robot)
            shot_any = True

            # อัปเดตสถานะ
            cur = path[-1]
            explorer.current_position = cur
            explorer.visited_path.append(cur)
            explorer.pose_handler.set_xy(cur[0] * test.GRID_SIZE_M, cur[1] * test.GRID_SIZE_M)

            # เอา object นี้ออกจากคิว
            idx = remaining.index(obj)
            remaining.pop(idx)

        if shot_any:
            print("\n[DONE] finished shooting all shootable/reachable targets.")
        else:
            print("\n[INFO] no shootable/reachable targets under current rules.")

    finally:
        # cleanup
        try:
            ep_robot.sensor.unsub_distance()
            ep_robot.gimbal.unsub_angle()
            ep_robot.chassis.unsub_position()
            ep_robot.chassis.unsub_attitude()
        except Exception as e:
            print(f"[WARN] unsub error: {e}")
        try:
            ep_robot.close()
        except Exception as e:
            print(f"[WARN] close error: {e}")

# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    rules = ShootRules(
        allowed_colors=ALLOWED_COLORS,
        allowed_shapes=ALLOWED_SHAPES,
        allowed_types=ALLOWED_TYPES,
        banned_colors=BANNED_COLORS,
        banned_shapes=BANNED_SHAPES,
        banned_types=BANNED_TYPES,
        respect_obj_flag=RESPECT_OBJ_FLAG,
    )
    run_round2_no_order(JSON_PATH, rules)
