"""RoboMaster Maze Explorer - Ultra Compact"""
import time, threading, math, json, sys, statistics
from datetime import datetime
from collections import deque
from robomaster import robot
from compact_vision import quick_target_check, scan_for_targets

# Config
START_POSITION = (0, 0)
GRID_SIZE_M = 0.635
WALL_THRESHOLD_MM = 450
BORDER = (3, 3)
GIMBAL_TURN_SPEED = 360

# Target Storage
class TargetStorage:
    def __init__(self):
        self.detected_targets = []
        self.storage_file = "detected_targets.json"
    
    def add_target(self, position, note="object", detection_method="vision", confidence=1.0):
        for existing in self.detected_targets:
            if tuple(existing["position"]) == tuple(position):
                existing["confidence"] = max(existing["confidence"], confidence)
                return
        self.detected_targets.append({
            "position": list(position), "note": note, "detection_method": detection_method,
            "confidence": confidence, "timestamp": datetime.now().isoformat(), "shot": False
        })
        self.save_to_file()
    
    def get_unshot_targets(self): return [t for t in self.detected_targets if not t["shot"]]
    
    def mark_target_shot(self, position):
        for target in self.detected_targets:
            if tuple(target["position"]) == tuple(position):
                target["shot"] = True
                target["shot_timestamp"] = datetime.now().isoformat()
                self.save_to_file()
                return True
    
    def save_to_file(self):
        try:
            with open(self.storage_file, "w") as f:
                json.dump(self.detected_targets, f, indent=2)
        except: pass
    
    def load_from_file(self):
        try:
            with open(self.storage_file, "r") as f:
                self.detected_targets = json.load(f)
            return True
        except: return False

target_storage = TargetStorage()

def calculate_target_grid_position(robot_position, robot_orientation, detection_data):
    if not detection_data: return robot_position
    try:
        distance_m = detection_data.get('distance_m', 0.635)
        angle_deg = detection_data.get('angle_deg', 0)
        absolute_angle = (robot_orientation * 90 + angle_deg) % 360
        dx_m = distance_m * math.sin(math.radians(absolute_angle))
        dy_m = distance_m * math.cos(math.radians(absolute_angle))
        robot_x, robot_y = robot_position
        return (robot_x + round(dx_m / GRID_SIZE_M), robot_y + round(dy_m / GRID_SIZE_M))
    except: return robot_position

def x(position, note="object", detection_method="vision", confidence=1.0):
    target_storage.add_target(position, note, detection_method, confidence)

def shoot_target(position, robot_instance=None, detection_data=None):
    print(f"[SHOOTING] {position}")
    if robot_instance:
        try:
            robot_instance.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            robot_instance.ep_led.set_led(r=255, g=0, b=0, effect="flash")
            time.sleep(0.5)
            robot_instance.ep_led.set_led(r=0, g=255, b=0, effect="on")
            time.sleep(1)
            robot_instance.ep_led.set_led(r=0, g=0, b=255, effect="on")
        except: pass
    target_storage.mark_target_shot(position)

# Data Handlers
class TofDataHandler:
    def __init__(self, window_size=3):
        self._lock = threading.Lock()
        self.raw_distance = 0.0
        self._window = deque(maxlen=window_size)
    
    def update(self, sub_info):
        d = float(sub_info[0]) if sub_info else 0.0
        with self._lock:
            self.raw_distance = d
            self._window.append(d)
    
    def get_distance(self):
        with self._lock: 
            return statistics.median(self._window) if self._window else self.raw_distance

class VisionDataHandler:
    def __init__(self): pass
    def update(self, vision_info): pass

class GimbalDataHandler:
    def __init__(self):
        self._lock = threading.Lock()
        self.yaw, self.pitch = 0.0, 0.0
    def update(self, angle_info):
        with self._lock: 
            self.yaw, self.pitch = float(angle_info[0]), float(angle_info[1])

class PoseDataHandler:
    def __init__(self):
        self.pose = [0.0] * 6
        self._lock = threading.Lock()
    def update_position(self, pos_info):
        with self._lock: 
            self.pose[0:3] = pos_info[0:3]
    def update_attitude(self, att_info):
        with self._lock: 
            self.pose[3:6] = att_info[0:3]
    def get_pose(self):
        with self._lock: 
            return self.pose[:]
    def set_xy(self, x_m, y_m):
        with self._lock: 
            self.pose[0], self.pose[1] = x_m, y_m
    def set_yaw(self, yaw_deg):
        with self._lock: 
            self.pose[3] = yaw_deg

# Map and Navigation
class RobotMap:
    def __init__(self):
        self.graph = {}
        self.explored = set()
        self.blocked = set()
    
    def add_connection(self, pos1, pos2):
        if pos1 not in self.graph: self.graph[pos1] = set()
        if pos2 not in self.graph: self.graph[pos2] = set()
        self.graph[pos1].add(pos2)
        self.graph[pos2].add(pos1)
    
    def add_blocked(self, pos1, pos2):
        self.blocked.add(tuple(sorted([pos1, pos2])))
    
    def mark_explored(self, position):
        self.explored.add(position)
    
    def get_unexplored_neighbors(self, position):
        if position not in self.graph: return []
        return [n for n in self.graph.get(position, []) if n not in self.explored]
    
    def get_path(self, start, goal):
        if start == goal: return [start]
        queue = [(start, [start])]
        visited = {start}
        while queue:
            current, path = queue.pop(0)
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == goal: return new_path
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)
        return None

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-1.0, 1.0)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._integral = self._previous_error = 0.0
        self._last_time = time.time()
    
    def update(self, current_value):
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0: return 0.0
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._previous_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        if self.output_limits:
            output = max(self.output_limits[0], min(self.output_limits[1], output))
        self._previous_error, self._last_time = error, current_time
        return output

# Main Explorer Class
class MazeExplorer:
    def __init__(self, ep_robot, tof_handler, vision_handler, gimbal_handler, pose_handler):
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_led = ep_robot.led
        self.ep_gimbal = ep_robot.gimbal
        self.ep_adaptor = ep_robot.sensor_adaptor
        self.ep_camera = ep_robot.camera
        self.targets = []
        self.is_lap1 = True
        self.lap2_shot_targets = set()
        self.target_storage = target_storage
        self.tof_handler = tof_handler
        self.pose_handler = pose_handler
        self.current_position = START_POSITION
        self.current_orientation = 0
        self.internal_map = RobotMap()
        self.visited_path = [self.current_position]
        self.visited_for_check = []
        self.ep_led.set_led(r=0, g=0, b=255)
        self.pose_handler.set_xy(START_POSITION[0] * GRID_SIZE_M, START_POSITION[1] * GRID_SIZE_M)
        self.pose_handler.set_yaw(0.0)

    def _read_ir_sensors(self):
        wall_ir_left = not self.ep_adaptor.get_io(id=1, port=1)
        wall_ir_right = not self.ep_adaptor.get_io(id=2, port=2)
        return wall_ir_left, wall_ir_right

    def _get_neighbor_position(self, scan_direction):
        x, y = self.current_position
        dirs = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        return dirs[scan_direction]

    def _detect_wall_in_direction(self, scan_direction, wall_left, wall_right):
        relative_direction = (scan_direction - self.current_orientation + 4) % 4
        if relative_direction == 0:
            distance_mm = self.tof_handler.get_distance()
            return distance_mm < WALL_THRESHOLD_MM, distance_mm
        elif relative_direction == 1:
            return wall_right, 100 if wall_right else 1000
        elif relative_direction == 3:
            return wall_left, 100 if wall_left else 1000
        else:
            return False, 1000.0

    def scan_surroundings_with_gimbal(self, previous_position=None):
        print(f"Scanning at {self.current_position}")
        self.internal_map.mark_explored(self.current_position)
        
        direction_to_skip = -1
        if previous_position:
            self.internal_map.add_connection(self.current_position, previous_position)
            direction_to_skip = (self.current_orientation + 2) % 4

        wall_left, wall_right = self._read_ir_sensors()
        
        for scan_direction in range(4):
            if scan_direction == direction_to_skip: continue
            neighbor_pos = self._get_neighbor_position(scan_direction)
            has_wall, _ = self._detect_wall_in_direction(scan_direction, wall_left, wall_right)

            if not has_wall:
                if (0 <= neighbor_pos[0] < BORDER[0] and 0 <= neighbor_pos[1] < BORDER[1]):
                    self.internal_map.add_connection(self.current_position, neighbor_pos)
            else:
                self.internal_map.add_blocked(self.current_position, neighbor_pos)

        # Vision Detection
        if self.is_lap1:
            try:
                target_found = quick_target_check(self.ep_camera, color='red', shape='circle')
                if target_found:
                    detections = scan_for_targets(self.ep_camera, color='red', shape='circle', duration=1.5)
                    if detections:
                        best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                        target_grid_pos = calculate_target_grid_position(
                            self.current_position, self.current_orientation, best_detection)
                        if target_grid_pos not in self.targets:
                            self.targets.append(target_grid_pos)
                            x(target_grid_pos, note="vision detected", detection_method="vision", 
                              confidence=best_detection.get('confidence', 1.0))
            except: pass

            if self.current_position not in self.targets:
                self.targets.append(self.current_position)
                x(self.current_position, note="fallback", detection_method="simulation", confidence=0.3)

    def decide_next_path(self):
        unexplored = self.internal_map.get_unexplored_neighbors(self.current_position)
        if unexplored: return [self.current_position, unexplored[0]]
        for pos in reversed(self.visited_path):
            nbs = self.internal_map.get_unexplored_neighbors(pos)
            if nbs: return self.internal_map.get_path(self.current_position, pos)
        return None

    def move_forward_pid(self, distance_m, speed_limit=0.2):
        pid = PIDController(Kp=2, Ki=0.02, Kd=0.1, setpoint=distance_m, output_limits=(-speed_limit, speed_limit))
        start_x, start_y, _, _, _, _ = self.pose_handler.get_pose()
        while True:
            curr_x, curr_y, _, _, _, _ = self.pose_handler.get_pose()
            dist_traveled = math.hypot(curr_x - start_x, curr_y - start_y)
            if abs(distance_m - dist_traveled) < 0.01: break
            if self.tof_handler.get_distance() <= 280: break
            vx_speed = pid.update(dist_traveled)
            self.ep_chassis.drive_speed(x=vx_speed, y=0, z=0, timeout=0.1)
            time.sleep(0.01)
        self.ep_chassis.drive_speed(0, 0, 0)

    def turn_pid(self, target_angle, speed_limit=90):
        pid = PIDController(Kp=2.5, Ki=0, Kd=0.15, setpoint=0, output_limits=(-speed_limit, speed_limit))
        while True:
            _, _, _, current_yaw, _, _ = self.pose_handler.get_pose()
            error = target_angle - current_yaw
            if error > 180: error -= 360
            if error < -180: error += 360
            if abs(error) < 1.5: break
            vz_speed = pid.update(-error)
            self.ep_chassis.drive_speed(x=0, y=0, z=vz_speed, timeout=0.1)
            time.sleep(0.01)
        self.ep_chassis.drive_speed(0, 0, 0)

    def execute_path(self, path):
        if not path or len(path) < 2: return
        for i in range(len(path) - 1):
            if self.current_position in self.visited_for_check: continue
            self.visited_for_check.append(self.current_position)
            start_node, end_node = path[i], path[i+1]
            dx, dy = end_node[0] - start_node[0], end_node[1] - start_node[1]
            
            target_angle = 0
            if dx == 1: target_angle = 90
            elif dx == -1: target_angle = -90
            elif dy == -1: target_angle = 180
            
            self.turn_pid(target_angle)
            
            if dx == 1: self.current_orientation = 1
            elif dx == -1: self.current_orientation = 3
            elif dy == 1: self.current_orientation = 0
            elif dy == -1: self.current_orientation = 2
            
            time.sleep(0.2)
            self.move_forward_pid(GRID_SIZE_M)
            self.current_position = end_node
            self.visited_path.append(self.current_position)
            self.pose_handler.set_xy(end_node[0] * GRID_SIZE_M, end_node[1] * GRID_SIZE_M)
            self.pose_handler.set_yaw(target_angle)
            time.sleep(0.2)

    def save_state(self, filename="maze_state.json"):
        try:
            data = {
                "graph": {f"{k[0]},{k[1]}":[f"{n[0]},{n[1]}" for n in v] for k,v in self.internal_map.graph.items()},
                "blocked": [[f"{a[0]},{a[1]}", f"{b[0]},{b[1]}"] for (a,b) in self.internal_map.blocked],
                "explored": [f"{p[0]},{p[1]}" for p in self.internal_map.explored],
                "targets": [list(t) for t in self.targets],
                "current_position": list(self.current_position),
                "current_orientation": self.current_orientation,
                "visited_path": [list(p) for p in self.visited_path],
            }
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            self.target_storage.save_to_file()
        except Exception as e:
            print(f"Save error: {e}")

    def load_state(self, filename="maze_state.json"):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            self.internal_map.graph.clear()
            for k, nb_list in data.get("graph", {}).items():
                x,y = map(int, k.split(","))
                self.internal_map.graph[(x,y)] = set()
                for nb in nb_list:
                    nx,ny = map(int, nb.split(","))
                    self.internal_map.graph[(x,y)].add((nx,ny))
            self.internal_map.blocked.clear()
            for a,b in data.get("blocked", []):
                ax,ay = map(int, a.split(","))
                bx,by = map(int, b.split(","))
                self.internal_map.blocked.add(tuple(sorted([(ax,ay),(bx,by)])))
            self.internal_map.explored = set()
            for p in data.get("explored", []):
                px,py = map(int, p.split(","))
                self.internal_map.explored.add((px,py))
            self.targets = [tuple(t) for t in data.get("targets", [])]
            self.current_position = tuple(data.get("current_position", START_POSITION))
            self.current_orientation = int(data.get("current_orientation", 0))
            self.visited_path = [tuple(p) for p in data.get("visited_path", [self.current_position])]
            self.target_storage.load_from_file()
            self.pose_handler.set_xy(self.current_position[0]*GRID_SIZE_M, self.current_position[1]*GRID_SIZE_M)
            yaw = [0, 90, 180, -90][self.current_orientation]
            self.pose_handler.set_yaw(yaw)
            return True
        except: return False

    def run_lap1(self):
        self.is_lap1 = True
        print("===== LAP 1 : Explore & Shoot on sight =====")
        while True:
            if self.current_position not in self.internal_map.explored:
                prev = self.visited_path[-2] if len(self.visited_path) > 1 else None
                self.scan_surroundings_with_gimbal(previous_position=prev)
            path_to_execute = self.decide_next_path()
            if not path_to_execute: break
            self.execute_path(path_to_execute)
        print(f"[LAP1] Targets found: {len(self.targets)}")
        self.save_state()

    def run_lap2(self):
        self.is_lap1 = False
        print("===== LAP 2 : Nearest-Target Navigation =====")
        if not self.load_state(): return
        stored_targets = self.target_storage.get_unshot_targets()
        targets_to_visit = [tuple(t["position"]) for t in stored_targets] if stored_targets else list(self.targets)
        self.lap2_shot_targets = set()
        
        while True:
            candidates = [t for t in targets_to_visit if t not in self.lap2_shot_targets]
            if not candidates: break
            
            best_tgt, best_path = None, None
            for tgt in candidates:
                p = self.internal_map.get_path(self.current_position, tgt)
                if not p: continue
                if (best_path is None) or (len(p) < len(best_path)):
                    best_tgt, best_path = tgt, p
            
            if best_path is None: break
            
            self.execute_path(best_path)
            shoot_target(best_tgt, robot_instance=self)
            self.lap2_shot_targets.add(best_tgt)
            self.target_storage.mark_target_shot(best_tgt)
            targets_to_visit = [t for t in targets_to_visit if t != best_tgt]
        
        print(f"[LAP2] Shot: {len(self.lap2_shot_targets)} targets")

    def print_target_summary(self):
        targets = self.target_storage.detected_targets
        if not targets: return
        print(f"=== TARGET SUMMARY: {len(targets)} เป้าหมาย ===")
        shot_count = sum(1 for t in targets if t["shot"])
        print(f"สถิติ: ยิงได้ {shot_count}/{len(targets)} เป้า ({shot_count/len(targets)*100:.1f}%)")

def main():
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        
        tof_handler = TofDataHandler()
        vision_handler = VisionDataHandler()
        gimbal_handler = GimbalDataHandler()
        pose_handler = PoseDataHandler()
        
        ep_robot.sensor.sub_distance(freq=10, callback=tof_handler.update)
        ep_robot.chassis.sub_position(freq=10, callback=pose_handler.update_position)
        ep_robot.chassis.sub_attitude(freq=10, callback=pose_handler.update_attitude)
        ep_robot.gimbal.sub_angle(freq=20, callback=gimbal_handler.update)
        
        explorer = MazeExplorer(ep_robot, tof_handler, vision_handler, gimbal_handler, pose_handler)
        time.sleep(2)
        
        mode = sys.argv[1] if len(sys.argv) > 1 else "lap1"
        
        if mode == "lap1":
            explorer.run_lap1()
        elif mode == "lap2":
            explorer.run_lap2()
        elif mode == "both":
            explorer.run_lap1()
            input("Press Enter for lap2...")
            explorer.run_lap2()
        
        explorer.print_target_summary()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            if 'ep_robot' in locals():
                ep_robot.sensor.unsub_distance()
                ep_robot.chassis.unsub_position()
                ep_robot.chassis.unsub_attitude()
                ep_robot.gimbal.unsub_angle()
                ep_robot.close()
        except: pass

if __name__ == '__main__': main()