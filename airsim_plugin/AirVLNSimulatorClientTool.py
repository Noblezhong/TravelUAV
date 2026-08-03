from collections import deque
import multiprocessing
import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import os,sys

import tqdm

cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

from utils.logger import logger


class BaseSensor:
    def __init__(self) -> None:
        pass

    def retrieve(self):
        raise NotImplementedError()

class State(BaseSensor):
    def __init__(self, client, drone_name=''):
        self.data = {'position': None, 'linear_velocity': None, 'linear_acceleration':None,
                     'orientation':None, 'angular_velocity':None, 'angular_acceleration':None}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name

    def retrieve(self):
        data = self.client.getMultirotorState(vehicle_name=self.drone_name)
        collision = {}
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        collision['has_collided'] = collision_info.has_collided
        collision['object_name'] = data.collision.object_name
        gps_location = [data.gps_location.latitude,data.gps_location.longitude,data.gps_location.altitude]
        timestamp = data.timestamp
        position = list(data.kinematics_estimated.position)
        linear_velocity = list(data.kinematics_estimated.linear_velocity)
        linear_acceleration = list(data.kinematics_estimated.linear_acceleration)
        orientation = list(data.kinematics_estimated.orientation)
        angular_velocity = list(data.kinematics_estimated.angular_velocity)
        angular_acceleration = list(data.kinematics_estimated.angular_acceleration)

        self.data.update({'collision': collision, 
                          'gps_location': gps_location,
                          'timestamp': timestamp, 
                          'position': position,
                          'linear_velocity': linear_velocity,
                          'linear_acceleration': linear_acceleration,
                          'orientation': orientation,
                          'angular_velocity': angular_velocity,
                          'angular_acceleration': angular_acceleration
                          })
        return self.data
        
        
class Imu(BaseSensor):
    def __init__(self, client, drone_name='', imu_name=''):
        self.data = {}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name
        self.imu_name = imu_name

    def retrieve(self):
        data = self.client.getImuData(imu_name=self.imu_name,vehicle_name=self.drone_name)
        time_stamp = data.time_stamp
        orientation = data.orientation
        angular_velocity = list(data.angular_velocity)
        linear_acceleration = list(data.linear_acceleration)
        q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                      [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                      [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)])).tolist()
        self.data.update({'time_stamp': time_stamp, 'rotation': rotation_matrix, 'orientation': list(data.orientation),
                          'linear_acceleration': linear_acceleration, 'angular_velocity': angular_velocity})
        return self.data


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        self.result = self.func(*self.args)
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info, fast_eval=False, fast_eval_speedup=5.0) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.fast_eval = bool(fast_eval)
        self.fast_eval_speedup = float(fast_eval_speedup)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]
        self.airsim_ports = []
        self.airsim_ip = '127.0.0.1'
        self._init_check()
        self.objects_name_cnt = [[0 for _ in list(item['open_scenes'])] for item in machines_info ]

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _confirmConnection(self) -> bool:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    confirmed = False
                    count = 0
                    while not confirmed and count < 60:
                        try:
                            self.airsim_clients[index_1][index_2].confirmConnection()
                            confirmed = True
                        except Exception as e:
                            time.sleep(1)
                            count += 1
                            pass

        return confirmed

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int=60) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients


        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f'开始打开场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            logger.info(f'gpus: {self.machines_info[index]}')
            reopen_args = (
                socket_client.address._host,
                list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])),
            )
            if self.fast_eval:
                result = socket_client.call('reopen_scenes', *reopen_args, self.fast_eval_speedup)
            else:
                result = socket_client.call('reopen_scenes', *reopen_args)
            if result[0] == False:
                logger.error(f'打开场景失败，机器: {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('打开场景失败')
            assert len(result[1]) == 2, '打开场景失败'
            print('waiting for airsim connection...')
            time.sleep(1)  # Server already waited for AirSim ready
            ip = result[1][0]
            if isinstance(ip, bytes):
                ip = ip.decode('utf-8')
            ports = result[1][1]
            self.airsim_ip = ip
            self.airsim_ports = ports
            assert str(ip) == str(socket_client.address._host), '打开场景失败'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), '打开场景失败'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.MultirotorClient(ip=ip, port=port, timeout_value=airsim_timeout)
                    print(port)

            logger.info(f'打开场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            return ports

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        
        if not (np.array(thread_results) == True).all():
            raise Exception('打开场景失败')

        after = time.time()
        diff = after - before
        logger.info(f"启动时间：{diff}")

        assert self._confirmConnection(), 'server connect failed'
        self._closeSocketConnection()
    
    def collect_DDP(self, data_dir, workers):
        def init_worker(index, lock):
            with lock:
                multiprocessing.current_process().client_port = self.machines_info[0]['SOCKET_PORT']
                multiprocessing.current_process().machine_ip = self.machines_info[0]['MACHINE_IP']
                multiprocessing.current_process().client = self.airsim_clients[0][index.value]
                multiprocessing.current_process().port = self.airsim_ports[index.value]
                index.value += 1
        index = multiprocessing.Value('i', 0)
        lock = multiprocessing.Lock()
        with multiprocessing.Pool(workers, initializer=init_worker, initargs=(index, lock)) as p:
            r = list(tqdm.tqdm(p.imap_unordered(collect, data_dir), total=len(data_dir)))

    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
                )

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients

            self._closeConnection()

            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'开始关闭所有场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'关闭所有场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)

    def move_path_by_waypoints(self, waypoints_list, start_states, target_idx=5):
        velocity = 1
        drivetrain = airsim.DrivetrainType.ForwardOnly
        yaw_mode=airsim.YawMode(is_rate=False)
        lookahead=3
        adaptive_lookahead=1
        def move_path(airsim_client: airsim.VehicleClient, waypoints, start_state):
            results = []
            state_sensor = State(airsim_client, )
            imu_sensor = Imu(airsim_client, imu_name='Imu')
            path = [airsim.Vector3r(*waypoint[0:3]) for waypoint in waypoints]
            target_limit = max(1, min(int(target_idx), len(path)))
            airsim_client.enableApiControl(True)
            airsim_client.armDisarm(True)
            airsim_client.simPause(False)
            airsim_client.simSetKinematics(start_state, ignore_collision=False)
            state_info = state_sensor.retrieve()
            action_wall_start = time.perf_counter()
            action_sim_start = int(state_info['timestamp'])
            airsim_client.moveOnPathAsync(path=path, 
                                velocity=velocity, 
                                drivetrain=drivetrain, 
                                yaw_mode=yaw_mode, 
                                lookahead=lookahead, 
                                adaptive_lookahead=adaptive_lookahead)
            current_idx = 0
            pos_queue = deque(maxlen=20)
            start_time = time.perf_counter()
            collision = False
            distance = 10000
            while True:
                time.sleep(0.005)
                wall_timeout = time.perf_counter() - start_time > 5
                if wall_timeout:
                    return None
                state_info = copy.deepcopy(state_sensor.retrieve())
                imu_info = copy.deepcopy(imu_sensor.retrieve())
                target = path[current_idx]
                position = np.array(state_info['position'])
                pos_queue.append(position)
                if len(pos_queue) == pos_queue.maxlen:
                    recent_loc = position
                    history_loc = pos_queue.popleft()
                    delta_distance = np.linalg.norm(history_loc -recent_loc)
                    if delta_distance < 0.1:
                        print('move on path api: stuck max len')
                        collision = True
                        break
                new_distance = np.linalg.norm(position - np.array([target.x_val, target.y_val, target.z_val]))
                if new_distance > distance:
                    results.append({'sensors': {'state': state_info, 'imu': imu_info}})
                    current_idx += 1
                    if current_idx == target_limit:
                        airsim_client.simPause(True)
                        break
                    else:
                        distance = 10000
                else:
                    distance = new_distance
            action_wall_ms = (time.perf_counter() - action_wall_start) * 1000.0
            action_sim_ms = max(0.0, (int(state_info['timestamp']) - action_sim_start) / 1e6)
            return {
                'states': results,
                'collision': collision,
                'wall_time_ms': action_wall_ms,
                'sim_time_ms': action_sim_ms,
            }
        
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(move_path, (self.airsim_clients[index_1][index_2], waypoints_list[index_1][index_2], start_states[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        result_poses_list = []
        error_flag = False
        for index_1, _ in enumerate(threads):
            result_poses_list.append([])
            for index_2, _ in enumerate(threads[index_1]):
                result =  threads[index_1][index_2].get_result()
                result_poses_list[index_1].append(
                    result
                )
                if result is None:
                    error_flag = True
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('move path by waypoints failed.')
            return None
        if error_flag:
            return None
        return result_poses_list

    def move_to_position(self, target_poses, start_states):
        """
        Move drone to target pose using moveToPositionAsync + rotateToYawAsync.
        For CMA discrete actions — avoids moveOnPathAsync "stuck" false positive
        on rotation-only actions (TURN_LEFT, TURN_RIGHT).
        target_poses: list of lists of [x, y, z, yaw_deg] per scene.
        Returns same format as move_path_by_waypoints: {states: [...], collision: bool}.
        """
        velocity = 2.0
        timeout_sec = 15.0

        def _move(airsim_client, target, start_state):
            state_sensor = State(airsim_client)
            imu_sensor = Imu(airsim_client, imu_name='Imu')

            target_x, target_y, target_z = float(target[0]), float(target[1]), float(target[2])
            target_yaw = float(target[3]) if len(target) >= 4 else None

            airsim_client.enableApiControl(True)
            airsim_client.armDisarm(True)
            airsim_client.simPause(False)

            teleport = os.environ.get('CMA_TELEPORT') == '1'
            collision = False

            if teleport:
                # Instant teleport — skip flight simulation for fast eval
                from scipy.spatial.transform import Rotation as R
                yaw_rad = np.deg2rad(target_yaw) if target_yaw is not None else 0.0
                r = R.from_euler('z', yaw_rad)
                q = r.as_quat()  # [x, y, z, w]
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(target_x, target_y, target_z),
                    orientation_val=airsim.Quaternionr(q[0], q[1], q[2], q[3]),
                )
                airsim_client.simSetKinematics(pose, ignore_collision=True)  # Teleport must ignore collision
                airsim_client.simContinueForFrames(1)
                # Collision not meaningful in teleport mode — drone is placed at exact target pose
            else:
                airsim_client.simSetKinematics(start_state, ignore_collision=False)

                current_state = state_sensor.retrieve()
                current_pos = np.array(current_state['position'])
                target_pos = np.array([target_x, target_y, target_z])
                distance = float(np.linalg.norm(target_pos - current_pos))

                collision = current_state.get('collision', {}).get('has_collided', False)

                if not collision:
                    if distance > 0.2:
                        # Position change: fly to target with desired yaw
                        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=target_yaw) \
                            if target_yaw is not None else airsim.YawMode()
                        try:
                            airsim_client.moveToPositionAsync(
                                target_x, target_y, target_z, velocity,
                                timeout_sec=timeout_sec,
                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                yaw_mode=yaw_mode,
                            ).join()
                        except Exception:
                            collision = True
                    elif target_yaw is not None:
                        # Rotation only (TURN action)
                        try:
                            airsim_client.rotateToYawAsync(target_yaw, timeout_sec=timeout_sec).join()
                        except Exception:
                            collision = True

            state_info = state_sensor.retrieve()
            imu_info = imu_sensor.retrieve()
            if not teleport and not collision:
                collision = state_info.get('collision', {}).get('has_collided', False)

            airsim_client.simPause(True)
            return {'states': [{'sensors': {'state': state_info, 'imu': imu_info}}], 'collision': collision}

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_move, (self.airsim_clients[index_1][index_2],
                                     target_poses[index_1][index_2],
                                     start_states[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        result_poses_list = []
        error_flag = False
        for index_1, _ in enumerate(threads):
            result_poses_list.append([])
            for index_2, _ in enumerate(threads[index_1]):
                result = threads[index_1][index_2].get_result()
                result_poses_list[index_1].append(result)
                if result is None:
                    error_flag = True
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('move to position failed.')
            return None
        if error_flag:
            return None
        return result_poses_list

    def move_path_by_velocity_waypoints(self, waypoints_list, start_states, target_idx=1):
        velocity = 1.0
        control_dt = 0.05
        arrival_radius = 0.2
        timeout_accept_radius = 0.5
        close_settle_timeout = 1.0
        close_progress_epsilon = 0.01
        target_timeout = 30.0
        drivetrain = airsim.DrivetrainType.ForwardOnly
        yaw_mode = airsim.YawMode(is_rate=False)

        def move_path(airsim_client: airsim.VehicleClient, waypoints, start_state):
            results = []
            state_sensor = State(airsim_client)
            imu_sensor = Imu(airsim_client, imu_name='Imu')
            targets = [np.asarray(waypoint[0:3], dtype=np.float64) for waypoint in waypoints]
            target_limit = max(1, min(int(target_idx), len(targets)))
            collision = False

            airsim_client.enableApiControl(True)
            airsim_client.armDisarm(True)
            airsim_client.simPause(False)
            airsim_client.simSetKinematics(start_state, ignore_collision=False)
            action_wall_start = time.perf_counter()
            action_sim_start = None
            state_info = None
            if self.fast_eval:
                state_info = copy.deepcopy(state_sensor.retrieve())
                action_sim_start = int(state_info['timestamp'])

            for target in targets[:target_limit]:
                target_start_time = time.perf_counter()
                target_sim_start = int(state_info['timestamp']) if self.fast_eval and state_info is not None else None
                close_start_time = None
                close_start_sim = None
                close_best_distance = None
                while True:
                    state_info = copy.deepcopy(state_sensor.retrieve())
                    imu_info = copy.deepcopy(imu_sensor.retrieve())
                    position = np.asarray(state_info['position'], dtype=np.float64)
                    delta = target - position
                    distance = float(np.linalg.norm(delta))

                    if state_info.get('collision', {}).get('has_collided', False):
                        collision = True
                        results.append({'sensors': {'state': state_info, 'imu': imu_info}})
                        break

                    if distance <= arrival_radius:
                        results.append({'sensors': {'state': state_info, 'imu': imu_info}})
                        break

                    if distance <= timeout_accept_radius:
                        if close_best_distance is None or distance < close_best_distance - close_progress_epsilon:
                            close_best_distance = distance
                            close_start_time = time.perf_counter()
                            close_start_sim = int(state_info['timestamp'])
                        elif self.fast_eval and close_start_sim is not None:
                            close_elapsed = (int(state_info['timestamp']) - close_start_sim) / 1e9
                            if close_elapsed >= close_settle_timeout:
                                logger.warning(
                                    f"velocity waypoint accepted after close-range stall: distance={distance:.2f}m, "
                                    f"settle={close_elapsed:.2f}s"
                                )
                                results.append({'sensors': {'state': state_info, 'imu': imu_info}})
                                break
                        elif not self.fast_eval and close_start_time is not None:
                            close_elapsed = time.perf_counter() - close_start_time
                            if close_elapsed >= close_settle_timeout:
                                logger.warning(
                                    f"velocity waypoint accepted after close-range stall: distance={distance:.2f}m, "
                                    f"settle={close_elapsed:.2f}s"
                                )
                                results.append({'sensors': {'state': state_info, 'imu': imu_info}})
                                break
                    else:
                        close_start_time = None
                        close_start_sim = None
                        close_best_distance = None

                    wall_timeout = time.perf_counter() - target_start_time > target_timeout
                    sim_timeout = (
                        (int(state_info['timestamp']) - target_sim_start) / 1e9 > target_timeout
                        if self.fast_eval and target_sim_start is not None
                        else False
                    )
                    if (self.fast_eval and sim_timeout) or (not self.fast_eval and wall_timeout):
                        if distance <= timeout_accept_radius:
                            logger.warning(
                                f"velocity waypoint timeout but close enough: distance={distance:.2f}m, "
                                f"timeout={target_timeout:.2f}s"
                            )
                            results.append({'sensors': {'state': state_info, 'imu': imu_info}})
                            break
                        logger.warning(
                            f"velocity waypoint timeout: distance={distance:.2f}m, "
                            f"timeout={target_timeout:.2f}s"
                        )
                        collision = True
                        results.append({'sensors': {'state': state_info, 'imu': imu_info}})
                        break

                    direction = delta / max(distance, 1e-6)
                    duration = min(control_dt, max(distance / velocity, 0.01))
                    command_velocity = direction * velocity
                    airsim_client.moveByVelocityAsync(
                        float(command_velocity[0]),
                        float(command_velocity[1]),
                        float(command_velocity[2]),
                        float(duration),
                        drivetrain=drivetrain,
                        yaw_mode=yaw_mode,
                    ).join()

                if collision:
                    break

            airsim_client.moveByVelocityAsync(0, 0, 0, 0.05, drivetrain=drivetrain, yaw_mode=yaw_mode).join()
            final_state_info = copy.deepcopy(state_sensor.retrieve()) if self.fast_eval else state_info
            airsim_client.simPause(True)
            action_wall_ms = (time.perf_counter() - action_wall_start) * 1000.0
            action_sim_ms = (
                max(0.0, (int(final_state_info['timestamp']) - action_sim_start) / 1e6)
                if self.fast_eval and final_state_info is not None and action_sim_start is not None
                else 0.0
            )
            return {
                'states': results,
                'collision': collision,
                'wall_time_ms': action_wall_ms,
                'sim_time_ms': action_sim_ms,
            }

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(
                        move_path,
                        (
                            self.airsim_clients[index_1][index_2],
                            waypoints_list[index_1][index_2],
                            start_states[index_1][index_2],
                        ),
                    )
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        result_poses_list = []
        error_flag = False
        for index_1, _ in enumerate(threads):
            result_poses_list.append([])
            for index_2, _ in enumerate(threads[index_1]):
                result = threads[index_1][index_2].get_result()
                result_poses_list[index_1].append(result)
                if result is None:
                    error_flag = True
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('move path by velocity waypoints failed.')
            return None
        if error_flag:
            return None
        return result_poses_list
    
    def setPoses(self, poses: list) -> bool:
        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose) -> None:
            if airsim_client is None:
                raise Exception('error')
                return

            airsim_client.simSetKinematics(
                state=pose,
                ignore_collision=True,
            )
            airsim_client.simContinueForFrames(1)
            airsim_client.simPause(True)

            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses失败')
            return False

        return True
    
    def setObjects(self, object_list: list):
        def _setObject(airsim_client: airsim.VehicleClient, object_info: dict) -> None:
            if airsim_client is None:
                raise Exception('error')
                return
            asset_name = object_info['asset_name']
            pose = object_info['pose']
            scale = object_info['scale']
            object_cnt = object_info['object_cnt']
            if object_cnt > 0:
                airsim_client.simDestroyObject('my_object_' + str(object_cnt - 1))
            success = airsim_client.simSpawnObject(
                    'my_object_' + str(object_cnt), asset_name, pose, scale, physics_enabled=False, is_blueprint=False)
            airsim_client.simContinueForFrames(1)
            airsim_client.simPause(True)
            return success

        threads = []
        thread_results = []
        cnt = 0
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                object_list[cnt]['object_cnt'] = self.objects_name_cnt[index_1][index_2]
                threads[index_1].append(
                    MyThread(_setObject, (self.airsim_clients[index_1][index_2], object_list[cnt]))
                )
                self.objects_name_cnt[index_1][index_2] += 1
                cnt += 1

        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('set Object失败')
            return False
        return True
    
    def getImageResponses(self, cameras=['FrontCamera', 'LeftCamera', 'RightCamera', 'RearCamera', 'DownCamera'], poses=None):
        def _getImages(airsim_client: airsim.VehicleClient):
            if airsim_client is None:
                raise Exception('client is None.')
                return None, None
            time_sleep_cnt = 0
            while True:
                try:
                    ImageRequest = []
                    for camera_name in cameras:
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False))
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
                    image_datas = airsim_client.simGetImages(requests=ImageRequest)
                    images, depth_images = [], []
                    for idx, camera_name in enumerate(cameras):
                        rgb_resp = image_datas[2 * idx]
                        image = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(rgb_resp.height, rgb_resp.width, 3)
                        depth_resp = image_datas[2* idx + 1]
                        depth_img_in_meters = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)
                        depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
                        images.append(image)
                        depth_images.append(depth_image)
                    break
                except Exception as e:
                    time_sleep_cnt += 1
                    logger.error("图片获取错误: " + str(e))
                    logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                    time.sleep(1)
                if time_sleep_cnt > 3:
                    raise Exception('图片获取失败')
            return images, depth_images

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None

        return responses
    
    
    def getImageResponsesForRecord(self, cameras=['FrontCameraRecord', 'DownCameraRecord'], poses=None):
        def _getImages(airsim_client: airsim.VehicleClient):
            if airsim_client is None:
                raise Exception('client is None.')
                return None, None
            time_sleep_cnt = 0
            while True:
                try:
                    ImageRequest = []
                    for camera_name in cameras:
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False))
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
                    image_datas = airsim_client.simGetImages(requests=ImageRequest)
                    images, depth_images = [], []
                    for idx, camera_name in enumerate(cameras):
                        rgb_resp = image_datas[2 * idx]
                        image = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(rgb_resp.height, rgb_resp.width, 3)
                        depth_resp = image_datas[2* idx + 1]
                        depth_img_in_meters = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)
                        depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
                        images.append(image)
                        depth_images.append(depth_image)
                    break
                except Exception as e:
                    time_sleep_cnt += 1
                    logger.error("图片获取错误: " + str(e))
                    logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                    time.sleep(1)
                if time_sleep_cnt > 3:
                    raise Exception('图片获取失败')
            return images, depth_images

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None

        return responses

    def getSensorInfo(self, ):
        def get_sensor_info(airsim_client: airsim.VehicleClient, ):
            state_sensor = State(airsim_client, )
            imu_sensor = Imu(airsim_client)
            state_info = state_sensor.retrieve()
            imu_info = imu_sensor.retrieve()
            return {'sensors': {'state':state_info, 'imu': imu_info}}
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(get_sensor_info, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        results = []
        for index_1, _ in enumerate(threads):
            results.append([])
            for index_2, _ in enumerate(threads[index_1]):
                results[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getSensorInfo failed.')
            return None
        return results 
