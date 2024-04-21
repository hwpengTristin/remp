from functools import partial
import random
from typing import List
import math
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, MultiLineString, box
from shapely.ops import unary_union
import ompl.base as ob
import ompl.geometric as og
import ompl.util

from mcts.node import Action

ompl.util.setLogLevel(ompl.util.LogLevel.LOG_WARN)

from constants import *
from utils import (
    shapely_rotate_translate_with_center,
    shapely_collision_check_ids,
    CustomSE2StateSpace,
    shapely_collision_check,
    plot_polygons,
    compute_long_axis_radius
)

import signal


def handler(signum, frame):
    raise TimeoutError()

def search_worker(greedy_instance, obj_polys, obj_start_poses, obj_goal_poses, obj_action_types, result_queue):
        t1 = time.time()
        res_cost = greedy_instance.search(obj_polys, obj_start_poses, obj_goal_poses, obj_action_types)

        result_queue.put(res_cost)

signal.signal(signal.SIGALRM, handler)

class Greedy_Solver:
    def __init__(self) -> None:
        self.np_rng = np.random.default_rng(seed=42)

        self.pick_drag_cost_scale = 1.0

        self.boundary_box = box(BOUNDARY[0, 0], BOUNDARY[1, 0], BOUNDARY[0, 1], BOUNDARY[1, 1])

        self.init_rrt()
    def init_rrt(self) -> None:
        space = CustomSE2StateSpace()
        bounds = ob.RealVectorBounds(2)  # type: ignore
        bounds.setLow(0, BOUNDARY[0, 0])
        bounds.setHigh(0, BOUNDARY[0, 1])
        bounds.setLow(1, BOUNDARY[1, 0])
        bounds.setHigh(1, BOUNDARY[1, 1])
        bounds.setLow(2, -math.pi)
        bounds.setHigh(2, math.pi)
        space.setBounds(bounds)

        self.rrt_ss = og.SimpleSetup(space)
        self.rrt_ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(Greedy.is_state_valid_rrt, self)))
        space.setup()
        self.rrt_ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)

        # RRTConnect, TRRT, RRTstar, InformedRRTstar, SORRTstar, TRRT
        self.planner_fast = og.RRTConnect(self.rrt_ss.getSpaceInformation())
        self.planner_fast.setRange(0.3)
        self.planner_fast.setup()

        self.planner_optimal = og.LazyLBTRRT(self.rrt_ss.getSpaceInformation())
        self.planner_optimal.setRange(0.3)
        self.planner_optimal.setup()

    
    def search(
        self,
        obj_polys: List[Polygon],
        obj_start_poses: np.ndarray,
        obj_goal_poses: np.ndarray = None,
        obj_action_types: List[int] = None,
    ) -> List[float]:
        
        # s = Greedy()
        # result = s.search(obj_polys,obj_start_poses,obj_goal_poses,obj_action_types)
        num_processes = multiprocessing.cpu_count() - 1  # You can adjust the number of processes as needed
        processes = []
        result_queue = multiprocessing.Queue()

        for _ in range(num_processes):
            greedy_instance = Greedy()  # Initialize a Greedy instance for each process
            process = multiprocessing.Process(target=search_worker, args=(greedy_instance, obj_polys, obj_start_poses, obj_goal_poses, obj_action_types, result_queue))
            processes.append(process)

            process.start()

        print('waiting for results')
        results = []
        start_time = time.time()
        first_process_finished = False
        time_limit = 40
        results = []
        while not first_process_finished and time.time() - start_time < time_limit:
            if not result_queue.empty():
                result = result_queue.get(timeout=5)
                results.append(result)
                if not first_process_finished:
                    first_process_finished = True
                    break
        first_time = time.time()  # Update start time after the first process finishes
        time_limit = 2
        while time.time() - first_time < time_limit and len(results) < num_processes:
            if not result_queue.empty():
                result = result_queue.get(timeout=1)
                results.append(result)  
    
        for process in processes:
            process.terminate()
            process.join()

        

        min_index = min(range(len(results)), key=lambda i: results[i][-1])

        print("------------------------")
        print("Choose the "+ str(min_index) +"th from ", str(len(results)) + " results")
        return results[min_index][0],results[min_index][1], results[min_index][2]



    def is_state_equal(self, poses1: np.ndarray, poses2: np.ndarray) -> bool:
        diff = poses1[:, 2] - poses2[:, 2]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return (
            np.allclose(poses1[:, 0:2], poses2[:, 0:2], atol=IS_CLOSE_TRANS_THRESHOLD) 
            and np.all(np.abs(diff) < IS_CLOSE_ROT_THRESHOLD)
        )

    def is_state_valid_rrt(self, state) -> bool:
        pose = (state.getX(), state.getY(), state.getYaw())
        robot_shape = shapely_rotate_translate_with_center(self.rrt_obj_poly, pose[0], pose[1], pose[2], 0, 0)

        if not robot_shape.within(self.boundary_box):
            return False

        if robot_shape.intersects(self.rrt_obs_polys):
            return False

        return True
    def solve_drag_rrt(self, obj_polys: List[Polygon], obj_poses: np.ndarray, action: Action, optimal=False):
        """Check if the goal pose is reachable"""

        obj_poly = obj_polys[action.obj_id]
        obj_pose = obj_poses[action.obj_id].copy()
        obstacles = obj_polys[: action.obj_id] + obj_polys[action.obj_id + 1 :]
        self.rrt_obs_polys = unary_union(obstacles)
        goal_pose = action.goal_pose

        # rotate and translate the object to the origin
        self.rrt_obj_poly = shapely_rotate_translate_with_center(
            obj_poly, -obj_pose[0], -obj_pose[1], -obj_pose[2], obj_pose[0], obj_pose[1]
        )

        start = ob.State(self.rrt_ss.getStateSpace())
        start().setX(obj_pose[0])
        start().setY(obj_pose[1])
        start().setYaw(obj_pose[2])
        goal = ob.State(self.rrt_ss.getStateSpace())
        goal().setX(goal_pose[0])
        goal().setY(goal_pose[1])
        goal().setYaw(goal_pose[2])
        self.rrt_ss.setStartAndGoalStates(start, goal)
        assert self.is_state_valid_rrt(start())
        assert self.is_state_valid_rrt(goal())
        if not optimal:
            self.rrt_ss.setPlanner(self.planner_fast)
            self.planner_fast.clear()
        else:
            self.rrt_ss.setPlanner(self.planner_optimal)
            self.planner_optimal.clear()

        if not optimal:
            self.rrt_ss.solve(0.2)
            if self.rrt_ss.haveExactSolutionPath():
                path = self.rrt_ss.getSolutionPath()
                cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * path.length()
                return True, cost, path
            else:
                return False, None, None
        else:
            t = 2
            count = 0
            while True:
                try:
                    signal.alarm(t)
                    self.rrt_ss.solve(t - 0.1)
                    signal.alarm(0)
                    if self.rrt_ss.haveExactSolutionPath():
                        break
                    else:
                        t = 3
                        print("no exact solution, try again")
                        self.rrt_ss.clear()
                except TimeoutError:
                    t = 3
                    print("timeout, try again")
                    self.rrt_ss.clear()
                finally:
                    signal.alarm(0)
                count += 1
                if count > 3:
                    print('switch to fast planner')
                    self.rrt_ss.setPlanner(self.planner_fast)

            signal.alarm(2)
            if self.rrt_ss.haveExactSolutionPath():
                path = self.rrt_ss.getSolutionPath()
                ps = og.PathSimplifier(self.rrt_ss.getSpaceInformation())
                ps.simplifyMax(path)
                if len(path.getStates()) > 2:
                    ps.smoothBSpline(path, 2)
                else:
                    path.interpolate(int(path.length() / 0.1))
                    # ps.smoothBSpline(path, 4)
                cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * path.length()
                signal.alarm(0)
                return True, cost, path
            else:
                signal.alarm(0)
                return False, None, None
class Greedy:
    def __init__(self) -> None:
        self.np_rng = np.random.default_rng(seed=42)

        self.pick_drag_cost_scale = 1.0

        self.boundary_box = box(BOUNDARY[0, 0], BOUNDARY[1, 0], BOUNDARY[0, 1], BOUNDARY[1, 1])

        self.init_rrt()

    def search(
        self,
        obj_polys: List[Polygon],
        obj_start_poses: np.ndarray,
        obj_goal_poses: np.ndarray = None,
        obj_action_types: List[int] = None,
    ) -> List[float]:
        # TODO: add a check to not undo previous action

        self.goal_poses = obj_goal_poses
        self.obj_action_types = obj_action_types
        self.obj_num = len(obj_polys)
        self.one_obj_goal_reward = 1.0
        self.obj_inner_radius = []
        self.obj_outer_radius = []
        self.obj_center_offset_radii = []
        self.obj_long_axis_angles = []
        for obj_poly, start_pose in zip(obj_polys, obj_start_poses):
            angle, short, long, center = compute_long_axis_radius(obj_poly)
            self.obj_outer_radius.append(long + 0.01)   # enlarge a little bit
            self.obj_inner_radius.append(short + 0.01)  # enlarge a little bit
            self.obj_center_offset_radii.append(center - start_pose[:2])
            self.obj_long_axis_angles.append(angle)
        # pre-compute grid actions
        self.grid_actions = []
        for radius, large_radius, center_offset, long_angle in zip(self.obj_inner_radius, self.obj_outer_radius, self.obj_center_offset_radii, self.obj_long_axis_angles):
            self.grid_actions.append(self.sample_grid_actions(radius, large_radius, center_offset, long_angle))

        # try to reach the goal
        r = self.get_best_action_at_goal(obj_polys, obj_start_poses, obj_goal_poses)
        if len(r) == 2:
            result, cost0 = r[0], r[1]
            return [result[0], result[1], cost0]
        else:
            all_collide_ids = r

        # try to move the object
        r1 = self.get_best_action_move_away(obj_polys, obj_start_poses, obj_goal_poses, all_collide_ids)

        if r1 is not None:
            result, cost1 = r1[0], r1[1]
            return [result[0], result[1], cost1]
        else:
            # print("random move")
            r2 = self.get_random_action(obj_polys, obj_start_poses, obj_goal_poses)
            if r2 is not None:
                result, cost2 = r2[0], r2[1]
                return [result[0], result[1], cost2]
            else:
                print("no action found")

    def init_rrt(self) -> None:
        space = CustomSE2StateSpace()
        bounds = ob.RealVectorBounds(2)  # type: ignore
        bounds.setLow(0, BOUNDARY[0, 0])
        bounds.setHigh(0, BOUNDARY[0, 1])
        bounds.setLow(1, BOUNDARY[1, 0])
        bounds.setHigh(1, BOUNDARY[1, 1])
        bounds.setLow(2, -math.pi)
        bounds.setHigh(2, math.pi)
        space.setBounds(bounds)

        self.rrt_ss = og.SimpleSetup(space)
        self.rrt_ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(Greedy.is_state_valid_rrt, self)))
        space.setup()
        self.rrt_ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)

        # RRTConnect, TRRT, RRTstar, InformedRRTstar, SORRTstar, TRRT
        self.planner_fast = og.RRTConnect(self.rrt_ss.getSpaceInformation())
        self.planner_fast.setRange(0.3)
        self.planner_fast.setup()

        self.planner_optimal = og.LazyLBTRRT(self.rrt_ss.getSpaceInformation())
        self.planner_optimal.setRange(0.3)
        self.planner_optimal.setup()

    def is_state_valid_rrt(self, state) -> bool:
        pose = (state.getX(), state.getY(), state.getYaw())
        robot_shape = shapely_rotate_translate_with_center(self.rrt_obj_poly, pose[0], pose[1], pose[2], 0, 0)

        if not robot_shape.within(self.boundary_box):
            return False

        if robot_shape.intersects(self.rrt_obs_polys):
            return False

        return True

    def get_distance_costs(self, obj_pose: np.ndarray, goal_pose: np.ndarray) -> float:
        trans_dist = np.linalg.norm(obj_pose[:, 0:2] - goal_pose[:, 0:2], axis=1)
        rot_dist = np.abs(obj_pose[:, 2] - goal_pose[:, 2])
        rot_dist[rot_dist > np.pi] = 2 * np.pi - rot_dist
        costs = TRANS_WEIGHT * trans_dist + ROT_WEIGHT * rot_dist
        return costs

    def get_pick_place_cost(self, obj_pose: np.ndarray, goal_pose: np.ndarray) -> float:
        """Calculate the cost of pick-and-place action"""
        trans_dist = math.sqrt((obj_pose[0] - goal_pose[0]) ** 2 + (obj_pose[1] - goal_pose[1]) ** 2)
        rot_dist = abs(obj_pose[2] - goal_pose[2])
        if rot_dist > math.pi:
            rot_dist = 2 * math.pi - rot_dist
        assert rot_dist >= 0
        # cost = PICK_PLACE_BASE_COST + PICK_PLACE_DIST_SCALE * (trans_dist * TRANS_WEIGHT + rot_dist * ROT_WEIGHT)
        cost = max(PICK_PLACE_BASE_COST + PICK_PLACE_DIST_SCALE * (trans_dist * TRANS_WEIGHT), PICK_PLACE_BASE_COST + PICK_PLACE_DIST_SCALE * (rot_dist * ROT_WEIGHT))

        return cost

    def solve_drag_rrt(self, obj_polys: List[Polygon], obj_poses: np.ndarray, action: Action, optimal=False):
        """Check if the goal pose is reachable"""

        obj_poly = obj_polys[action.obj_id]
        obj_pose = obj_poses[action.obj_id].copy()
        obstacles = obj_polys[: action.obj_id] + obj_polys[action.obj_id + 1 :]
        self.rrt_obs_polys = unary_union(obstacles)
        goal_pose = action.goal_pose

        # rotate and translate the object to the origin
        self.rrt_obj_poly = shapely_rotate_translate_with_center(
            obj_poly, -obj_pose[0], -obj_pose[1], -obj_pose[2], obj_pose[0], obj_pose[1]
        )

        start = ob.State(self.rrt_ss.getStateSpace())
        start().setX(obj_pose[0])
        start().setY(obj_pose[1])
        start().setYaw(obj_pose[2])
        goal = ob.State(self.rrt_ss.getStateSpace())
        goal().setX(goal_pose[0])
        goal().setY(goal_pose[1])
        goal().setYaw(goal_pose[2])
        self.rrt_ss.setStartAndGoalStates(start, goal)
        assert self.is_state_valid_rrt(start())
        assert self.is_state_valid_rrt(goal())
        if not optimal:
            self.rrt_ss.setPlanner(self.planner_fast)
            self.planner_fast.clear()
        else:
            self.rrt_ss.setPlanner(self.planner_optimal)
            self.planner_optimal.clear()

        if not optimal:
            self.rrt_ss.solve(0.2)
            if self.rrt_ss.haveExactSolutionPath():
                path = self.rrt_ss.getSolutionPath()
                cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * path.length()
                return True, cost, path
            else:
                return False, None, None
        else:
            t = 2
            count = 0
            while True:
                try:
                    signal.alarm(t)
                    self.rrt_ss.solve(t - 0.1)
                    signal.alarm(0)
                    if self.rrt_ss.haveExactSolutionPath():
                        break
                    else:
                        t = 3
                        print("no exact solution, try again")
                        self.rrt_ss.clear()
                except TimeoutError:
                    t = 3
                    print("timeout, try again")
                    self.rrt_ss.clear()
                finally:
                    signal.alarm(0)
                count += 1
                if count > 3:
                    print('switch to fast planner')
                    self.rrt_ss.setPlanner(self.planner_fast)

            signal.alarm(2)
            if self.rrt_ss.haveExactSolutionPath():
                path = self.rrt_ss.getSolutionPath()
                ps = og.PathSimplifier(self.rrt_ss.getSpaceInformation())
                ps.simplifyMax(path)
                if len(path.getStates()) > 2:
                    ps.smoothBSpline(path, 2)
                else:
                    path.interpolate(int(path.length() / 0.1))
                    # ps.smoothBSpline(path, 4)
                cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * path.length()
                signal.alarm(0)
                return True, cost, path
            else:
                signal.alarm(0)
                return False, None, None

    def get_best_action_at_goal(
        self, obj_polys: List[Polygon], obj_start_poses: np.ndarray, obj_goal_poses: np.ndarray
    ):
        """Move object to the goal pose"""

        move_obj_id = None
        move_cost = float("inf")
        all_collide_ids = [[]] * len(obj_polys)
        # costs = [[]] * len(obj_polys)
        for i in range(len(obj_polys)):
            if self.is_obj_at_goal(obj_start_poses[i], obj_goal_poses[i]):
                continue

            move = obj_goal_poses[i] - obj_start_poses[i]
            new_obj_poly = shapely_rotate_translate_with_center(
                obj_polys[i], move[0], move[1], move[2], obj_start_poses[i][0], obj_start_poses[i][1]
            )
            collide_ids = shapely_collision_check_ids(i, new_obj_poly, obj_polys)

            if len(collide_ids) == 0:
                # pick-and-place action
                if self.obj_action_types[i] == 0:
                    cost = self.get_pick_place_cost(obj_start_poses[i], obj_goal_poses[i])
                    if cost < move_cost:
                        move_cost = cost
                        move_obj_id = i
                # drag action
                else:
                    solved, cost, _ = self.solve_drag_rrt(obj_polys, obj_start_poses, Action(1, i, obj_goal_poses[i]))
                    if solved:
                        cost *= self.pick_drag_cost_scale
                        if cost < move_cost:
                            move_cost = cost
                            move_obj_id = i
            else:
                all_collide_ids[i] = collide_ids


        if move_obj_id is not None:
            return [move_obj_id, obj_goal_poses[move_obj_id]], move_cost
        else:
            return all_collide_ids

    def get_best_action_move_away(
        self, obj_polys: List[Polygon], obj_start_poses: np.ndarray, obj_goal_poses: np.ndarray, all_collide_ids
    ):
        """Move object which occupied the current goal to a place where it does not collide with other objects, but in direction of the goal pose
        all objects cannot move to the goal pose in one step
        only do translation, no rotation"""

        move_obj_id = None
        move_goal = None
        move_cost = float("inf")

        # generate the move away points
        for i in range(len(obj_polys)):
            for collide_id in all_collide_ids[i]:
                # build the occupied space
                collide_poly = obj_polys[collide_id]
                move = obj_goal_poses[i] - obj_start_poses[i]
                new_obj_poly = shapely_rotate_translate_with_center(
                    obj_polys[i], move[0], move[1], move[2], obj_start_poses[i][0], obj_start_poses[i][1]
                )
                obs_polys = obj_polys[:collide_id] + obj_polys[collide_id + 1 :]
                obs_polys.append(new_obj_poly)

                # try to move the collide object away
                move = obj_goal_poses[collide_id] - obj_start_poses[collide_id]
                move_dist = np.linalg.norm(move[0:2])
                move_vec = move[0:2] / move_dist
                dist_step = 0.02
                found_move = False
                for dist_step in np.arange(0.02, move_dist, 0.02):
                    for rot_step in np.arange(0, move[2], np.deg2rad(10) * math.copysign(1, move[2])):
                        next_move = np.zeros_like(obj_start_poses[collide_id])
                        next_move[0:2] = move_vec * dist_step
                        next_move[2] = rot_step
                        new_collide_poly = shapely_rotate_translate_with_center(
                            collide_poly,
                            next_move[0],
                            next_move[1],
                            next_move[2],
                            obj_start_poses[collide_id][0],
                            obj_start_poses[collide_id][1],
                        )
                        if new_collide_poly.within(self.boundary_box) and not shapely_collision_check(
                            new_collide_poly, obs_polys
                        ):
                            curr_goal_pose = obj_start_poses[collide_id] + next_move
                            if self.obj_action_types[collide_id] == 0:
                                cost = self.get_pick_place_cost(obj_start_poses[collide_id], curr_goal_pose)
                                if self.obj_action_types[i] == 1:
                                    cost *= self.pick_drag_cost_scale
                                if cost < move_cost:
                                    move_cost = cost
                                    move_obj_id = collide_id
                                    move_goal = curr_goal_pose
                                found_move = True
                            else:
                                solved, cost, _ = self.solve_drag_rrt(
                                    obj_polys,
                                    obj_start_poses,
                                    Action(self.obj_action_types[collide_id], collide_id, curr_goal_pose),
                                )
                                if solved:
                                    if self.obj_action_types[i] == 1:
                                        cost *= self.pick_drag_cost_scale
                                    if cost < move_cost:
                                        move_cost = cost
                                        move_obj_id = collide_id
                                        move_goal = curr_goal_pose
                                    found_move = True
                        if found_move:
                            break
                    if found_move:
                        break

                # random move
                if not found_move:
                    sampled_poses = self.sample_next_poses(obj_start_poses[collide_id], obj_goal_poses[collide_id], self.grid_actions[collide_id], self.obj_inner_radius[collide_id])
                    for sampled_pose in sampled_poses:
                        move = sampled_pose - obj_start_poses[collide_id]
                        new_collide_poly = shapely_rotate_translate_with_center(
                            collide_poly,
                            move[0],
                            move[1],
                            move[2],
                            obj_start_poses[collide_id][0],
                            obj_start_poses[collide_id][1],
                        )
                        if new_collide_poly.within(self.boundary_box) and not shapely_collision_check(
                            new_collide_poly, obs_polys
                        ):
                            if self.obj_action_types[collide_id] == 0:
                                cost = self.get_pick_place_cost(obj_start_poses[collide_id], sampled_pose)
                                if self.obj_action_types[i] == 1:
                                    cost *= self.pick_drag_cost_scale
                                if cost < move_cost:
                                    move_cost = cost
                                    move_obj_id = collide_id
                                    move_goal = sampled_pose
                            else:
                                solved, cost, _ = self.solve_drag_rrt(
                                    obj_polys,
                                    obj_start_poses,
                                    Action(self.obj_action_types[collide_id], collide_id, sampled_pose),
                                )
                                if solved:
                                    if self.obj_action_types[i] == 1:
                                        cost *= self.pick_drag_cost_scale
                                    if cost < move_cost:
                                        move_cost = cost
                                        move_obj_id = collide_id
                                        move_goal = sampled_pose

        if move_obj_id is not None:
            return [move_obj_id, move_goal], move_cost
        else:
            return None

    def get_random_action(self, obj_polys: List[Polygon], obj_start_poses: np.ndarray, obj_goal_poses: np.ndarray):
        # get the indices of the objects and shuffle them
        obj_ids = np.arange(self.obj_num)
        self.np_rng.shuffle(obj_ids)

        for obj_id in obj_ids:
            curr_pose = obj_start_poses[obj_id]
            goal_pose = obj_goal_poses[obj_id]
            obj_poly = obj_polys[obj_id]
            obs_polys = obj_polys[:obj_id] + obj_polys[obj_id + 1 :]
            action_type = self.obj_action_types[obj_id]


            action_goal_poses = list(self.sample_next_poses(curr_pose, goal_pose, self.grid_actions[obj_id], self.obj_inner_radius[obj_id]))
            random.shuffle(action_goal_poses)
            for action_goal_pose in action_goal_poses:
                move = action_goal_pose - curr_pose
                new_obj_poly = shapely_rotate_translate_with_center(
                    obj_poly, move[0], move[1], move[2], curr_pose[0], curr_pose[1]
                )
                if new_obj_poly.within(self.boundary_box) and not shapely_collision_check(new_obj_poly, obs_polys):
                    if action_type == 0:
                        cost = self.get_pick_place_cost(curr_pose, action_goal_pose)
                        return [obj_id, action_goal_pose], cost
                    elif action_type == 1:
                        solved, cost, _ = self.solve_drag_rrt(
                            obj_polys, obj_start_poses, Action(action_type, obj_id, action_goal_pose)
                        )
                        if solved:
                            return[obj_id, action_goal_pose], cost
        return None

    def sample_grid_actions(self, short_radius: float, long_radius: float, center_offset: float, long_angle: float) -> np.ndarray:
        """Sample poses as a grid over the boundary"""
        rot_half = (short_radius / long_radius > 0.9)
        max_nx, max_ny = 8, 5
        max_radius = 0.25

        # make grid horizontal
        nx = round((BOUNDARY[0, 1] - BOUNDARY[0, 0]) / (max_radius - long_radius))
        ny = round((BOUNDARY[1, 1] - BOUNDARY[1, 0]) / (max_radius - short_radius))
        nx = min(max_nx, nx)
        ny = min(max_ny, ny)
        x_min, x_max = BOUNDARY[0, 0] + long_radius, BOUNDARY[0, 1] - long_radius
        y_min, y_max = BOUNDARY[1, 0] + short_radius, BOUNDARY[1, 1] - short_radius
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        xx, yy = np.meshgrid(x, y)
        new_pos = np.column_stack((xx.ravel(), yy.ravel()))
        # center it
        # new_pos = new_pos - center_offset
        new_pos[:, 0] = new_pos[:, 0] - center_offset[0]
        new_pos[:, 1] = new_pos[:, 1] - center_offset[1]
        angles = np.array([-long_angle])
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        angles = np.tile(angles, len(new_pos))
        new_pose_grid_hori = np.c_[new_pos, angles]
        new_pose_grid = new_pose_grid_hori

        # make grid vertical
        if not rot_half:
            nx = round((BOUNDARY[0, 1] - BOUNDARY[0, 0]) / (max_radius - short_radius))
            ny = round((BOUNDARY[1, 1] - BOUNDARY[1, 0]) / (max_radius - long_radius))
            nx = min(max_nx, nx)
            ny = min(max_ny, ny)
            x_min, x_max = BOUNDARY[0, 0] + short_radius, BOUNDARY[0, 1] - short_radius
            y_min, y_max = BOUNDARY[1, 0] + long_radius, BOUNDARY[1, 1] - long_radius
            x = np.linspace(x_min, x_max, nx)
            y = np.linspace(y_min, y_max, ny)
            xx, yy = np.meshgrid(x, y)
            new_pos = np.column_stack((xx.ravel(), yy.ravel()))
            # center it
            new_pos[:, 0] = new_pos[:, 0] + center_offset[1]
            new_pos[:, 1] = new_pos[:, 1] - center_offset[0]
            angles = np.array([-long_angle + np.pi / 2])
            angles = (angles + np.pi) % (2 * np.pi) - np.pi
            angles = np.tile(angles, len(new_pos))
            new_pose_grid_vert = np.c_[new_pos, angles]
            new_pose_grid = np.vstack((new_pose_grid, new_pose_grid_vert))

        return new_pose_grid

    def sample_next_poses(self, curr_pose: np.ndarray, goal_pose: np.ndarray, grid_action: np.ndarray, inner_radius: float) -> np.ndarray:
        """Sample next poses for the object, all poses are within the boundary"""
        at_goal = self.is_obj_at_goal(curr_pose, goal_pose)

        # sample poses around the goal pose
        num_trans_samples = 30
        num_rot_samples = 3
        trans_low, trans_high = 0.03, 0.15
        signs = np.random.choice([-1, 1], size=(num_trans_samples, 2))
        trans_offset = self.np_rng.uniform(low=trans_low, high=trans_high, size=(num_trans_samples, 2)) * signs
        new_pos = trans_offset + goal_pose[0:2]
        new_pos[:, 0] = np.clip(new_pos[:, 0], BOUNDARY[0, 0] + inner_radius, BOUNDARY[0, 1] - inner_radius)
        new_pos[:, 1] = np.clip(new_pos[:, 1], BOUNDARY[1, 0] + inner_radius, BOUNDARY[1, 1] - inner_radius)
        new_pos = np.repeat(new_pos, num_rot_samples, axis=0)
        angles = self.np_rng.uniform(low=-np.pi, high=np.pi, size=(len(new_pos), 1))
        new_pose_goal = np.hstack((new_pos, angles))

        # sample poses within the boundary
        num_trans_samples = 30
        num_rot_samples = 3
        new_pos = self.np_rng.uniform(
            low=BOUNDARY[:2, 0] + inner_radius, high=BOUNDARY[:2, 1] - inner_radius, size=(num_trans_samples, 2)
        )
        new_pos = np.repeat(new_pos, num_rot_samples, axis=0)
        angles = self.np_rng.uniform(low=-np.pi, high=np.pi, size=(len(new_pos), 1))
        new_pose_rand = np.hstack((new_pos, angles))

        # sample poses as a grid over the boundary
        new_pose_grid = grid_action

        if not at_goal:
            # sample poses around the current pose
            num_trans_samples = 15
            num_rot_samples = 3
            trans_low, trans_high = 0.03, 0.15
            signs = np.random.choice([-1, 1], size=(num_trans_samples, 2))
            trans_offset = self.np_rng.uniform(low=trans_low, high=trans_high, size=(num_trans_samples, 2)) * signs
            new_pos = trans_offset + curr_pose[0:2]
            new_pos[:, 0] = np.clip(new_pos[:, 0], BOUNDARY[0, 0] + inner_radius, BOUNDARY[0, 1] - inner_radius)
            new_pos[:, 1] = np.clip(new_pos[:, 1], BOUNDARY[1, 0] + inner_radius, BOUNDARY[1, 1] - inner_radius)
            new_pos = np.repeat(new_pos, num_rot_samples, axis=0)
            angles = self.np_rng.uniform(low=-np.pi, high=np.pi, size=(len(new_pos), 1))
            new_pose_curr = np.hstack((new_pos, angles))
            next_poses = np.vstack((new_pose_goal, new_pose_curr, new_pose_rand, new_pose_grid))
        else:
            next_poses = np.vstack((new_pose_goal, new_pose_rand, new_pose_grid))

        return next_poses

    def is_state_equal(self, poses1: np.ndarray, poses2: np.ndarray) -> bool:
        diff = poses1[:, 2] - poses2[:, 2]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return (
            np.allclose(poses1[:, 0:2], poses2[:, 0:2], atol=IS_CLOSE_TRANS_THRESHOLD) 
            and np.all(np.abs(diff) < IS_CLOSE_ROT_THRESHOLD)
        )

    def is_obj_at_goal(self, obj_pose: np.ndarray, goal_pose: np.ndarray) -> bool:
        diff = obj_pose[2] - goal_pose[2]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        return (
            math.isclose(obj_pose[0], goal_pose[0], abs_tol=IS_CLOSE_TRANS_THRESHOLD)
            and math.isclose(obj_pose[1], goal_pose[1], abs_tol=IS_CLOSE_TRANS_THRESHOLD)
            and math.isclose(diff, 0, abs_tol=IS_CLOSE_ROT_THRESHOLD)
        )

    def is_goal_state(self, poses: np.ndarray) -> bool:
        return self.is_state_equal(poses, self.goal_poses)
