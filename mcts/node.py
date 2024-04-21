import math
from typing import List
from dataclasses import dataclass
import heapq
import numpy as np
from shapely.geometry import Polygon

from constants import *


@dataclass
class Action:
    """The action takes the obj_id and goal_pose.
    0: pick-and-place, 1: pick-and-drag"""

    type: int
    obj_id: int
    goal_pose: np.ndarray
    path: List[float] = None  # drag path

class FixedSizeHeap(object):
    def __init__(self, max_size, init_value):
        self.max_size = max_size
        self.data = [init_value]
        self.sum_Q = init_value

    def push(self, item):
        if len(self.data) < self.max_size:
            heapq.heappush(self.data, item)
            self.sum_Q += item
        else:
            pop_item = heapq.heappushpop(self.data, item)
            self.sum_Q = self.sum_Q - pop_item + item

    def get_data(self):
        return self.data
    
    @property
    def size(self):
        return len(self.data)


class Node:
    """MCTS search node.

    It stores the object shape and pose
    """

    def __init__(
        self,
        obj_polys: List[Polygon],
        obj_poses: np.ndarray,
        at_goal: bool,
        parent: "Node" = None,  # type: ignore
        prev_action: Action = None,  # type: ignore
        untried_actions: List[Action] = None,  # type: ignore
        cost_so_far: float = 0.0,
    ) -> None:
        """
        obj_polys should have the same center as obj_poses.
        the order of the objects should not change.
        """

        self.obj_polys = obj_polys
        self.obj_poses = obj_poses
        self.at_goal = at_goal
        self.parent = parent
        self.children = []
        self.prev_action = prev_action
        self.untried_actions = untried_actions
        self.cost_so_far = cost_so_far
        self.depth = parent.depth + 1 if parent else 0
        self.N = 0  # numer of visits
        self.Q = FixedSizeHeap(Q_LIST_SIZE, 0)  # total reward
        self.VN = 0  # number of virtual visits
        self.unexplored_children = []
        self.pool_results = []

    def __str__(self) -> str:
        return f"Poses: {self.obj_poses}\nAt Goal: {self.at_goal}\nPrev action: {self.prev_action}\nDepth: {self.depth}"

    @property
    def has_children(self) -> bool:
        return len(self.children) > 0

    @property
    def is_fully_expanded(self) -> bool:
        # return self.untried_actions is not None and len(self.untried_actions) == 0 and len(self.unexplored_children) == 0 and len(self.pool_results) == 0
        return isinstance(self.untried_actions, List) and len(self.untried_actions) == 0 and len(self.unexplored_children) == 0 and len(self.pool_results) == 0
    
    def is_terminal(self, max_depth) -> bool:
        if (self.is_fully_expanded and not self.has_children) or self.depth >= max_depth or self.at_goal:
            return True
        return False

    def best_child_with_explore(self, parallel) -> "Node":
        """Return the best child node"""
        if parallel:
            scores = [
                (c.Q.sum_Q / c.Q.size) * c.N / (c.N + c.VN) + C_PUCT * math.sqrt(math.log(self.N + self.VN) / (c.N + c.VN))
                for c in self.children
            ]
        else:
            scores = [c.Q.sum_Q / c.N + C_PUCT * math.sqrt(math.log(self.N) / c.N) for c in self.children]

        return self.children[np.argmax(scores)], scores[np.argmax(scores)]

    def best_child(self) -> "Node":
        """Return the best child node"""
        
        scores = [(c.Q.sum_Q / c.Q.size) for c in self.children]

        print(
            f"best child out of {len(self.children)}, Q: {scores[np.argmax(scores)]:.5f}, N: {self.children[np.argmax(scores)].N}, cost: {self.children[np.argmax(scores)].cost_so_far:.3f}"
        )


        if np.max(scores) == 0:
            print("all zero")
            return self.children[np.random.randint(len(self.children))]

        return self.children[np.argmax(scores)]
