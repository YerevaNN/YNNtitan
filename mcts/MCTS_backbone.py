"""
https://github.com/zhentingqi/rStar/blob/17e499b7b76f92af9125c148b079c9d0c4484c9e/run_src/MCTS_backbone.py
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import math, random

node_cnt = 0
safe_with_rewards = []

def final_results():
    return safe_with_rewards

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class MCTS_Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    def __init__(self) -> None:
        super().__init__()

        global node_cnt
        self.id = node_cnt
        node_cnt += 1

        self.rollout_id = None

    def set_rollout_id(self, rollout_id: int):
        self.rollout_id = rollout_id

    @abstractmethod
    def find_children(self, rollout_id: int):
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self):
        raise NotImplementedError


class MCTS_Searcher:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(
        self,
        exploration_weight: float,
        weight_scheduler: str,
        num_rollouts: int,
        verbose: bool = False,
    ):
        self.Q: Dict[MCTS_Node, float] = defaultdict(lambda: 0.0)  # reward
        self.N: Dict[MCTS_Node, int] = defaultdict(lambda: 0)  # visit count
        self.parent2children: Dict[MCTS_Node, List[MCTS_Node]] = dict()  # children of each node

        # explored = expanded + simulated, i.e. has seen terminal at least once, i.e. we can calculate its UCT value, i.e. has Q and N
        self.explored_nodes = set()

        self.exploration_weight = exploration_weight
        self.weight_scheduler = weight_scheduler
        self.num_rollouts = num_rollouts

        self.verbose = verbose

        global node_cnt
        node_cnt = 0

    def do_rollout(self, root_node: MCTS_Node, rollout_id: int):
        "Make the tree one layer better. (Train for one iteration.)"
        verbose_print("==> Selecting a node...", self.verbose)
        path_1 = self._select(root_node, rollout_id)
        leaf = path_1[-1]
        verbose_print(f"==> Expanding node {leaf.id}...", self.verbose)
        children = self._expand(leaf, rollout_id)
        for child in children:
            verbose_print(f"==> Simulating node {child.id}...", self.verbose)
            reward = self._simulate(child)
            verbose_print(f"==> Backpropagating the reward {reward}...", self.verbose)
            self._backpropagate(reward, path_1 + [child])

    def _select(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            # case 1: a node does not have children, then select the node itself
            if node not in self.parent2children.keys():
                return path

            # case 2: a node has children but not all children have been explored, then randomly select an unexplored child
            # unexplored = set(self.parent2children[node]) - self.explored_nodes   # `set` introduces randomness
            unexplored = [n for n in self.parent2children[node] if n not in self.explored_nodes]
            if unexplored:
                n = random.choice(unexplored)
                path.append(n)
                return path

            # case 3: a node has children and all children have been explored, then select one child and go to the next layer
            node = self._uct_select(node, rollout_id)

    def _expand(self, node: MCTS_Node, rollout_id: int):
        "Update the `children` dict with the children of `node`"
        if node in self.explored_nodes:
            print("You are trying to expand te node that already has been expanded.")
            return []

        if node.is_terminal():
            self.explored_nodes.add(node)
            print("Terminal node is non-expandable")
            return []

        self.parent2children[node] = node.find_children(rollout_id)
        return self.parent2children[node]

    def _simulate(self, node: MCTS_Node) -> List[MCTS_Node]:
        "Returns the reward for the completion of `node`"
        node.safe = node.generator.generate_full_answer("[SAFE]" + node.node_path + ".") # we can put . in the end, because if the node_path was a full molecule, we wouldn't have called simulate on it 
        reward = node.calculate_reward()
        global safe_with_rewards
        safe_with_rewards.append((node.safe, reward))
        return reward

    def _backpropagate(self, reward, path: List[MCTS_Node]):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.Q[node] += reward
            self.N[node] += 1
            self.explored_nodes.add(node)

    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        if self.weight_scheduler == "exp":
            return self.exploration_weight * (0.1 ** (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "lin":
            return self.exploration_weight * (1 - 0.9 * (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "const":
            return self.exploration_weight

    def _uct_select(self, node: MCTS_Node, rollout_id: int):
        "Select a child of node, balancing exploration & exploitation"

        # All children of the node should already be expanded
        assert all(n in self.explored_nodes for n in self.parent2children[node])

        return max(
            self.parent2children[node], key=lambda n: self._compute_uct(parent_node=node, node=n, rollout_id=rollout_id)
        )

    def _compute_uct(self, parent_node: MCTS_Node, node: MCTS_Node, rollout_id: int):
        "Upper confidence bound for trees"
        if parent_node is None:  # uct will be invalid
            raise ValueError("UCT can't be calculated on the root node.")
        elif self.N[node] == 0:
            print("The node has not been explored yet.")
            return -10
        else:
            weight = self._get_weight(rollout_id)
            return self.Q[node] / self.N[node] + weight * math.sqrt(math.log(self.N[parent_node]) / self.N[node])
