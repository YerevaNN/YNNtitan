# Licensed under the MIT license.

import sys
import random

sys.path.append(".")

import string
from tqdm import trange
from typing import List

from MCTS_backbone import MCTS_Searcher, MCTS_Node

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)

def create_letter_to_number_dict():
  """Creates a dictionary mapping letters to numbers."""
  letters = ".abcdefghijklmnopqrstuvwxyz"
  return {letter: i for i, letter in enumerate(letters)}

letter_to_number_dict = create_letter_to_number_dict()


class Generator:
    """Generator generates children nodes"""

    def __init__(self, num_children) -> None:
        self.num_children = num_children

    def generate_answers(self, user_question: str = None):
        """Right now generates 30 possible random strings each with equal probability"""
        random_strings = []
        for _ in range(self.num_children):
            length = random.randint(1, 2)
            random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
            random_strings.append(random_string)
        # letters = "abcdefghijklmnopqrstuvwxyz"
        # random_strings = [st for st in letters]
        probs = self.num_children * [1.0/self.num_children]
        return random_strings, probs


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        max_depth: int,
        verbose: bool = False,
        node_value: float = None,
        added_fragment: string = None,
        node_path: string = None,
        generator: Generator = None
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.node_value = node_value
        self.added_fragment = added_fragment
        self.node_path = node_path
        self.max_depth = max_depth

        if parent is None: 
            self.verbose = verbose
            self.generator = generator
            self.depth = 0
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.generator = parent.generator
            self.depth = parent.depth + 1

    def _create_children(self):
        verbose_print(f"---- Generating children for node {self.id}...", self.verbose)

        strings, probs = self.generator.generate_answers()
        for ans, prob in zip(strings, probs):
            if self.node_path:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        node_value=prob,
                        added_fragment = ans,
                        node_path=self.node_path + "." + ans,
                        max_depth=self.max_depth
                    )
                )
            else:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        node_value=prob,
                        added_fragment = ans,
                        node_path=ans,
                        max_depth=self.max_depth
                    )
                )

        assert self.children
        return self.children

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth

    def calculate_reward(self):
        ans = 0
        for char in self.node_path:
            ans += letter_to_number_dict[char]
        return ans
        # return self.node_value

def find_valid_solution_nodes(root_node):
    valid_solution_nodes = []

    def recursion(node:Reasoning_MCTS_Node):
        if node.is_terminal():
            valid_solution_nodes.append(node)
            return

        if not node.children:
            return

        for child in node.children:
            recursion(child)

    recursion(root_node)

    return valid_solution_nodes


def search_for_answers(args, generator: Generator = None, user_question: str = None):
    verbose_print(
        f"********************* Searching for answers ********************* ", args.verbose
    )

    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        verbose=args.verbose,
    )

    root_node = Reasoning_MCTS_Node(
        parent=None,
        verbose=args.verbose,
        node_value=0,
        node_path="",
        generator=generator,
        max_depth=args.max_depth_allowed
    )

    model_rollout_nodes = []
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)
    all_solution_nodes = find_valid_solution_nodes(root_node)
    nodes_with_rewards = [(node.calculate_reward(), node) for node in all_solution_nodes]
    nodes_with_rewards.sort(key=lambda x: x[0], reverse=True)
    strings_with_rewards = [(node.node_path, reward) for reward, node in nodes_with_rewards]

    return strings_with_rewards
