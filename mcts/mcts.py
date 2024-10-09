# Licensed under the MIT license.

import sys
import random
import numpy as np
import torch

import MCST_for_testing

from argparse import ArgumentParser

sys.path.append(".")

def main(args):
    if(args.seed):
        fix_seeds(args.seed)
    generator = MCST_for_testing.Generator(args.num_children)
    nodes_with_rewards = MCST_for_testing.search_for_answers(args, generator)
    print(nodes_with_rewards[:10])

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--num_rollouts", type=int, default=100000)
    parser.add_argument("--num_children", type=int, default=77)
    parser.add_argument("--max_depth_allowed", type=int, default=6)
    parser.add_argument("--mcts_exploration_weight", type=float, default=7)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    # parser.add_argument("--mcts_discount_factor", type=float, default=0.8)

    # parser.add_argument("--print_tree", action="store_true", default="true")

    # parser.add_argument("--num_beams", type=int, default=1, help="num_beams")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)