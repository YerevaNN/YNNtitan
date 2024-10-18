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
    generator = MCST_for_testing.Generator(args.num_children, args.model_path, args.tokenizer_path)
    nodes_with_rewards = MCST_for_testing.search_for_answers(args, generator)
    print(nodes_with_rewards)

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=bool, default = "true")

    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_children", type=int, default=2)
    parser.add_argument("--max_depth_allowed", type=int, default=8)
    parser.add_argument("--mcts_exploration_weight", type=float, default=7)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")

    parser.add_argument("--uct_algo", type=str, choices=["max", "mean"], default="mean")

    parser.add_argument("--model_path", default="/nfs/dgx/raid/chem/titan_outputs/hf/yerevann/Llama-3.2-1B/0e17dae1afc943d8a9bfed83/step-4000")
    parser.add_argument("--tokenizer_path", default="/nfs/dgx/raid/chem/titan_outputs/tokenizers/Llama-3.2-chem-1B-v0")

    # parser.add_argument("--mcts_discount_factor", type=float, default=0.8)

    # parser.add_argument("--num_beams", type=int, default=1, help="num_beams")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)