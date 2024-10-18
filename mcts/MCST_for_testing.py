import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

import sys
sys.path.append(".")

import string
from tqdm import trange
from typing import List

from MCTS_backbone import MCTS_Searcher, MCTS_Node, final_results

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)

class StopForFragment(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = self.tokenizer.decode(input_ids[0][-1])
        return '.' in last_token or "[/SAFE]" == last_token

class Stop(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = self.tokenizer.decode(input_ids[0][-1])
        return "[/SAFE]" == last_token
    
class Generator:
    """Generator generates children nodes"""

    def __init__(self, num_children, model_path, tokenizer_path) -> None:
        self.num_children = num_children
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.stop_for_fragment = StopForFragment(self.tokenizer)
        self.stop_for_molecule = Stop(self.tokenizer)

    def generate_answers(self, user_question: str = None):
        input_ids = self.tokenizer(user_question, return_tensors="pt")["input_ids"]

        outputs = self.model.generate(
            input_ids,
            stopping_criteria=StoppingCriteriaList([self.stop_for_fragment]),
            do_sample=True,
            num_return_sequences=self.num_children,
            max_new_tokens=1000
        )

        # only decode the newly generated part
        decoded_outputs = [self.tokenizer.decode(output[len(input_ids[0]):], skip_special_tokens=True) for output in outputs]
        cropped_answers = [decoded_output.split('.')[0] for decoded_output in decoded_outputs]

        return cropped_answers

    def generate_full_answer(self, user_question: str):
        input_ids = self.tokenizer(user_question, return_tensors="pt")["input_ids"]
        output = self.model.generate(
            input_ids,
            stopping_criteria=StoppingCriteriaList([self.stop_for_molecule]),
            do_sample=True,
            num_return_sequences=1,
            max_new_tokens=1000
        )
        decoded_output = self.tokenizer.decode(output[0])
        return decoded_output

class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        max_depth: int,
        verbose: bool = False,
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

        answers = self.generator.generate_answers(self.node_path)
        for ans in answers:
            if self.node_path == "[SAFE]": # root node
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        added_fragment = ans,
                        node_path=self.node_path + ans,
                        max_depth=self.max_depth
                    )
                )
            else:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        added_fragment = ans,
                        node_path=self.node_path + "." + ans,
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
        return self.node_path.endswith("[/SAFE]") or self.depth >= self.max_depth

    def calculate_reward(self):
        molecule_str = self.safe
        last_safe_index = molecule_str.rfind("[SAFE]")
        print(molecule_str)
        molecule_str = molecule_str[last_safe_index + 6:-7].strip() # remove [SAFE] and [/SAFE] tags
        print(molecule_str)

        mol = Chem.MolFromSmiles(molecule_str)
        if mol is None:
            raise ValueError("Invalid molecule string {mol}")

        ring_count = mol.GetRingInfo().NumRings()
        if not (3 <= ring_count <= 5):
            return -1e6

        tpsa = rdMolDescriptors.CalcTPSA(mol)
        return tpsa

def search_for_answers(args, generator: Generator = None, user_question: str = None):
    verbose_print(
        f"********************* Searching for answers ********************* ", args.verbose
    )

    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        verbose=args.verbose,
        uct_algo=args.uct_algo
    )

    root_node = Reasoning_MCTS_Node(
        parent=None,
        verbose=args.verbose,
        node_path="[SAFE]",
        generator=generator,
        max_depth=args.max_depth_allowed
    )

    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        mcts_searcher.do_rollout(root_node, i)

    return final_results()
