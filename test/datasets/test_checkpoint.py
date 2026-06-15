# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.datasets.hf_datasets import build_hf_data_loader
from torchtitan.tokenizers.tokenizer import build_tokenizer


class TestCheckpoint:
    def test_c4_resumption(self):
        dataset_name = "c4_test"
        dataset_path = "./test/assets/c4_test"
        batch_size = 1
        seq_len = 1024
        world_size = 4
        rank = 0

        dl = self._build_dataloader(
            dataset_name, dataset_path, batch_size, seq_len, world_size, rank
        )

        it = iter(dl)
        for _ in range(250):
            next(it)
        state = dl.state_dict()
        expected_input_ids, expected_labels = next(it)

        # Create new dataloader, restore checkpoint, and check if next data yielded is the same as above
        dl = self._build_dataloader(
            dataset_name, dataset_path, batch_size, seq_len, world_size, rank
        )
        dl.load_state_dict(state)
        input_ids, labels = next(iter(dl))

        assert torch.equal(input_ids, expected_input_ids)
        assert torch.equal(labels, expected_labels)

    def test_pubchem_mini_rng_resumption(self):
        """RNG + token buffer must survive dataloader checkpoint (pubchem style)."""
        dataset_name = "pubchem_train_mini"
        dataset_path = "./test/assets/chemlactica_train_mini"
        tokenizer = build_tokenizer(
            "tiktoken", "./torchtitan/tokenizers/chemlactica-125m"
        )
        common = dict(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            data_processing_style="pubchem_data_processing",
            tokenizer=tokenizer,
            batch_size=1,
            seq_len=512,
            world_size=1,
            rank=0,
            representation_type="SMILES",
            num_workers=0,
            seed=42,
        )
        dl = build_hf_data_loader(**common)
        it = iter(dl)
        for _ in range(120):
            next(it)
        state = dl.state_dict()
        expected_input_ids, expected_labels = next(it)

        dl2 = build_hf_data_loader(**common)
        dl2.load_state_dict(state)
        input_ids, labels = next(iter(dl2))

        assert torch.equal(input_ids, expected_input_ids)
        assert torch.equal(labels, expected_labels)

    def _build_dataloader(
        self, dataset_name, dataset_path, batch_size, seq_len, world_size, rank
    ):
        tokenizer_type = "tiktoken"
        tokenizer = build_tokenizer("tiktoken", "./torchtitan/tokenizers/chemlactica-125m")
        return build_hf_data_loader(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            batch_size=1,
            seq_len=1024,
            world_size=4,
            rank=0,
        )
