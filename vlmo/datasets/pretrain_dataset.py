from .vg_caption_dataset import VisualGenomeCaptionDataset
from .coco_caption_karpathy_dataset import CocoCaptionKarpathyDataset
from .f30k_caption_karpathy_dataset import F30KCaptionKarpathyDataset
from .conceptual_caption_dataset import ConceptualCaptionDataset
from .sbu_caption_dataset import SBUCaptionDataset
from .wikibk_dataset import WikibkDataset
from .vqav2_dataset import VQAv2Dataset
from .nlvr2_dataset import NLVR2Dataset

from torch.utils.data.dataset import ConcatDataset
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import torch
import functools

_datasets = {
    "vg": VisualGenomeCaptionDataset,
    "f30k": F30KCaptionKarpathyDataset,
    "coco": CocoCaptionKarpathyDataset,
    "gcc": ConceptualCaptionDataset,
    "sbu": SBUCaptionDataset,
    "wikibk": WikibkDataset,
    "vqa": VQAv2Dataset,
    "nlvr2": NLVR2Dataset,
}

def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )

class Pretrain_dataset:
    def __init__(self, config):
        datasets_list = config["datasets"]
        no_false = ['coco', 'f30k']
        self.train_dataset = ConcatDataset([_datasets[dataset](
            config["data_root"],
            (
                ["default_train"]
                if len(config["train_transform_keys"]) == 0
                else config["train_transform_keys"]
            ),
            split="train",
            image_size=config["image_size"],
            max_text_len=config["max_text_len"],
            draw_false_image=config["draw_false_image"],
            draw_false_text=config["draw_false_text"],
            image_only=config["image_only"],
        ) for dataset in datasets_list])

        self.val_dataset = ConcatDataset([_datasets[dataset](
            config["data_root"],
            (
                ["default_val"]
                if len(config["val_transform_keys"]) == 0
                else config["val_transform_keys"]
            ),
            split="val",
            image_size=config["image_size"],
            max_text_len=config["max_text_len"],
            draw_false_image=config["draw_false_image"] if dataset not in no_false else 0,
            draw_false_text=config["draw_false_text"] if dataset not in no_false else 0,
            image_only=config["image_only"],
        ) for dataset in datasets_list])

        self.test_dataset = ConcatDataset([_datasets[dataset](
            config["data_root"],
            (
                ["default_val"]
                if len(config["val_transform_keys"]) == 0
                else config["val_transform_keys"]
            ),
            split="test",
            image_size=config["image_size"],
            max_text_len=config["max_text_len"],
            draw_false_image=config["draw_false_image"],
            draw_false_text=config["draw_false_text"],
            image_only=config["image_only"],
        ) for dataset in datasets_list])

        tokenizer = config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        # self.vocab_size = self.tokenizer.vocab_size

        self.train_dataset.tokenizer = self.tokenizer
        self.val_dataset.tokenizer = self.tokenizer
        self.test_dataset.tokenizer = self.tokenizer

        collator = (
            DataCollatorForWholeWordMask
            if config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=config["mlm_prob"]
        )
        self.collate = functools.partial(
            self.train_dataset.collate, mlm_collator=self.mlm_collator,
        )


    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_collate(self):
        return self.collate