"""
EOPSN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys

# fmt: off
sys.path.insert(1, sys.path[0])
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import util.misc as utils
import wandb

import time
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator, verify_results, DatasetEvaluators, COCOPanopticEvaluator,
    SemSegEvaluator
)

from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.solver import build_optimizer

from lib import (DatasetMapper, add_config,
             COCOOpenEvaluator, COCOPanopticOpenEvaluator, SemSegOpenEvaluator
                 )



class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to EOPSN.
    """

    def __init__(self, cfg, use_wandb=True):
        """
        Args:
            cfg (CfgNode):
        """
        self.clip_norm_val = 0.0
        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                self.clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        super().__init__(cfg)
        self.jaed_step = 1
        self.wandb = use_wandb

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        if self.clip_norm_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm_val)
        self.optimizer.step()

        if self.jaed_step % 2500 == 0:
            print("CLEAR CACHE")
            torch.cuda.empty_cache()
        self.jaed_step += 1

    @classmethod
    def test(self, cfg, model, evaluators=None, use_wandb=False):
        ret = super().test(cfg, model, evaluators)
        use_wandb = use_wandb or getattr(self, 'wandb', False)
        if use_wandb and utils.get_rank() == 0:
            wandb_dict = {}
            for k, v in ret.items():
                wandb_dict.update( {k+'/'+metric: value for metric, value in v.items()})
            wandb.log(wandb_dict)


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if 'Panoptic' in cfg.MODEL.META_ARCHITECTURE or 'panoptic' in cfg.DATASETS.TRAIN[0]:
            evaluator_list = []
            evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
            if evaluator_type in ["coco", "coco_panoptic_seg"]:
                if cfg.DATASETS.UNSEEN_LABEL_SET != '':
                    evaluator_list.append(COCOOpenEvaluator(dataset_name, cfg, True, output_folder))
                else:
                    evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
            if evaluator_type == "coco_panoptic_seg":
                if cfg.DATASETS.UNSEEN_LABEL_SET != '':
                    evaluator_list.append(COCOPanopticOpenEvaluator(dataset_name, output_folder, cfg))
                else:
                    evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))

            if len(evaluator_list) == 1:
                return evaluator_list[0]

            return DatasetEvaluators(evaluator_list)

        if cfg.DATASETS.UNSEEN_LABEL_SET != '':
            evaluator = COCOOpenEvaluator(dataset_name, cfg, True, output_folder)
        else:
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        return evaluator

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)



    @classmethod
    def build_optimizer(cls, cfg, model):
        if "Detr" not in cfg.MODEL.META_ARCHITECTURE:
            return build_optimizer(cfg, model)
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if utils.get_rank() == 0:
        output_dir = cfg.OUTPUT_DIR
        i = 0
        while os.path.exists('{}/commit_{}'.format(output_dir, i)):
            i += 1
        os.system('git log | head -n 1 > {}/commit_{}'.format(output_dir,i))
        os.system('git diff --no-prefix > {}/diff_{}'.format(output_dir,i))

    if args.wandb and utils.get_rank()==0:
        prj_name = 'EOPSN'
        wandb.init(
            name=cfg.OUTPUT_DIR,
            project=prj_name,
            entity='openset_panoptic',
            config=cfg,
            sync_tensorboard=True,
            dir=cfg.OUTPUT_DIR,
            resume=args.resume
        )

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model, use_wandb=args.wandb)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg, use_wandb=args.wandb)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
