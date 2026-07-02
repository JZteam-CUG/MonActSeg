import argparse
import os
import random
import sys

import numpy as np
import torch
from loguru import logger

from eval import edit_score, f_score, read_file
from fused_batch_gen import BatchGenerator
from fused_trainer import FusedTrainer

FUSION_MODE_NAME = "attention_gate"


def _resolve_split_file(dataset, split, split_mode, split_type):
    candidates = []
    if split_mode == "loso":
        candidates.append(f"./data/{dataset}/splits/{split_type}_loso_{split}.bundle")
    candidates.append(f"./data/{dataset}/splits/{split_type}_split_{split}.bundle")
    candidates.append(f"./data/{dataset}/splits/{split_type}.split{split}.bundle")
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def run_evaluation_script(dataset, split, actions_dict, results_dir_custom, split_mode="cv"):
    logger.info(f">>> Final Evaluation (Dir: {results_dir_custom}) <<<")
    gt_path = f"./data/{dataset}/groundTruth/"
    res_path = results_dir_custom + "/"
    file_list = _resolve_split_file(dataset, split, split_mode, "test")
    if not os.path.exists(file_list):
        logger.error(f"Missing test split file: {file_list}")
        return
    list_of_videos = read_file(file_list).split("\n")[:-1]
    if not list_of_videos:
        logger.error(f"Empty test split file: {file_list}")
        return

    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    correct, total, edit = 0, 0, 0
    overlap = [0.1, 0.25, 0.5]

    for vid in list_of_videos:
        gt = read_file(gt_path + vid).split("\n")[:-1]
        res_file = os.path.join(res_path, vid.split(".")[0] + ".txt")
        if not os.path.exists(res_file):
            raise FileNotFoundError(f"Missing prediction file: {res_file}")
        rec = read_file(res_file).split("\n")[1].split()

        if len(gt) != len(rec):
            raise ValueError(f"Prediction length mismatch for {vid}: gt={len(gt)} pred={len(rec)}")
        for i in range(len(gt)):
            total += 1
            if gt[i] == rec[i]:
                correct += 1
        edit += edit_score(rec, gt)

        for s in range(len(overlap)):
            tp1, fp1, fn1, _ = f_score(rec, gt, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    logger.info("=" * 40)
    logger.info(f"Dataset: {dataset} | Split: {split} | Mode: {results_dir_custom.split('_')[-1]}")
    logger.info(f"Accuracy: {100 * correct / total:.4f}%")
    logger.info(f"Edit Score: {edit / len(list_of_videos):.4f}")
    for s in range(len(overlap)):
        p = tp[s] / (tp[s] + fp[s]) if tp[s] + fp[s] > 0 else 0
        r = tp[s] / (tp[s] + fn[s]) if tp[s] + fn[s] > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        logger.info(f"F1@{overlap[s]}: {f1 * 100:.4f}")
    logger.info("=" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="monkey")
    parser.add_argument("--split", default="1")
    parser.add_argument(
        "--split_mode",
        choices=["cv", "loso"],
        default="cv",
        help="Split file mode: cv (train/val/test_split_*.bundle) or loso (train/val/test_loso_*.bundle).",
    )
    parser.add_argument("--bz", default=4, type=int)
    parser.add_argument("--lr_phase1", default=0.0005, type=float)
    parser.add_argument("--lr_phase2", default=0.0001, type=float)
    parser.add_argument("--epochs_phase1", default=100, type=int)
    parser.add_argument("--epochs_phase2", default=50, type=int)
    parser.add_argument("--rgb_features_dim", default=2048, type=int)
    parser.add_argument("--skeleton_features_dim", default=6, type=int)
    parser.add_argument("--num_f_maps", default=64, type=int)
    parser.add_argument("--num_layers_PG", default=10, type=int)
    parser.add_argument("--num_layers_RF", default=10, type=int)
    parser.add_argument("--num_R", default=3, type=int)
    args = parser.parse_args()

    dataset_name = args.dataset
    exp_name = f"split_{args.split}_mode_{FUSION_MODE_NAME}_R_{args.num_R}"
    model_dir = f"./models/{dataset_name}/{exp_name}"
    results_dir = f"./results/{dataset_name}/{exp_name}"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    os.makedirs("logs", exist_ok=True)
    logger.add(f"logs/train_{exp_name}_{{time}}.log")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vid_list_file = _resolve_split_file(dataset_name, args.split, args.split_mode, "train")
    vid_list_file_val = _resolve_split_file(dataset_name, args.split, args.split_mode, "val")
    vid_list_file_tst = _resolve_split_file(dataset_name, args.split, args.split_mode, "test")
    missing_splits = [p for p in (vid_list_file, vid_list_file_val, vid_list_file_tst) if not os.path.exists(p)]
    if missing_splits:
        logger.error(f"Missing split file(s): {', '.join(missing_splits)}")
        sys.exit(1)

    skeleton_features_path = f"./data/{dataset_name}/features/skeleton/"
    rgb_features_path = f"./data/{dataset_name}/features/rgb/"
    gt_path = f"./data/{dataset_name}/groundTruth/"
    mapping_file = f"./data/{dataset_name}/mapping.txt"

    with open(mapping_file, "r", encoding="utf-8") as f:
        actions = f.readlines()
    actions_dict = {a.strip(): i for i, a in enumerate(actions)}

    sample_rate = 1
    if dataset_name == "50salads":
        sample_rate = 2

    trainer = FusedTrainer(args, actions_dict, device)
    batch_gen = BatchGenerator(
        len(actions_dict),
        actions_dict,
        gt_path,
        skeleton_features_path,
        rgb_features_path,
        sample_rate,
    )
    batch_gen_val = BatchGenerator(
        len(actions_dict),
        actions_dict,
        gt_path,
        skeleton_features_path,
        rgb_features_path,
        sample_rate,
    )
    trainer.train_phase_1(batch_gen, batch_gen_val, vid_list_file, vid_list_file_val, model_dir)
    trainer.train_phase_2(batch_gen, batch_gen_val, vid_list_file, vid_list_file_val, model_dir)

    trainer.predict(
        os.path.join(model_dir, "best_model.pth"),
        results_dir,
        skeleton_features_path,
        rgb_features_path,
        vid_list_file_tst,
        sample_rate,
    )
    logger.info("Final Evaluation:")
    run_evaluation_script(dataset_name, args.split, actions_dict, results_dir, args.split_mode)


if __name__ == "__main__":
    main()
