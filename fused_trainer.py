import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from torch import optim

from fused_batch_gen import get_features
from models.fused_model import MultiStageModel

def soften_and_normalize_boundary(boundary_seq, boundary_smooth, device):
    if boundary_smooth is not None:
        boundary_seq = boundary_seq.detach().cpu().numpy()
        if boundary_seq.shape[1] == 0:
            return torch.from_numpy(boundary_seq).to(device)
        boundary_seq = gaussian_filter1d(boundary_seq, sigma=boundary_smooth, axis=1)
        temp_seq = np.zeros_like(boundary_seq)
        mid = boundary_seq.shape[1] // 2
        if mid > 0:
            temp_seq[:, mid] = 1
            temp_seq[:, mid - 1] = 1
        else:
            temp_seq[:, 0] = 1
        norm_z = gaussian_filter1d(temp_seq, sigma=boundary_smooth, axis=1).max()
        boundary_seq[boundary_seq > norm_z] = norm_z
        max_val = boundary_seq.max()
        if max_val > 0:
            boundary_seq /= max_val
        boundary_seq = torch.from_numpy(boundary_seq).to(device)
    return boundary_seq


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, reduction="mean", ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FusedTrainer:
    def __init__(self, args, actions_dict, device):
        self.args = args
        self.actions_dict = actions_dict
        self.device = device
        self.num_classes = len(actions_dict)
        self.dil = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        logger.info("Initializing MultiStageModel with Fusion_Attention_Gate")

        self.model = MultiStageModel(
            dil=self.dil,
            num_layers_RF=args.num_layers_RF,
            num_R=args.num_R,
            num_f_maps=args.num_f_maps,
            skeleton_features_dim=args.skeleton_features_dim,
            rgb_features_dim=args.rgb_features_dim,
            num_classes=self.num_classes,
            num_layers_PG=args.num_layers_PG,
            feedback_features=False,
        )
        self.model.to(device)

        self.ce = FocalLoss(alpha=0.5, gamma=2, reduction="mean", ignore_index=-100)
        self.mse = nn.MSELoss(reduction="none")
        self.ce_none = FocalLoss(alpha=0.5, gamma=2, reduction="none", ignore_index=-100)
        self.smooth_max = 4

    def _apply_input_noise(self, ske_in, rgb_in):
        return ske_in, rgb_in

    def load_phase1(self, ckpt_path):
        if not ckpt_path or not os.path.exists(ckpt_path):
            return False
        state = torch.load(ckpt_path, map_location=self.device)
        model_state = self.model.state_dict()

        filtered = {}
        skipped_mismatch = []
        skipped_non_branch = []
        for k, v in state.items():
            if k.startswith("fusion_modules"):
                skipped_non_branch.append(k)
                continue
            if not (k.startswith("skeleton_") or k.startswith("rgb_")):
                skipped_non_branch.append(k)
                continue
            if k not in model_state:
                skipped_mismatch.append(k)
                continue
            if model_state[k].shape != v.shape:
                skipped_mismatch.append(k)
                continue
            filtered[k] = v

        self.model.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded Phase 1 single-branch weights from: {ckpt_path}")
        logger.info(
            f"Phase1 load: loaded={len(filtered)} skipped_non_branch={len(skipped_non_branch)} skipped_mismatch={len(skipped_mismatch)}"
        )
        return True

    def load_phase1_dual(self, ckpt_ske, ckpt_rgb):
        if not ckpt_ske or not ckpt_rgb:
            return False
        if not os.path.exists(ckpt_ske) or not os.path.exists(ckpt_rgb):
            return False

        state_ske = torch.load(ckpt_ske, map_location=self.device)
        state_rgb = torch.load(ckpt_rgb, map_location=self.device)
        if "state_dict" in state_ske:
            state_ske = state_ske["state_dict"]
        if "state_dict" in state_rgb:
            state_rgb = state_rgb["state_dict"]

        model_state = self.model.state_dict()
        filtered = {}
        skipped_mismatch = []
        skipped_non_branch = []

        for k, v in state_ske.items():
            if not k.startswith("skeleton_"):
                skipped_non_branch.append(k)
                continue
            if k not in model_state or model_state[k].shape != v.shape:
                skipped_mismatch.append(k)
                continue
            filtered[k] = v

        for k, v in state_rgb.items():
            if not k.startswith("rgb_"):
                skipped_non_branch.append(k)
                continue
            if k not in model_state or model_state[k].shape != v.shape:
                skipped_mismatch.append(k)
                continue
            filtered[k] = v

        self.model.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded Phase 1 dual-branch weights from: ske={ckpt_ske} rgb={ckpt_rgb}")
        logger.info(
            f"Phase1 dual load: loaded={len(filtered)} skipped_non_branch={len(skipped_non_branch)} skipped_mismatch={len(skipped_mismatch)}"
        )
        return True

    def _calc_loss_single_stream(self, predictions, batch_target, mask):
        loss = 0
        boundary_gt = (batch_target[:, 1:] != batch_target[:, :-1]).float()
        boundary_gt = soften_and_normalize_boundary(boundary_gt, boundary_smooth=1, device=self.device)
        boundary_mask = mask[:, 0, 1:] * mask[:, 0, :-1]

        if isinstance(predictions, torch.Tensor) and predictions.dim() == 4:
            predictions = [predictions[i] for i in range(predictions.shape[0])]
        elif isinstance(predictions, torch.Tensor) and predictions.dim() == 3:
            predictions = [predictions]
        elif isinstance(predictions, (list, tuple)):
            predictions = [p for p in predictions if isinstance(p, torch.Tensor) and p.dim() == 3]

        for p in predictions:
            p_flat = p.transpose(2, 1).contiguous().view(-1, self.num_classes)
            t_flat = batch_target.view(-1)
            loss += self.ce(p_flat, t_flat)

            loss += 0.15 * torch.mean(
                torch.clamp(
                    self.mse(
                        F.log_softmax(p[:, :, 1:], dim=1),
                        F.log_softmax(p.detach()[:, :, :-1], dim=1),
                    ),
                    min=0,
                    max=self.smooth_max,
                )
                * mask[:, :, 1:]
            )

            p_soft = F.softmax(p, dim=1)
            boundary_pred = 1.0 - torch.sum(p_soft[:, :, 1:] * p_soft[:, :, :-1], dim=1)
            boundary_loss = F.binary_cross_entropy(boundary_pred, boundary_gt, reduction="none")
            boundary_loss = (boundary_loss * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)
            loss += 0.1 * boundary_loss

        return loss

    def train_phase_1(self, batch_gen, batch_gen_val, vid_list_file, vid_list_file_val, model_dir):
        logger.info(">>> [Phase 1] Training Single Branches (Skeleton & RGB) <<<")
        best_val_loss = None
        for p in self.model.fusion_modules.parameters():
            p.requires_grad = False
        for p in self.model.skeleton_PG.parameters():
            p.requires_grad = True
        for p in self.model.rgb_PG.parameters():
            p.requires_grad = True
        for p in self.model.skeleton_Rs.parameters():
            p.requires_grad = True
        for p in self.model.rgb_Rs.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr_phase1,
            weight_decay=0.0005,
        )
        batch_gen.read_data(vid_list_file)
        batch_gen_val.read_data(vid_list_file_val)

        for epoch in range(self.args.epochs_phase1):
            self.model.train()
            self.model.fusion_modules.eval()
            epoch_loss = 0
            while batch_gen.has_next():
                ske_in, rgb_in, target, mask, _ = batch_gen.next_batch(self.args.bz)
                ske_in, rgb_in = ske_in.to(self.device), rgb_in.to(self.device)
                target, mask = target.to(self.device), mask.to(self.device)
                ske_in, rgb_in = self._apply_input_noise(ske_in, rgb_in)
                optimizer.zero_grad()

                _, ske_preds, rgb_preds, _, _ = self.model(ske_in, rgb_in, mask)

                loss_ske = self._calc_loss_single_stream(ske_preds, target, mask)
                loss_rgb = self._calc_loss_single_stream(rgb_preds, target, mask)

                loss = loss_ske + loss_rgb
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()
                epoch_loss += loss.item()
            batch_gen.reset()
            logger.info(
                f"[Phase 1] Epoch {epoch+1} Loss: {epoch_loss / len(batch_gen.list_of_examples):.4f}"
            )

            metrics = self.validate_phase1(batch_gen_val, epoch)
            val_loss = metrics["loss"]
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(model_dir, "phase1_best.pth"))
                logger.info(
                    f"[Phase 1] New best Val Loss: {val_loss:.4f}. Saved phase1_best.pth."
                )

        torch.save(self.model.state_dict(), os.path.join(model_dir, "phase1_singlebranch.pth"))

    def validate_phase1(self, batch_gen, epoch):
        """Validate on val set for Phase 1 (loss only)."""
        self.model.eval()
        epoch_loss_ske = 0
        epoch_loss_rgb = 0

        with torch.no_grad():
            while batch_gen.has_next():
                ske_in, rgb_in, target, mask, _ = batch_gen.next_batch(self.args.bz)
                ske_in, rgb_in = ske_in.to(self.device), rgb_in.to(self.device)
                target, mask = target.to(self.device), mask.to(self.device)

                _, ske_preds, rgb_preds, _, _ = self.model(ske_in, rgb_in, mask)

                loss_ske = self._calc_loss_single_stream(ske_preds, target, mask)
                loss_rgb = self._calc_loss_single_stream(rgb_preds, target, mask)
                epoch_loss_ske += loss_ske.item()
                epoch_loss_rgb += loss_rgb.item()

        batch_gen.reset()
        nb = len(batch_gen.list_of_examples)
        avg_loss_ske = epoch_loss_ske / nb if nb > 0 else 0
        avg_loss_rgb = epoch_loss_rgb / nb if nb > 0 else 0
        avg_loss = (avg_loss_ske + avg_loss_rgb) / 2.0

        logger.info(f">>> [Phase 1 Val Epoch {epoch+1}] Val Loss: {avg_loss:.4f}")
        return {"loss": avg_loss}

    def validate(self, batch_gen, epoch, phase=2):
        """Validate on val set (loss only)."""
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            while batch_gen.has_next():
                ske_in, rgb_in, target, mask, _ = batch_gen.next_batch(self.args.bz)
                ske_in, rgb_in = ske_in.to(self.device), rgb_in.to(self.device)
                target, mask = target.to(self.device), mask.to(self.device)

                fused_preds, _, _, _, _ = self.model(ske_in, rgb_in, mask)

                loss_fused = self._calc_loss_single_stream(fused_preds, target, mask)
                epoch_loss += loss_fused.item()

        batch_gen.reset()
        nb = len(batch_gen.list_of_examples)
        avg_loss = epoch_loss / nb if nb > 0 else 0

        logger.info(f">>> [Phase {phase} Val Epoch {epoch+1}] Val Loss: {avg_loss:.4f}")
        return {"loss": avg_loss}

    def train_phase_2(self, batch_gen, batch_gen_val, vid_list_file, vid_list_file_val, model_dir):
        logger.info(">>> [Phase 2] End-to-End Training (All Modules) <<<")
        phase1_best = os.path.join(model_dir, "phase1_best.pth")
        if os.path.exists(phase1_best):
            self.model.load_state_dict(torch.load(phase1_best, map_location=self.device), strict=False)
            logger.info(f"Loaded Phase 1 best model from: {phase1_best}")
        best_val_loss = None
        for p in self.model.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr_phase2, weight_decay=0.0005)
        batch_gen.read_data(vid_list_file)
        batch_gen_val.read_data(vid_list_file_val)

        for epoch in range(self.args.epochs_phase2):
            self.model.train()
            epoch_loss = 0

            while batch_gen.has_next():
                ske_in, rgb_in, target, mask, _ = batch_gen.next_batch(self.args.bz)
                ske_in, rgb_in = ske_in.to(self.device), rgb_in.to(self.device)
                target, mask = target.to(self.device), mask.to(self.device)
                ske_in, rgb_in = self._apply_input_noise(ske_in, rgb_in)
                optimizer.zero_grad()

                fused_preds, _, _, _, _ = self.model(ske_in, rgb_in, mask)

                loss_fused = self._calc_loss_single_stream(fused_preds, target, mask)
                loss = loss_fused

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()
                epoch_loss += loss.item()
            batch_gen.reset()

            nb = len(batch_gen.list_of_examples)
            metrics = self.validate(batch_gen_val, epoch, phase=2)
            val_loss = metrics["loss"]
            logger.info(
                f"[Phase 2] Epoch {epoch+1}/{self.args.epochs_phase2} "
                f"Train Loss: {epoch_loss/nb:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                logger.info(f"New Best Val Loss: {val_loss:.4f}. Saved best_model.pth.")

        torch.save(self.model.state_dict(), os.path.join(model_dir, "phase2_full.pth"))

    def predict(self, model_path, results_dir, skeleton_features_path, rgb_features_path, vid_list_file, sample_rate):
        logger.info(f"Predicting with: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.eval()
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        file_ptr = open(vid_list_file, "r")
        list_of_vids = file_ptr.read().split("\n")[:-1]
        file_ptr.close()
        idx_to_action = {v: k for k, v in self.actions_dict.items()}

        with torch.no_grad():
            for vid in list_of_vids:
                ske = np.load(os.path.join(skeleton_features_path, vid.split(".")[0] + ".npy"))
                rgb = np.load(os.path.join(rgb_features_path, vid.split(".")[0] + ".npy"))
                ske = get_features(ske)
                seq_len = min(ske.shape[1], rgb.shape[1])
                if seq_len <= 0:
                    raise ValueError(f"Empty sequence after alignment: {vid}")
                ske = ske[:, :seq_len, :, :][:, ::sample_rate, :, :]
                rgb = rgb[:, :seq_len][:, ::sample_rate]
                ske_in = torch.tensor(ske, dtype=torch.float).unsqueeze(0).to(self.device)
                rgb_in = torch.tensor(rgb, dtype=torch.float).unsqueeze(0).to(self.device)
                N, C, T, V, M = ske_in.size()
                mask = torch.ones(N, self.num_classes, T).to(self.device)

                fused_preds, _, _, _, _ = self.model(ske_in, rgb_in, mask)
                _, predicted = torch.max(fused_preds[-1].data, 1)
                predicted = predicted.squeeze(0).data.detach().cpu().numpy()

                res = []
                for i in range(len(predicted)):
                    res.extend([idx_to_action[predicted[i].item()]] * sample_rate)
                res = res[:seq_len]

                with open(os.path.join(results_dir, vid[:-4] + ".txt"), "w") as f:
                    f.write("### Frame level recognition: ###\n")
                    f.write(" ".join(res))
