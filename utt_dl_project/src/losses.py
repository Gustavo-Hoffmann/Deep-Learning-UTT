#!/usr/bin/env python3
"""Loss composta sobre a curva final predita vs Vicon."""

from __future__ import annotations

import torch
import torch.nn as nn


def _peak_to_peak(x: torch.Tensor) -> torch.Tensor:
    """Amplitude pico-a-pico via sort (compativel com DirectML; max/min falham no backward DML)."""
    sorted_x, _ = torch.sort(x, dim=-1)
    return sorted_x[:, -1] - sorted_x[:, 0]


def _safe_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Correlação de Pearson por amostra (batch), média no batch."""
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    targ_c = target - target.mean(dim=-1, keepdim=True)
    num = (pred_c * targ_c).sum(dim=-1)
    den = torch.sqrt((pred_c**2).sum(dim=-1) * (targ_c**2).sum(dim=-1) + eps)
    corr = num / den
    return corr.mean()


class CompositeCurveLoss(nn.Module):
    """
    Loss sobre pred_dl_cm = calibrado + residuo_predito:

    0.50 SmoothL1(pred, vicon)
    0.20 MSE(diff(pred), diff(vicon))
    0.15 |amp_pred - amp_vicon|  (pico-a-pico)
    0.15 1 - corr(pred, vicon)
    """

    def __init__(
        self,
        w_curve: float = 0.50,
        w_deriv: float = 0.20,
        w_amp: float = 0.15,
        w_corr: float = 0.15,
    ) -> None:
        super().__init__()
        total = w_curve + w_deriv + w_amp + w_corr
        self.w_curve = w_curve / total
        self.w_deriv = w_deriv / total
        self.w_amp = w_amp / total
        self.w_corr = w_corr / total
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        pred_dl: torch.Tensor,
        vicon: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_curve = self.smooth_l1(pred_dl, vicon)

        if pred_dl.shape[-1] > 1:
            dp = pred_dl[:, 1:] - pred_dl[:, :-1]
            dv = vicon[:, 1:] - vicon[:, :-1]
            l_deriv = torch.mean((dp - dv) ** 2)
        else:
            l_deriv = torch.tensor(0.0, device=pred_dl.device, dtype=pred_dl.dtype)

        amp_pred = _peak_to_peak(pred_dl)
        amp_true = _peak_to_peak(vicon)
        l_amp = torch.mean(torch.abs(amp_pred - amp_true))

        corr = _safe_corr(pred_dl, vicon)
        l_corr = 1.0 - corr

        total = (
            self.w_curve * l_curve
            + self.w_deriv * l_deriv
            + self.w_amp * l_amp
            + self.w_corr * l_corr
        )
        parts = {
            "loss_total": float(total.detach().cpu()),
            "loss_curve": float(l_curve.detach().cpu()),
            "loss_deriv": float(l_deriv.detach().cpu()),
            "loss_amp": float(l_amp.detach().cpu()),
            "loss_corr": float(l_corr.detach().cpu()),
        }
        return total, parts
