import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighAttenPreserving(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_g, t_g):
        # --------------------------------------------
        # Shape of s_g : ((s_nl,), (bs, s_ch, s_h, s_w))
        # Shape of t_g : ((t_nl,), (bs, t_ch, t_h, t_w))
        # --------------------------------------------
        s_nl = len(s_g)
        t_nl = len(t_g)

        s_g_mtx = torch.stack([self.get_sim_matrix(s_f) for s_f in s_g])  # (s_nl, bs, bs)
        t_g_mtx = torch.stack([self.get_sim_matrix(t_f) for t_f in t_g])  # (t_nl, bs, bs)

        s_g_mtx = torch.unsqueeze(s_g_mtx, dim=1).repeat(1, t_nl, 1, 1)  # (s_nl, t_nl, bs, bs)
        t_g_mtx = torch.unsqueeze(t_g_mtx, dim=0).repeat(s_nl, 1, 1, 1)  # (s_nl, t_nl, bs, bs)

        loss = (s_g_mtx - t_g_mtx).pow(2).view(s_nl, -1).mean(1).mean(0)  # (1,)

        return loss

    def get_sim_matrix(self, f, is_at=True):
        # --------------------------------------------
        # Shape of f : (bs, ch, h, w)
        # --------------------------------------------
        if is_at:
            f = self.at(f)
        f = f.view(f.shape[0], -1)  # (bs, ch * h * w)
        f = F.normalize(f, dim=1)  # (bs, ch * h * w)
        mtx = torch.matmul(f, torch.t(f))  # (bs, bs)
        mtx = F.normalize(mtx, dim=1)  # (bs, bs)
        return mtx

    def at(self, f, is_flat=True):
        # --------------------------------------------
        # Shape of f : (bs, ch, h, w)
        # --------------------------------------------
        out = f.pow(2).mean(1)  # (bs, h, w)
        if is_flat:
            out =F.normalize(out.view(f.size(0), -1))  # (bs, h * w)
        return out


class HAPPlotter(HighAttenPreserving):
    def __init__(self):
        super().__init__()

    def plot(self, s_g, t_g, input, target, n_samp=10):
        # --------------------------------------------
        # Shape of s_g    : ((s_nl,), (bs, s_ch, s_h, s_w))
        # Shape of t_g    : ((t_nl,), (bs, t_ch, t_h, t_w))
        # Shape of input  : (bs, ch, h, w)
        # Shape of target : (bs,)
        # --------------------------------------------
        s_g = [s_f.detach() for s_f in s_g]
        t_g = [t_f.detach() for t_f in t_g]

        input = input.permute(0, 2, 3, 1)  # (bs, h, w, ch)

        s_attn_g = [self.at(s_f, is_flat=False) for s_f in s_g]  # ((s_nl,), (bs, s_h, s_w))
        t_attn_g = [self.at(t_f, is_flat=False) for t_f in t_g]  # ((t_nl,), (bs, t_h, t_w))

        s_sim_g = torch.stack([self.get_sim_matrix(s_f) for s_f in s_g])  # (s_nl, bs, bs)
        t_sim_g = torch.stack([self.get_sim_matrix(t_f) for t_f in t_g])  # (t_nl, bs, bs)

        s_pos_ind_g = torch.argsort(-s_sim_g, dim=2)  # (s_nl, bs, bs)
        t_pos_ind_g = torch.argsort(-t_sim_g, dim=2)  # (t_nl, bs, bs)
        s_neg_ind_g = torch.argsort(s_sim_g, dim=2)  # (s_nl, bs, bs)
        t_neg_ind_g = torch.argsort(t_sim_g, dim=2)  # (t_nl, bs, bs)

        pivot = 0
        for i, (t_pos_ind_f, t_neg_ind_f, t_attn_f) in enumerate(zip(t_pos_ind_g, t_neg_ind_g, t_attn_g)):
            pos_ind = t_pos_ind_f[pivot, :n_samp]  # (n_samp,)
            neg_ind = t_neg_ind_f[pivot, :n_samp]  # (n_samp,)
            r_pos_imgs = input[pos_ind]  # (n_samp, h, w, ch). Sampled raw images
            r_neg_imgs = input[neg_ind]  # (n_samp, h, w, ch). Sampled raw images
            t_pos_attns = t_attn_f[pos_ind]  # (n_samp, t_h, t_w)
            t_neg_attns = t_attn_f[neg_ind]  # (n_samp, t_h, t_w)
            fig, axs = plt.subplots(4, n_samp)  # 4 rows and n_samp columns
            for j in range(n_samp):
                axs[0, j].imshow(r_pos_imgs[j], interpolation="bicubic")
                axs[1, j].imshow(t_pos_attns[j],  cmap="jet")
                axs[2, j].imshow(r_neg_imgs[j], interpolation="bicubic")
                axs[3, j].imshow(t_neg_attns[j],  cmap="jet")
        plt.show()

