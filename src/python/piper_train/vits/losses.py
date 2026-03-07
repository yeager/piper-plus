import torch
from torch.nn import functional as F


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g, strict=False):
        for rl, gl in zip(dr, dg, strict=False):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=False):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l_dg = torch.mean((1 - dg) ** 2)
        gen_losses.append(l_dg)
        loss += l_dg

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l_kl = kl / torch.sum(z_mask)
    return l_kl


def speaker_consistency_loss(gen_embedding, ref_embedding):
    """Speaker Consistency Loss (SCL) — コサイン類似度ベースの話者一貫性損失

    Parameters
    ----------
    gen_embedding : torch.Tensor
        生成音声から抽出した話者埋め込み [B, D]
    ref_embedding : torch.Tensor
        参照音声の話者埋め込み [B, D]

    Returns
    -------
    torch.Tensor
        スカラー損失値 (範囲: 0-2, 0が完全一致)
    """
    return 1.0 - F.cosine_similarity(gen_embedding, ref_embedding, dim=-1).mean()


def dino_loss(student_emb, teacher_emb, center, tau_s=0.1, tau_t=0.04):
    """DINO自己蒸留損失 — 話者埋め込み空間の正則化

    Parameters
    ----------
    student_emb : torch.Tensor
        学生ネットワーク出力 [B, D]
    teacher_emb : torch.Tensor
        教師ネットワーク出力 [B, D] (通常は detach 済み)
    center : torch.Tensor
        EMA センター [D]
    tau_s : float
        学生温度パラメータ (default: 0.1)
    tau_t : float
        教師温度パラメータ (default: 0.04)

    Returns
    -------
    torch.Tensor
        スカラー損失値 (正の値)
    """
    student_out = F.log_softmax(student_emb / tau_s, dim=-1)
    teacher_out = F.softmax((teacher_emb - center) / tau_t, dim=-1)
    return -(teacher_out * student_out).sum(dim=-1).mean()
