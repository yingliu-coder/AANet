import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.StandardNorm import Normalize


def get_activation(name: str):
    if name.lower() == "gelu":
        return nn.GELU()
    elif name.lower() == "relu":
        return nn.ReLU()
    else:
        return nn.GELU()


class ChannelPatchEmbedding(nn.Module):
    """
    Channel-independent patch embedding.

    Input:
        x: [B, L, D]
    Output:
        H: [B, D, N, d_model]
    """

    def __init__(self, seq_len, patch_len, stride, d_model, dropout=0.1, padding_patch="end"):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        if padding_patch == "end":
            self.patch_num = int((seq_len + stride - patch_len) / stride + 1)
        else:
            self.patch_num = int((seq_len - patch_len) / stride + 1)

        self.value_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, self.patch_num, d_model))
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _get_pos_embed(self, N):
        if N == self.position_embedding.size(2):
            return self.position_embedding

        # interpolate positional embedding if actual patch number changes
        pos = self.position_embedding.squeeze(1)      # [1, N0, d]
        pos = pos.transpose(1, 2)                     # [1, d, N0]
        pos = F.interpolate(pos, size=N, mode="linear", align_corners=False)
        pos = pos.transpose(1, 2).unsqueeze(1)        # [1, 1, N, d]
        return pos

    def forward(self, x):
        B, L, D = x.shape

        # [B, L, D] -> [B, D, L]
        x = x.permute(0, 2, 1).contiguous()

        if L < self.patch_len:
            x = F.pad(x, (0, self.patch_len - L), mode="replicate")

        if self.padding_patch == "end":
            x = F.pad(x, (0, self.stride), mode="replicate")

        # [B, D, N, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # [B, D, N, d_model]
        H = self.value_embedding(x)
        H = H + self._get_pos_embed(H.size(2))
        H = self.dropout(H)

        return H


class TemporalEncoderBlock(nn.Module):
    """
    Lightweight temporal encoder applied independently to each variable.
    Complexity is linear in D because variables are flattened into batch dimension.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x: [B * D, N, d_model]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class VariableClustering(nn.Module):
    """
    Differentiable variable clustering.

    Input:
        H: [B, D, N, d_model]

    Output:
        P: [B, D, K]
    """

    def __init__(self, d_model, cluster_dim, num_clusters, tau_c=1.0):
        super().__init__()

        self.num_clusters = num_clusters
        self.tau_c = tau_c

        self.proj = nn.Linear(d_model, cluster_dim)
        self.centers = nn.Parameter(torch.randn(num_clusters, cluster_dim))

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.centers, std=0.02)

    def forward(self, H):
        # variable descriptor: [B, D, d_model]
        z = H.mean(dim=2)

        # [B, D, d_c]
        z = self.proj(z)

        # squared Euclidean distance: [B, D, K]
        dist = torch.cdist(z, self.centers.unsqueeze(0), p=2) ** 2

        # soft assignment: [B, D, K]
        P = torch.softmax(-dist / self.tau_c, dim=-1)

        return P

    @staticmethod
    def balance_loss(P):
        """
        Cluster balance loss:
            L2 = 1/K sum_k (K * p_bar_k - 1)^2

        P: [B, D, K]
        """
        K = P.size(-1)
        p_bar = P.mean(dim=1)  # [B, K]
        loss = ((K * p_bar - 1.0) ** 2).mean()
        return loss


class EfficientAsynchronousAlignment(nn.Module):
    """
    Cluster-prototype asynchronous alignment.

    This module approximates pair-wise variable interaction by soft cluster prototypes,
    reducing complexity from O(D^2) to O(DK).

    It still follows:
        Q_i = W_Q H_i
        K_j = W_K H_j
        V_j = W_V H_j
        lag-aware score
        adaptive routing
        R_i = sum_j sum_delta w_ij alpha_ij^delta Shift(V_j, delta)

    The difference is that variables j are aggregated into differentiable cluster
    prototypes weighted by P.
    """

    def __init__(
        self,
        d_model,
        qk_dim,
        max_lag,
        tau_r=1.0,
        tau_lag=1.0,
        lambda_cluster=0.5,
        dropout=0.1,
        exclude_self=True
    ):
        super().__init__()

        self.d_model = d_model
        self.qk_dim = qk_dim
        self.max_lag = max_lag
        self.tau_r = tau_r
        self.tau_lag = tau_lag
        self.lambda_cluster = lambda_cluster
        self.exclude_self = exclude_self
        self.eps = 1e-6

        self.q_proj = nn.Linear(d_model, qk_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def shift(x, delta):
        """
        Shift along patch dimension.

        x: [B, C, N, d]
        return:
            shifted x with the same shape
        """
        if delta == 0:
            return x

        y = torch.zeros_like(x)
        N = x.size(-2)

        if abs(delta) >= N:
            return y

        if delta > 0:
            # y[t] = x[t + delta]
            y[..., :N - delta, :] = x[..., delta:, :]
        else:
            # y[t] = x[t + delta]
            delta_abs = -delta
            y[..., delta_abs:, :] = x[..., :N - delta_abs, :]

        return y

    @staticmethod
    def valid_mean(x, delta):
        """
        x: [B, D, K, N]
        """
        N = x.size(-1)

        if abs(delta) >= N:
            return x.new_zeros(x.size(0), x.size(1), x.size(2))

        if delta > 0:
            return x[..., :N - delta].mean(dim=-1)
        elif delta < 0:
            return x[..., -delta:].mean(dim=-1)
        else:
            return x.mean(dim=-1)

    def forward(self, H, P):
        """
        H: [B, D, N, d_model]
        P: [B, D, K]

        Return:
            R:      [B, D, N, d_model]
            route:  [B, D, K]
            alpha:  [B, D, K, S]
        """
        B, D, N, _ = H.shape
        K = P.size(-1)

        q = self.q_proj(H)      # [B, D, N, qk_dim]
        k_var = self.k_proj(H)  # [B, D, N, qk_dim]
        v_var = self.v_proj(H)  # [B, D, N, d_model]

        # unnormalized cluster sums
        # [B, K, N, qk_dim]
        k_sum = torch.einsum("bdk,bdnh->bknh", P, k_var)

        # [B, K, N, d_model]
        v_sum = torch.einsum("bdk,bdnh->bknh", P, v_var)

        # [B, K]
        denom_all = P.sum(dim=1).clamp_min(self.eps)

        if self.exclude_self:
            # [B, D, K]
            denom = (denom_all.unsqueeze(1) - P).clamp_min(self.eps)
        else:
            denom = denom_all.unsqueeze(1).expand(B, D, K).clamp_min(self.eps)

        lags = list(range(-self.max_lag, self.max_lag + 1))
        score_list = []
        shifted_v_sum_list = []
        shifted_v_self_list = []

        scale = math.sqrt(self.qk_dim)

        for delta in lags:
            k_sum_shift = self.shift(k_sum, delta)      # [B, K, N, qk_dim]
            v_sum_shift = self.shift(v_sum, delta)      # [B, K, N, d_model]

            # score between target variable and cluster prototype
            # total score numerator: [B, D, K, N]
            total_score = torch.einsum("bdnh,bknh->bdkn", q, k_sum_shift)

            if self.exclude_self:
                k_self_shift = self.shift(k_var, delta)  # [B, D, N, qk_dim]
                self_score = (q * k_self_shift).sum(dim=-1)  # [B, D, N]

                # remove variable i from its own source cluster
                total_score = total_score - P.unsqueeze(-1) * self_score.unsqueeze(2)

            total_score = total_score / denom.unsqueeze(-1)
            total_score = total_score / scale

            # [B, D, K]
            score_delta = self.valid_mean(total_score, delta)

            score_list.append(score_delta)
            shifted_v_sum_list.append(v_sum_shift)

            if self.exclude_self:
                shifted_v_self_list.append(self.shift(v_var, delta))

        # [B, D, K, S]
        scores = torch.stack(score_list, dim=-1)

        # lag distribution alpha_{i,k}^{Delta}
        alpha = torch.softmax(scores / self.tau_lag, dim=-1)

        # expected lag-aware score
        r_bar = (alpha * scores).sum(dim=-1)  # [B, D, K]

        # cluster prior for target variable i
        # p_{i,k} corresponds to cluster-level similarity between target i and cluster k
        cluster_prior = torch.log(P.clamp_min(self.eps))

        # adaptive routing score
        u = r_bar + self.lambda_cluster * cluster_prior

        # routing weight over K source clusters
        route = torch.softmax(u / self.tau_r, dim=-1)  # [B, D, K]
        route = self.dropout(route)

        # context R_i
        R = H.new_zeros(B, D, N, self.d_model)

        for s, _ in enumerate(lags):
            coeff = route * alpha[..., s]  # [B, D, K]

            # normalize by cluster size after self-exclusion
            norm_coeff = coeff / denom  # [B, D, K]

            # contribution from cluster sums
            # [B, D, N, d_model]
            ctx = torch.einsum("bdk,bknh->bdnh", norm_coeff, shifted_v_sum_list[s])

            if self.exclude_self:
                # remove self contribution exactly
                self_coeff = (norm_coeff * P).sum(dim=-1)  # [B, D]
                ctx = ctx - self_coeff.unsqueeze(-1).unsqueeze(-1) * shifted_v_self_list[s]

            R = R + ctx

        return R, route, alpha, r_bar


class FusionAttention(nn.Module):
    """
    Cross-attention fusion:
        O_i = CrossAttn(H_i, R_i, R_i)
        g_i = sigmoid(W_g [H_i; O_i])
        H_hat_i = LN(H_i + g_i * O_i)
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, R):
        B, D, N, d = H.shape

        H_flat = H.reshape(B * D, N, d)
        R_flat = R.reshape(B * D, N, d)

        O, _ = self.cross_attn(H_flat, R_flat, R_flat, need_weights=False)
        O = O.reshape(B, D, N, d)

        g = torch.sigmoid(self.gate(torch.cat([H, O], dim=-1)))
        H_hat = self.norm(H + self.dropout(g * O))

        return H_hat


class FlattenPredictionHead(nn.Module):
    """
    Shared prediction head for all variables.

    Input:
        H: [B, D, N, d_model]
    Output:
        y: [B, pred_len, D]
    """

    def __init__(self, patch_num, d_model, pred_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(patch_num * d_model, pred_len)

    def forward(self, H):
        B, D, N, d = H.shape

        H = self.dropout(H)
        H = H.reshape(B, D, N * d)

        # [B, D, pred_len]
        y = self.proj(H)

        # [B, pred_len, D]
        y = y.permute(0, 2, 1).contiguous()

        return y


class Model(nn.Module):
    """
    CAformer for long-term / short-term multivariate time series forecasting.

    Main modules:
        1. Channel-independent patch embedding
        2. Temporal encoder
        3. Differentiable variable clustering
        4. Lag-aware asynchronous alignment
        5. Adaptive routing
        6. Fusion attention
        7. Prediction head
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.task_name = configs.task_name

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        dropout = configs.dropout
        activation = configs.activation

        # -------------------------
        # CAformer-specific configs
        # -------------------------
        self.patch_len = getattr(configs, "patch_len", 16)
        self.stride = getattr(configs, "stride", 8)

        self.num_clusters = min(
            getattr(configs, "ca_clusters", 8),
            self.enc_in
        )

        self.cluster_dim = getattr(configs, "ca_cluster_dim", max(16, d_model // 2))
        self.max_lag = getattr(configs, "ca_lag", 3)

        self.qk_dim = getattr(configs, "ca_qk_dim", max(8, d_model // 4))

        self.tau_c = getattr(configs, "tau_c", 1.0)
        self.tau_r = getattr(configs, "tau_r", 1.0)
        self.tau_lag = getattr(configs, "tau_lag", 1.0)

        self.lambda_cluster = getattr(configs, "ca_lambda", 0.5)
        self.exclude_self = getattr(configs, "ca_exclude_self", True)

        # Normalization
        self.normalize_layers = Normalize(
            self.enc_in,
            affine=True,
            non_norm=True if configs.use_norm == 0 else False
        )

        # 1. Multivariate Time Series Embedding
        self.patch_embedding = ChannelPatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=d_model,
            dropout=dropout,
            padding_patch=getattr(configs, "padding_patch", "end")
        )

        patch_num = self.patch_embedding.patch_num

        # 2. Temporal encoder
        self.temporal_encoder = nn.ModuleList([
            TemporalEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
            for _ in range(configs.e_layers)
        ])

        # 3. Variable clustering
        self.variable_clustering = VariableClustering(
            d_model=d_model,
            cluster_dim=self.cluster_dim,
            num_clusters=self.num_clusters,
            tau_c=self.tau_c
        )

        # 4-5. Asynchronous alignment + adaptive routing
        self.async_alignment = EfficientAsynchronousAlignment(
            d_model=d_model,
            qk_dim=self.qk_dim,
            max_lag=self.max_lag,
            tau_r=self.tau_r,
            tau_lag=self.tau_lag,
            lambda_cluster=self.lambda_cluster,
            dropout=dropout,
            exclude_self=self.exclude_self
        )

        # 6. Fusion attention
        self.fusion_attention = FusionAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # 7. Prediction head
        self.prediction_head = FlattenPredictionHead(
            patch_num=patch_num,
            d_model=d_model,
            pred_len=self.pred_len,
            dropout=dropout
        )

        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        self._aux_loss = self._zero
        self.last_routing = None

    def get_aux_loss(self):
        return self._aux_loss

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [B, L, D]
        return:
            dec_out: [B, pred_len, D]
        """
        B, L, D = x_enc.shape

        # Normalize
        x_enc = self.normalize_layers(x_enc, "norm")

        # 1. Patch embedding
        # H: [B, D, N, d_model]
        H = self.patch_embedding(x_enc)

        B, D, N, d = H.shape

        # 2. Temporal encoder independently for each variable
        H_flat = H.reshape(B * D, N, d)

        for block in self.temporal_encoder:
            H_flat = block(H_flat)

        H = H_flat.reshape(B, D, N, d)

        # 3. Differentiable variable clustering
        P = self.variable_clustering(H)

        # cluster regularization loss
        self._aux_loss = self.variable_clustering.balance_loss(P)

        # 4-5. Asynchronous alignment and adaptive routing
        R, route, alpha, r_bar = self.async_alignment(H, P)

        # 6. Fusion attention
        H_hat = self.fusion_attention(H, R)

        # 7. Prediction
        dec_out = self.prediction_head(H_hat)

        # Denormalize
        dec_out = self.normalize_layers(dec_out, "denorm")
        dec_out = dec_out[:, -self.pred_len:, :]

        self.last_routing = {
            "cluster_assignment": P.detach(),
            "routing_weight": route.detach(),
            "lag_weight": alpha.detach(),
            "lag_score": r_bar.detach()
        }

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # During training, return auxiliary clustering loss together.
            # This is more robust for DataParallel than storing loss only as an attribute.
            if self.training:
                return dec_out, self._aux_loss

            if getattr(self.configs, "output_attention", False):
                return dec_out, self.last_routing

            return dec_out

        else:
            raise ValueError("Only forecasting tasks are implemented in CAformer.")
