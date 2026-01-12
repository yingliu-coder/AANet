import torch
import torch.nn as nn
from deepspeed.inference.v2.modules import configs
from fastdtw import fastdtw
from layers.StandardNorm import Normalize
from sklearn.metrics.pairwise import rbf_kernel
from src.Lagformer.Embedding import DataEmbedding, TokenEmbedding, TemporalEmbedding, TimeFeatureEmbedding, PositionalEmbedding
from src.Lagformer.Encoder import Encoder, EncoderLayer
from src.Lagformer.Attention_Family import FullAttention, AttentionLayer
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.cluster import KMeans

class Model(nn.Module):

    def __init__(self, configs, embed_type='fixed'):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.enc_in = configs.enc_in
        self.clusters = configs.clusters
        self.down_window = configs.down_window

        self.down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                  kernel_size=self.down_window, padding=1,
                                  stride=self.down_window, padding_mode='zeros',bias=False)
        self.normalize_layers = Normalize(self.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)

        # Embedding
        self.temporal_embedding = TemporalEmbedding(d_model=self.d_model, embed_type=embed_type,
                                                    freq=configs.freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
                                                    d_model=self.d_model, embed_type=embed_type, freq=configs.freq)
        self.position_embedding = PositionalEmbedding(d_model=self.d_model, max_len=self.seq_len * self.enc_in)


        self.patch_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model,
                  kernel_size=self.patch_len, padding=1, stride=self.stride,
                  padding_mode='zeros', bias=False)

        self.patch_recover = nn.ConvTranspose1d(in_channels=self.d_model, out_channels=self.d_model,
                           kernel_size=self.patch_len, padding=1, stride=self.stride,
                           output_padding=2, padding_mode='zeros', bias=False)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for i in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.predict_layers = torch.nn.Linear(self.seq_len, self.pred_len)
        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

    def SpectralCluster(self, x_enc):
        B, L, K = x_enc.shape

        x_enc = x_enc.permute(0, 2, 1).detach().cpu().numpy()

        label_batch = []
        for index in range(B):

            x_var = x_enc[index, :, :]
            dtw_matrix = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    if i <= j:
                        distance, _ = fastdtw(x_var[i], x_var[j])
                        dtw_matrix[i, j] = distance
                        dtw_matrix[j, i] = dtw_matrix[i, j]

            similarity_matrix = rbf_kernel(-dtw_matrix)

            sc = SpectralClustering(
                n_clusters=self.clusters,
                affinity='precomputed',
                assign_labels='kmeans'
            )

            labels = sc.fit_predict(similarity_matrix)
            label_batch.append(labels)

        return label_batch


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, T_enc, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_down = self.down_pool(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        label_batch = self.SpectralCluster(x_down)

        x_out = []
        for b in range(B):
            x_b = x_enc[b].reshape(1, T_enc, N)
            x_b = x_b.permute(0, 2, 1).contiguous().reshape(N, T_enc, 1)
            if x_mark_enc is not None:
                x_mark_b = x_mark_enc[b].reshape(1, T_enc, -1).repeat(N, 1, 1)
            else:
                x_mark_b = None

            x_emb_b = x_b + self.temporal_embedding(x_mark_b)
            x_emb_b = x_emb_b.reshape(1, -1, self.d_model)
            x_emb_b = x_emb_b + self.position_embedding(x_emb_b)
            x_emb_b = x_emb_b.reshape(N, -1, self.d_model)
            x_pat_b = self.patch_conv(x_emb_b.permute(0, 2, 1)).permute(0, 2, 1)

            cluster_label = label_batch[b]
            cluster_unique = np.sort(np.unique(cluster_label))
            cluster_num = len(cluster_unique)
            for u in range(cluster_num):
                num = cluster_unique[u]
                x_bc = []
                index = []
                for i in range(len(cluster_label)):
                    if cluster_label[i] == num:
                        x_bc.append(x_pat_b[i, :, :])
                        index.append(i)
                x_bc = torch.stack(x_bc, dim=0)
                x_bc = x_bc.reshape(1, -1, self.d_model).contiguous()
                x_bc = self.encoder(x_bc)
                x_bc = x_bc.reshape(len(index), -1, self.d_model).contiguous()
                for i in range(len(index)):
                    x_pat_b[index[i]] += x_bc[i]
            x_dec = self.patch_recover(x_pat_b.permute(0, 2, 1)).permute(0, 2, 1)
            x_out.append(x_dec)

        x_out = torch.cat(x_out, dim=0)
        x_out = self.predict_layers(x_out.permute(0, 2, 1)).permute(0, 2, 1)
        x_out = self.projection_layer(x_out)
        x_out = x_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        x_out = self.normalize_layers(x_out, 'denorm')
        x_out = x_out[:, -self.pred_len:, :]
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return x_out