import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
import torch.nn.functional as F

class DocREModel(nn.Module):
    def __init__(self,
                 config,
                 model,
                 emb_size=768,
                 block_size=64,
                 num_labels=-1,
                 num_attention_heads=8):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        # 双向GRU用于特征聚合
        self.entity_encoder = nn.GRU(input_size=self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     bidirectional=True,
                                     batch_first=True)

        # 双线性注意力机制
        self.bilinear_attention = nn.Bilinear(emb_size, emb_size, num_labels)

        # 投影层
        self.projection_layer = nn.Sequential(
            nn.Linear(2 * config.hidden_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # 分类器
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_entity_embeddings(self, sequence_output, entity_pos):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        entity_embs = []

        for i, entity_list in enumerate(entity_pos):
            batch_entity_embs = []
            for e in entity_list:
                e_embs = []
                for start, end in e:
                    if start + offset < sequence_output.size(1):
                        e_embs.append(sequence_output[i, start + offset])
                if len(e_embs) > 0:
                    e_embs = torch.stack(e_embs, dim=0)  # [entity_length, hidden_size]
                    # 使用双向GRU聚合特征
                    e_embs, _ = self.entity_encoder(e_embs.unsqueeze(0))  # [1, entity_length, 2*hidden_size]
                    e_embs = e_embs.squeeze(0).mean(0)  # 聚合为 [2*hidden_size]
                else:
                    e_embs = torch.zeros(2 * self.hidden_size).to(sequence_output.device)
                batch_entity_embs.append(e_embs)
            entity_embs.append(torch.stack(batch_entity_embs, dim=0))  # [n_entities, 2*hidden_size]
        return entity_embs

    def get_hrt(self, sequence_output, entity_pos, hts):
        entity_embs = self.get_entity_embeddings(sequence_output, entity_pos)
        hss, tss, rss = [], [], []

        for i, ht_pairs in enumerate(hts):
            ht_pairs = torch.LongTensor(ht_pairs).to(sequence_output.device)
            hs = torch.index_select(entity_embs[i], 0, ht_pairs[:, 0])  # [n_pairs, 2*hidden_size]
            ts = torch.index_select(entity_embs[i], 0, ht_pairs[:, 1])  # [n_pairs, 2*hidden_size]

            # 计算关系嵌入
            rs = torch.stack([sequence_output[i].mean(dim=0) for _ in range(len(ht_pairs))])  # 平均池化
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                entity_pos=None,
                hts=None,
                labels=None,
                return_contrastive_features=False):
        # 编码输入序列
        sequence_output, attention = self.encode(input_ids, attention_mask)

        # 获取实体头、尾和关系嵌入
        hs, rs, ts = self.get_hrt(sequence_output, entity_pos, hts)

        if return_contrastive_features:
            # 正负样本掩码
            is_first_class = labels[:, 0] == 1
            is_rest_zero = labels[:, 1:].sum(dim=1) == 0
            neg_mask = is_first_class & is_rest_zero
            pos_mask = ~neg_mask

            # 提取正负样本
            hs_pos, rs_pos, ts_pos = hs[pos_mask], rs[pos_mask], ts[pos_mask]
            hs_neg, rs_neg, ts_neg = hs[neg_mask], rs[neg_mask], ts[neg_mask]

            # 投影
            hs_pos_proj = self.projection_layer(torch.cat([hs_pos, rs_pos], dim=1))
            ts_pos_proj = self.projection_layer(torch.cat([ts_pos, rs_pos], dim=1))
            hs_neg_proj = self.projection_layer(torch.cat([hs_neg, rs_neg], dim=1))
            ts_neg_proj = self.projection_layer(torch.cat([ts_neg, rs_neg], dim=1))

            # 归一化
            hs_pos_proj = F.normalize(hs_pos_proj, p=2, dim=-1)
            ts_pos_proj = F.normalize(ts_pos_proj, p=2, dim=-1)
            hs_neg_proj = F.normalize(hs_neg_proj, p=2, dim=-1)
            ts_neg_proj = F.normalize(ts_neg_proj, p=2, dim=-1)

        # 分类任务
        hs_proj = torch.tanh(self.projection_layer(hs))
        ts_proj = torch.tanh(self.projection_layer(ts))

        b1 = hs_proj.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts_proj.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        if return_contrastive_features:
            return logits, {
                "positive": {"head": hs_pos_proj, "tail": ts_pos_proj, "relation": rs_pos},
                "negative": {"head": hs_neg_proj, "tail": ts_neg_proj, "relation": rs_neg}
            }
        else:
            return logits
