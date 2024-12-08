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
                 num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.projection_layer = nn.Sequential(
            nn.Linear(2 * config.hidden_size, emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size)
        )

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

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    # def forward(self,
    #             input_ids=None,
    #             attention_mask=None,
    #             entity_pos=None,
    #             hts=None,
    #             return_contrastive_features=False):
    #     sequence_output, attention = self.encode(input_ids, attention_mask)
    #     hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
    #     hs_proj = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
    #     ts_proj = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
    #     b1 = hs_proj.view(-1, self.emb_size // self.block_size, self.block_size)
    #     b2 = ts_proj.view(-1, self.emb_size // self.block_size, self.block_size)
    #     bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
    #     logits = self.bilinear(bl)

    #     if return_contrastive_features:
    #         return logits, hs, rs, ts  # 返回 logits 和局部特征
    #     return logits

    def forward(self,
                input_ids=None,
                attention_mask=None,
                entity_pos=None,
                hts=None,
                labels=None,  # 用于区分正负样本
                return_contrastive_features=False):
        # 编码输入序列
        sequence_output, attention = self.encode(input_ids, attention_mask)
        
        # 提取局部特征
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        
        if return_contrastive_features:
            # 区分正负样本
        # 判断负样本和正样本
            is_first_class = labels[:, 0] == 1
            is_rest_zero = labels[:, 1:].sum(dim=1) == 0

            neg_mask = is_first_class & is_rest_zero  # 负样本掩码
            pos_mask = ~neg_mask                      # 正样本掩码
                    # 负样本掩码，无关系
            
            # 分别提取正负样本特征
            hs_pos, rs_pos, ts_pos = hs[pos_mask], rs[pos_mask], ts[pos_mask]
            hs_neg, rs_neg, ts_neg = hs[neg_mask], rs[neg_mask], ts[neg_mask]
            
            # 投影到对比学习空间
            hs_pos_proj = self.projection_layer(torch.cat([hs_pos, rs_pos], dim=1))
            ts_pos_proj = self.projection_layer(torch.cat([ts_pos, rs_pos], dim=1))
            hs_neg_proj = self.projection_layer(torch.cat([hs_neg, rs_neg], dim=1))
            ts_neg_proj = self.projection_layer(torch.cat([ts_neg, rs_neg], dim=1))
            
            # 归一化特征
            hs_pos_proj = F.normalize(hs_pos_proj, p=2, dim=-1)
            ts_pos_proj = F.normalize(ts_pos_proj, p=2, dim=-1)
            hs_neg_proj = F.normalize(hs_neg_proj, p=2, dim=-1)
            ts_neg_proj = F.normalize(ts_neg_proj, p=2, dim=-1)
            
            # 返回 logits 和对比学习特征
        
        # 投影到分类空间
        hs_proj = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts_proj = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        
        # 计算 logits（用于主任务分类）
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

