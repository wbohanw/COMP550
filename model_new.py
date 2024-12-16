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
        

        self.multihead_attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_attention_heads)

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
            entity_embs = []
            entity_weights = []

            for e in entity_pos[i]:
                e_emb, e_weight = [], []
                for start, end in e:
                    if start + offset < c:
                        e_emb.append(sequence_output[i, start + offset])
                        e_weight.append(attention[i, :, start + offset])
                if len(e_emb) > 0:
                    e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                    e_weight = torch.stack(e_weight, dim=0).mean(0)
                else:
                    e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    e_weight = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_weights.append(e_weight)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_weights = torch.stack(entity_weights, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_weight = torch.index_select(entity_weights, 0, ht_i[:, 0])
            t_weight = torch.index_select(entity_weights, 0, ht_i[:, 1])
            

            ht_att = (h_weight * t_weight).sum(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)


            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            rs, _ = self.multihead_attention(rs.unsqueeze(0), rs.unsqueeze(0), rs.unsqueeze(0))
            rs = rs.squeeze(0)

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

        sequence_output, attention = self.encode(input_ids, attention_mask)
        

        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        if return_contrastive_features:

            is_first_class = labels[:, 0] == 1
            is_rest_zero = labels[:, 1:].sum(dim=1) == 0
            neg_mask = is_first_class & is_rest_zero
            pos_mask = ~neg_mask


            hs_pos, rs_pos, ts_pos = hs[pos_mask], rs[pos_mask], ts[pos_mask]
            hs_neg, rs_neg, ts_neg = hs[neg_mask], rs[neg_mask], ts[neg_mask]


            hs_pos_proj = self.projection_layer(torch.cat([hs_pos, rs_pos], dim=1))
            ts_pos_proj = self.projection_layer(torch.cat([ts_pos, rs_pos], dim=1))
            hs_neg_proj = self.projection_layer(torch.cat([hs_neg, rs_neg], dim=1))
            ts_neg_proj = self.projection_layer(torch.cat([ts_neg, rs_neg], dim=1))


            hs_pos_proj = F.normalize(hs_pos_proj, p=2, dim=-1)
            ts_pos_proj = F.normalize(ts_pos_proj, p=2, dim=-1)
            hs_neg_proj = F.normalize(hs_neg_proj, p=2, dim=-1)
            ts_neg_proj = F.normalize(ts_neg_proj, p=2, dim=-1)


        hs_proj = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts_proj = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))

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
