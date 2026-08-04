import unittest

import torch 
import torch.nn as nn

# from https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        # average the attention scores over the multiple heads, otherwise return per-head attention score
        kwargs["average_attn_weights"] = True 

        return forward_orig(*args, **kwargs)

    m.forward = wrap

activation, handles = dict(), dict()

def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output[1].clone().detach()
    return hook

class TestForwardHook(unittest.TestCase):
    
    def test_mask_forward(self):
        
        batch_size, len_q, len_k, embed_dim = 3, 4, 5, 6

        Q: torch.Tensor = torch.rand(batch_size, len_q, embed_dim, requires_grad=True)
        K: torch.Tensor = torch.rand(batch_size, len_k, embed_dim)
        V: torch.Tensor = torch.rand(batch_size, len_k, embed_dim)
        causal_mask = torch.ones(len_q, len_k, dtype=torch.bool).tril(diagonal=0)

        # y = F.scaled_dot_product_attention(Q, K, V, dropout_p=0, attn_mask=causal_mask)

        # in multihead attention, a True indicates "not allowed" in attention
        mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=0, batch_first=True)
        y, attn_weights = mha(Q, K, V, attn_mask=torch.logical_not(causal_mask), need_weights=True)

        tf_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=1, dim_feedforward=32, dropout=0, batch_first=True)
            
        patch_attention(tf_encoder.self_attn)
        handles['attn'] = tf_encoder.self_attn.register_forward_hook(getActivation('attn'))

        y2 = tf_encoder(Q, torch.logical_not(torch.ones(len_q, len_q, dtype=torch.bool).tril(diagonal=0)))
        
        self.assertIn('attn', activation.keys())
        
        for sample in range(batch_size):
            sample_attention = activation['attn'][sample]
            zeros_row_cols = torch.triu_indices(len_q, len_q, offset=1)
            self.assertTrue(sum(sample_attention[zeros_row_cols[0], zeros_row_cols[1]])==0)
            
        for name, h in handles.items():
            h.remove()
        activation.clear()
        
if __name__ == '__main__':
    unittest.main()