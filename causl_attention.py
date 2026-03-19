import torch
import torch.nn as nn
from typing import Optional


# class CausalUnifiedAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim, num_heads, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_proj = nn.Linear(dim_q, dim, bias=False)
#         self.k_proj = nn.Linear(dim_kv, dim, bias=False)
#         self.v_proj = nn.Linear(dim_kv, dim, bias=False)
#         self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
#
#         self.q_norm = norm_layer(self.head_dim)
#         self.k_norm = norm_layer(self.head_dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#     def _forward_sdpa(self, q, k, v, attn_mask=None):
#         N, H, Lq, _ = q.shape
#         _, _, Lk, _ = k.shape
#
#         q = q * self.scale
#         k = k.transpose(-2, -1)
#
#         if attn_mask is not None:
#             attn_mask = attn_mask.unsqueeze(1)  # (N, 1, Lq, Lk)
#             attn_mask = attn_mask.expand(-1, H, -1, -1)  # only expand across heads
#             attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill(attn_mask, float("-inf"))
#             attn = torch.matmul(q, k) + attn_mask
#         else:
#             attn = torch.matmul(q, k)
#
#         attn = torch.softmax(attn, dim=-1)
#         x = torch.matmul(attn, v)  # (N, H, Lq, D)
#         return x.transpose(1, 2).reshape(N, Lq, -1)
#
#     def _forward_kv_cache(self, x, layer_index, kv_cache, attn_mask=None, update_kv_cache=False):
#         N, Lq = x.shape[:2]
#         qkv = self.qkv_proj(x).reshape(N, Lq, 3, self.num_heads, self.head_dim)
#         q, curr_k, curr_v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
#         q = self.q_norm(q)
#         curr_k = self.k_norm(curr_k)
#
#         if kv_cache[layer_index]["k"] is not None:
#             k = torch.cat((kv_cache[layer_index]["k"], curr_k), dim=2)
#             v = torch.cat((kv_cache[layer_index]["v"], curr_v), dim=2)
#         else:
#             k = curr_k
#             v = curr_v
#
#         if update_kv_cache:
#             kv_cache[layer_index]["k"] = k
#             kv_cache[layer_index]["v"] = v
#
#         out = self._forward_sdpa(q, k, v, attn_mask)
#         return self.out_proj(out)
#
#     def forward_with_cache(
#         self,
#         text_feat, scene_feat, obj_to_frame, frame_anchor,
#         kv_cache: dict, layer_index: int,
#         update_kv_cache: bool = True,
#         mode: str = "predict"
#     ):
#         B, Lq, _ = text_feat.shape
#         _, Lk, _ = scene_feat.shape
#
#         # project & concat
#         text_proj = self.q_proj(text_feat)
#         scene_proj = self.k_proj(scene_feat)
#         x = torch.cat([text_proj, scene_proj], dim=1)  # (B, Lq+Lk, D)
#
#         obj_to_frame_full = torch.cat([
#             torch.full((Lq,), frame_anchor, device=text_feat.device),
#             obj_to_frame
#         ], dim=0)  # (Lq + Lk)
#
#         if update_kv_cache:
#             kv_cache[layer_index]["obj_to_frame"] = obj_to_frame_full
#         else:
#             obj_to_frame_full = kv_cache[layer_index]["obj_to_frame"]
#
#         # === 构造因果 Mask ===
#         q_frame = obj_to_frame_full.unsqueeze(0).unsqueeze(-1)  # (1, Lq+Lk, 1)
#         k_frame = obj_to_frame_full.unsqueeze(0).unsqueeze(1)   # (1, 1, Lq+Lk)
#
#         if mode == "predict":
#             mask = q_frame >= k_frame
#         elif mode == "retrospect":
#             mask = q_frame <= k_frame
#         else:
#             raise ValueError("Mode must be 'predict' or 'retrospect'")
#
#         output = self._forward_kv_cache(
#             x,
#             layer_index=layer_index,
#             kv_cache=kv_cache,
#             attn_mask=mask,
#             update_kv_cache=update_kv_cache
#         )
#
#         return output[:, :Lq], output[:, Lq:]
#
#
# # ==== ✅ DEMO ====
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     B, Lq, Lk = 1, 6, 10
#     D_q, D_kv, D = 1024, 64, 512
#     heads = 8
#
#     text_feat = torch.randn(B, Lq, D_q).cuda()
#     scene_feat = torch.randn(B, Lk, D_kv).cuda()
#     obj_to_frame = torch.tensor([0, 0, 1, 1, 2, 3, 4, 4, 5, 6], device='cuda')
#
#     model = CausalUnifiedAttention(D_q, D_kv, D, heads).cuda()
#     kv_cache = {0: {"k": None, "v": None, "obj_to_frame": None}}
#
#     text_out, scene_out = model.forward_with_cache(
#         text_feat=text_feat,
#         scene_feat=scene_feat,
#         obj_to_frame=obj_to_frame,
#         frame_anchor=2,
#         kv_cache=kv_cache,
#         layer_index=0,
#         update_kv_cache=True,
#         mode="predict"
#     )
#
#     print("✅ Text out shape:", text_out.shape)   # (1, 6, 512)
#     print("✅ Scene out shape:", scene_out.shape) # (1, 10, 512)






# import torch
# import torch.nn as nn
# from typing import Optional
#
#
# class GeneralizedCausalUnifiedAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim, num_heads, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_proj = nn.Linear(dim_q, dim, bias=False)
#         self.k_proj = nn.Linear(dim_kv, dim, bias=False)
#         self.v_proj = nn.Linear(dim_kv, dim, bias=False)
#         self.qkv = nn.Linear(dim, 3 * dim, bias=False)
#
#         self.q_norm = norm_layer(self.head_dim)
#         self.k_norm = norm_layer(self.head_dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#     def _forward_sdpa(self, q, k, v, attn_mask=None):
#         N, H, Lq, _ = q.shape
#         q = q * self.scale
#         k = k.transpose(-2, -1)  # (N, H, D, Lk)
#
#         if attn_mask is not None:
#             attn_mask = attn_mask.to(dtype=torch.float, device=q.device)
#             attn_mask = torch.zeros_like(attn_mask).masked_fill(attn_mask.bool(), float("-inf"))
#             attn = torch.matmul(q, k) + attn_mask
#         else:
#             attn = torch.matmul(q, k)
#
#         attn = torch.softmax(attn, dim=-1)
#         x = torch.matmul(attn, v)
#         x = x.transpose(1, 2).reshape(N, Lq, -1)
#         return self.out_proj(x)
#
#     def _forward_kv_cache(self, x, layer_index, kv_cache, update_kv_cache=False, attn_mask=None):
#         N, Lq = x.shape[:2]
#         qkv = self.qkv(x).reshape(N, Lq, 3, self.num_heads, self.head_dim)
#         q, curr_k, curr_v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
#         q = self.q_norm(q)
#         curr_k = self.k_norm(curr_k)
#
#         if kv_cache[layer_index]["k"] is not None:
#             k = torch.cat((kv_cache[layer_index]["k"], curr_k), dim=2)
#             v = torch.cat((kv_cache[layer_index]["v"], curr_v), dim=2)
#         else:
#             k = curr_k
#             v = curr_v
#
#         if update_kv_cache:
#             kv_cache[layer_index]["k"] = k
#             kv_cache[layer_index]["v"] = v
#
#         return self._forward_sdpa(q, k, v, attn_mask)
#
#     def forward_with_cache(self, text_feat, scene_feat, obj_to_frame, frame_anchor,
#                            kv_cache, layer_index, update_kv_cache=True, step=4):
#         B, Lq, _ = text_feat.shape
#         _, Lk, _ = scene_feat.shape
#
#         # Step 1: unify dimensions
#         text_proj = self.q_proj(text_feat)
#         scene_k = self.k_proj(scene_feat)
#
#         # Step 2: concat
#         x = torch.cat([text_proj, scene_k], dim=1)  # (B, Lq+Lk, dim)
#         full_frame = torch.cat([
#             torch.full((Lq,), frame_anchor, device=text_feat.device),
#             obj_to_frame
#         ])
#
#         # Step 3: build generalized causal mask
#         seq_len = Lq + Lk
#         mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=text_feat.device)
#
#         def generalized_mask(ctx_len, x_len, step):
#             import random
#             sumk = [0] + sorted(random.sample(range(1, x_len), step - 1)) + [x_len]
#             mask = torch.ones(ctx_len + x_len, ctx_len + x_len, dtype=torch.bool)
#             mask[:ctx_len, :ctx_len] = 0
#             mask[ctx_len:, :ctx_len] = 0
#             for i in range(len(sumk) - 1):
#                 s = ctx_len + sumk[i]
#                 e = ctx_len + sumk[i + 1]
#                 mask[s:e, s:e] = torch.tril(torch.ones(e - s, e - s)).logical_not()
#                 mask[s:e, ctx_len:ctx_len + sumk[i]] = 1
#             return mask
#
#         attn_mask = generalized_mask(Lq, Lk, step)
#         attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, seq_len, seq_len)
#
#         # Step 4: forward attention with KV cache
#         output = self._forward_kv_cache(
#             x,
#             layer_index=layer_index,
#             kv_cache=kv_cache,
#             update_kv_cache=update_kv_cache,
#             attn_mask=attn_mask
#         )
#
#         return output[:, :Lq], output[:, Lq:]




# if __name__ == "__main__":
#     torch.manual_seed(0)
#     B, Lq, Lk = 1, 6, 10
#     Dq, Dkv, D = 1024, 64, 512
#     heads = 8
#
#     text_feat = torch.randn(B, Lq, Dq).cuda()
#     scene_feat = torch.randn(B, Lk, Dkv).cuda()
#     obj_to_frame = torch.tensor([0, 0, 1, 1, 2, 3, 4, 4, 5, 5], device='cuda')
#
#     model = GeneralizedCausalUnifiedAttention(Dq, Dkv, D, heads).cuda()
#     kv_cache = {0: {"k": None, "v": None}}
#
#     text_out, scene_out = model.forward_with_cache(
#         text_feat=text_feat,
#         scene_feat=scene_feat,
#         obj_to_frame=obj_to_frame,
#         frame_anchor=2,
#         kv_cache=kv_cache,
#         layer_index=0,
#         update_kv_cache=True,
#         step=4
#     )
#
#     print("✅ Text Output:", text_out.shape)   # [1, 6, 512]
#     print("✅ Scene Output:", scene_out.shape) # [1, 10, 512]



















# class CausalUnifiedAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim, num_heads, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_linear = nn.Linear(dim_q, dim, bias=False)
#         self.k_linear = nn.Linear(dim_kv, dim, bias=False)
#         self.v_linear = nn.Linear(dim_kv, dim, bias=False)
#         self.qkv_linear = nn.Linear(dim, 3 * dim, bias=False)
#
#         self.q_norm = norm_layer(self.head_dim)
#         self.k_norm = norm_layer(self.head_dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#     def _forward_sdpa(self, q, k, v, attn_mask=None):
#         B, H, Lq, _ = q.shape
#         q = q * self.scale
#         k = k.transpose(-2, -1)  # (B, H, D, Lk)
#
#         if attn_mask is not None:
#             attn_mask = attn_mask.unsqueeze(1)  # (B, 1, Lq, Lk)
#             attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill(attn_mask, float("-inf"))
#             attn = q @ k + attn_mask
#         else:
#             attn = q @ k
#         attn = attn.masked_fill(torch.isnan(attn), float("-1e9"))  # 避免 NaN
#         attn = torch.softmax(attn, dim=-1)
#         x = attn @ v  # (B, H, Lq, D)
#         x = x.transpose(1, 2).reshape(B, Lq, -1)
#         return self.out_proj(x)
#
#     def _forward_kv_cache(self, x, layer_index, kv_cache, update_kv_cache, attn_mask=None):
#         B, Lq = x.shape[:2]
#         qkv = self.qkv_linear(x).reshape(B, Lq, 3, self.num_heads, self.head_dim)
#         q, curr_k, curr_v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
#         q = self.q_norm(q)
#         curr_k = self.k_norm(curr_k)
#
#         if kv_cache[layer_index]["k"] is not None:
#             k = torch.cat([kv_cache[layer_index]["k"], curr_k], dim=2)
#             v = torch.cat([kv_cache[layer_index]["v"], curr_v], dim=2)
#         else:
#             k = curr_k
#             v = curr_v
#
#         if update_kv_cache:
#             kv_cache[layer_index]["k"] = k
#             kv_cache[layer_index]["v"] = v
#
#         return self._forward_sdpa(q, k, v, attn_mask)
#
#     def forward_with_cache(self,
#                            text_feat,         # (B, Lq, Dq)
#                            scene_feat,        # (B, Lk, Dkv)
#                            obj_to_frame,      # (Lk,)
#                            frame_anchor: int,
#                            kv_cache: dict,
#                            layer_index: int,
#                            update_kv_cache: bool = True,
#                            mode: str = "predict"):
#         B, Lq, _ = text_feat.shape
#         _, Lk, _ = scene_feat.shape
#
#         # Linear projections
#         text_proj = self.q_linear(text_feat)
#         scene_proj_k = self.k_linear(scene_feat)
#
#         # unified input
#         x = torch.cat([text_proj, scene_proj_k], dim=1)  # (B, Lq + Lk, D)
#
#         # 构造 obj_to_frame 全部向量（Lq个文本特征视作 frame_anchor）
#         obj_to_frame_full = torch.cat([
#             torch.full((Lq,), frame_anchor, device=text_feat.device),
#             obj_to_frame
#         ], dim=0)  # (Lq + Lk, )
#
#         # === 构造 attention mask (B, Lq+Lk, Lq+Lk) ===
#         q_frame = obj_to_frame_full.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
#         k_frame = obj_to_frame_full.unsqueeze(0).unsqueeze(1)   # (1, 1, L)
#
#         if mode == "predict":
#             attn_mask = q_frame >= k_frame
#         elif mode == "retrospect":
#             attn_mask = q_frame <= k_frame
#         else:
#             raise ValueError("mode must be 'predict' or 'retrospect'")
#
#         if update_kv_cache:
#             kv_cache[layer_index]["obj_to_frame"] = obj_to_frame_full
#         else:
#             obj_to_frame_full = kv_cache[layer_index]["obj_to_frame"]
#
#         # === forward self-attention ===
#         output = self._forward_kv_cache(
#             x, layer_index=layer_index,
#             kv_cache=kv_cache,
#             update_kv_cache=update_kv_cache,
#             attn_mask=attn_mask
#         )
#
#         # 拆分 text 和 scene 输出
#         text_out = output[:, :Lq]
#         scene_out = output[:, Lq:]
#         return text_out, scene_out
#
#
#
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     B, Lq, Lk = 1, 6, 10
#     Dq, Dkv, D = 1024, 64, 512
#     heads = 8
#
#     text_feat = torch.randn(B, Lq, Dq).cuda()
#     scene_feat = torch.randn(B, Lk, Dkv).cuda()
#     obj_to_frame = torch.tensor([0, 0, 1, 1, 2, 3, 4, 4, 5, 5], device='cuda')
#
#     model = CausalUnifiedAttention(Dq, Dkv, D, heads).cuda()
#     kv_cache = {0: {"k": None, "v": None, "obj_to_frame": None}}
#
#     text_out, scene_out = model.forward_with_cache(
#         text_feat=text_feat,
#         scene_feat=scene_feat,
#         obj_to_frame=obj_to_frame,
#         frame_anchor=2,
#         kv_cache=kv_cache,
#         layer_index=0,
#         update_kv_cache=True,
#         mode="predict"  # or "retrospect"
#     )
#
#     print("✅ Text output shape:", text_out.shape)
#     print("✅ Scene output shape:", scene_out.shape)

















# class CausalUnifiedAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim, num_heads, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_linear = nn.Linear(dim_q, dim, bias=False)
#         self.k_linear = nn.Linear(dim_kv, dim, bias=False)
#         self.v_linear = nn.Linear(dim_kv, dim, bias=False)
#         self.qkv_linear = nn.Linear(dim, 3 * dim, bias=False)
#
#         self.q_norm = norm_layer(self.head_dim)
#         self.k_norm = norm_layer(self.head_dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#     def _forward_sdpa(self, q, k, v, attn_mask=None):
#         B, H, Lq, _ = q.shape
#         q = q * self.scale
#         k = k.transpose(-2, -1)  # (B, H, D, Lk)
#
#         attn = torch.matmul(q, k)
#
#         if attn_mask is not None:
#             # 注意这里是布尔类型：True=mask掉
#             attn = attn.masked_fill(attn_mask.unsqueeze(1), float("-1e9"))  # 更稳定
#
#         attn = attn.masked_fill(torch.isnan(attn), float("-1e9"))  # 保底修复 NaN
#         attn = torch.softmax(attn, dim=-1)
#
#         x = torch.matmul(attn, v)
#         x = x.transpose(1, 2).reshape(B, Lq, -1)
#         return self.out_proj(x)
#
#     def _forward_kv_cache(self, x, layer_index, kv_cache, update_kv_cache, attn_mask=None):
#         B, Lq = x.shape[:2]
#         qkv = self.qkv_linear(x).reshape(B, Lq, 3, self.num_heads, self.head_dim)
#         q, curr_k, curr_v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
#         q = self.q_norm(q)
#         curr_k = self.k_norm(curr_k)
#
#         if kv_cache[layer_index]["k"] is not None:
#             k = torch.cat([kv_cache[layer_index]["k"], curr_k], dim=2)
#             v = torch.cat([kv_cache[layer_index]["v"], curr_v], dim=2)
#         else:
#             k = curr_k
#             v = curr_v
#
#         if update_kv_cache:
#             kv_cache[layer_index]["k"] = k
#             kv_cache[layer_index]["v"] = v
#
#         return self._forward_sdpa(q, k, v, attn_mask)
#
#     def forward_with_cache(
#         self,
#         text_feat,          # (B, Lq, Dq)
#         scene_feat,         # (B, Lk, Dkv)
#         obj_to_frame,       # (Lk,)
#         frame_anchor: int,
#         kv_cache: dict,
#         layer_index: int,
#         update_kv_cache: bool = True,
#         mode: str = "predict"
#     ):
#         B, Lq, _ = text_feat.shape
#         _, Lk, _ = scene_feat.shape
#
#         # Projection
#         text_proj = self.q_linear(text_feat)
#         scene_proj_k = self.k_linear(scene_feat)
#
#         x = torch.cat([text_proj, scene_proj_k], dim=1)  # (B, Lq + Lk, D)
#
#         # Construct obj_to_frame
#         obj_to_frame_full = torch.cat([
#             torch.full((Lq,), frame_anchor, device=text_feat.device),
#             obj_to_frame
#         ], dim=0)
#
#         if update_kv_cache:
#             kv_cache[layer_index]["obj_to_frame"] = obj_to_frame_full
#         else:
#             obj_to_frame_full = kv_cache[layer_index]["obj_to_frame"]
#
#         q_frame = obj_to_frame_full.view(1, -1, 1)  # (1, L, 1)
#         k_frame = obj_to_frame_full.view(1, 1, -1)  # (1, 1, L)
#
#         if mode == "predict":
#             attn_mask = q_frame >= k_frame
#         elif mode == "retrospect":
#             attn_mask = q_frame <= k_frame
#         else:
#             raise ValueError("mode must be 'predict' or 'retrospect'")
#
#         output = self._forward_kv_cache(
#             x, layer_index=layer_index, kv_cache=kv_cache,
#             update_kv_cache=update_kv_cache, attn_mask=attn_mask
#         )
#
#         text_out = output[:, :Lq]
#         scene_out = output[:, Lq:]
#         return text_out, scene_out
#
#
#
#
# if __name__ == "__main__":
#     torch.manual_seed(42)
#
#     B, Lq, Lk = 1, 77, 33
#     Dq, Dkv, D = 1024, 64, 512
#     heads = 8
#
#     text_feat = torch.randn(B, Lq, Dq).cuda()
#     scene_feat = torch.randn(B, Lk, Dkv).cuda()
#     obj_to_frame = torch.tensor([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1, 1, 2, 3, 4, 4, 5, 5,6,7,8,9,10,11,12,13,14,15], device='cuda')
#
#     model = CausalUnifiedAttention(Dq, Dkv, D, heads).cuda()
#     kv_cache = {0: {"k": None, "v": None, "obj_to_frame": None}}
#
#     text_out, scene_out = model.forward_with_cache(
#         text_feat=text_feat,
#         scene_feat=scene_feat,
#         obj_to_frame=obj_to_frame,
#         frame_anchor=2,
#         kv_cache=kv_cache,
#         layer_index=0,
#         update_kv_cache=True,
#         mode="predict"  # or "retrospect"
#     )
#
#     print("✅ Text output shape:", text_out.shape)
#     print("✅ Scene output shape:", scene_out.shape)


# class CausalUnifiedAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim, num_heads, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_linear = nn.Linear(dim_q, dim, bias=False)
#         self.k_linear = nn.Linear(dim_kv, dim, bias=False)
#         self.v_linear = nn.Linear(dim_kv, dim, bias=False)
#         self.qkv_linear = nn.Linear(dim, 3 * dim, bias=False)
#
#         self.q_norm = norm_layer(self.head_dim)
#         self.k_norm = norm_layer(self.head_dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#     def _forward_sdpa(self, q, k, v, attn_mask=None):
#         B, H, Lq, _ = q.shape
#         q = q * self.scale
#         k = k.transpose(-2, -1)  # (B, H, D, Lk)
#
#         attn = torch.matmul(q, k)
#
#         # if attn_mask is not None:
#         #     # attn = attn.masked_fill(attn_mask.unsqueeze(1), float("-1e9"))
#         #     attn = attn.masked_fill(attn_mask.unsqueeze(1), torch.finfo(attn.dtype).min)
#
#         # attn = attn.masked_fill(torch.isnan(attn), float("-1e9"))
#         # attn = attn.masked_fill(attn_mask.unsqueeze(1), torch.finfo(attn.dtype).min)
#         attn = torch.softmax(attn, dim=-1)
#
#         x = torch.matmul(attn, v)
#         x = x.transpose(1, 2).reshape(B, Lq, -1)
#         return self.out_proj(x)
#
#     def _forward_kv_cache(self, x, layer_index, kv_cache, update_kv_cache, attn_mask=None):
#         B, Lq = x.shape[:2]
#
#         qkv = self.qkv_linear(x).reshape(B, Lq, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # 防止共享内存造成的 inplace 报错
#
#         q, curr_k, curr_v = qkv[0].clone(), qkv[1].clone(), qkv[2].clone()  # 全部 clone
#
#         q = self.q_norm(q)
#         curr_k = self.k_norm(curr_k)
#
#         if kv_cache[layer_index]["k"] is not None:
#             k = torch.cat([kv_cache[layer_index]["k"], curr_k], dim=2)
#             v = torch.cat([kv_cache[layer_index]["v"], curr_v], dim=2)
#         else:
#             k = curr_k
#             v = curr_v
#
#         if update_kv_cache:
#             kv_cache[layer_index]["k"] = k
#             kv_cache[layer_index]["v"] = v
#
#         return self._forward_sdpa(q, k, v, attn_mask)
#
#     def forward_with_cache(
#         self,
#         text_feat,          # (B, Lq, Dq)
#         scene_feat,         # (B, Lk, Dkv)
#         obj_to_frame,       # (Lk,)
#         frame_anchor: int,
#         kv_cache: dict,
#         layer_index: int,
#         update_kv_cache: bool = True,
#         mode: str = "predict"
#     ):
#
#
#         B, Lq, _ = text_feat.shape
#         _, Lk, _ = scene_feat.shape
#
#         # Projection
#         text_proj = self.q_linear(text_feat)
#         scene_proj_k = self.k_linear(scene_feat)
#
#         x = torch.cat([text_proj, scene_proj_k], dim=1)  # (B, Lq + Lk, D)
#
#         # Construct obj_to_frame
#         obj_to_frame_full = torch.cat([
#             torch.full((Lq,), frame_anchor, device=text_feat.device),
#             obj_to_frame
#         ], dim=0)  # (Lq + Lk,)
#
#         if update_kv_cache:
#             kv_cache[layer_index]["obj_to_frame"] = obj_to_frame_full
#         else:
#             obj_to_frame_full = kv_cache[layer_index]["obj_to_frame"]
#
#
#
#         # Create attention mask
#         total_len = Lq + Lk
#         attn_mask = torch.zeros((1, total_len, total_len), dtype=torch.bool, device=text_feat.device)
#
#         scene_start = Lq
#         scene_idx = torch.arange(scene_start, total_len, device=text_feat.device)
#
#         scene_q_frame = obj_to_frame_full[scene_idx].view(1, -1, 1)  # (1, Lk, 1)
#         k_frame_all = obj_to_frame_full.view(1, 1, -1)  # (1, 1, Lq + Lk)
#
#         if mode == "predict":
#             sub_mask = scene_q_frame >= k_frame_all
#         elif mode == "retrospect":
#             sub_mask = scene_q_frame <= k_frame_all
#         else:
#             raise ValueError("mode must be 'predict' or 'retrospect'")
#
#         attn_mask[:, scene_idx, :] = sub_mask  # only mask scene-to-all attention rows
#
#         output = self._forward_kv_cache(
#             x, layer_index=layer_index, kv_cache=kv_cache,
#             update_kv_cache=update_kv_cache, attn_mask=attn_mask
#         )
#
#         text_out = output[:, :Lq]
#         scene_out = output[:, Lq:]
#         return text_out, scene_out
#
#
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     B, Lq, Lk = 1, 77, 33
#     Dq, Dkv, D = 1024, 64, 512
#     heads = 8
#     text_feat = torch.randn(B, Lq, Dq).cuda()
#     scene_feat = torch.randn(B, Lk, Dkv).cuda()
#     obj_to_frame = torch.tensor(
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#         device='cuda'
#     )
#     model = CausalUnifiedAttention(Dq, Dkv, D, heads).cuda()
#     kv_cache = {0: {"k": None, "v": None, "obj_to_frame": None}}
#     text_out, scene_out = model.forward_with_cache(
#         text_feat=text_feat,
#         scene_feat=scene_feat,
#         obj_to_frame=obj_to_frame,
#         frame_anchor=6,
#         kv_cache=kv_cache,
#         layer_index=0,
#         update_kv_cache=True,
#         mode="predict"  # or "retrospect"
#     )
#     print("✅ Text output shape:", text_out.shape)
#     print("✅ Scene output shape:", scene_out.shape)

# class CausalUnifiedAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim, num_heads, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_linear = nn.Linear(dim_q, dim)
#         self.k_linear = nn.Linear(dim_kv, dim)
#         self.v_linear = nn.Linear(dim_kv, dim)
#         self.qkv_linear = nn.Linear(dim, 3 * dim)
#
#         self.q_norm = norm_layer(self.head_dim)
#         self.k_norm = norm_layer(self.head_dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#     def _forward_sdpa(self, q, k, v, attn_mask=None):
#         B, H, Lq, _ = q.shape
#         q = q * self.scale
#         k = k.transpose(-2, -1)  # (B, H, D, Lk)
#         attn = torch.matmul(q, k)
#         attn = torch.softmax(attn, dim=-1)
#         x = torch.matmul(attn, v)
#         x = x.transpose(1, 2).reshape(B, Lq, -1)
#         return self.out_proj(x)
#
#     def _forward_kv_cache(self, x, layer_index, kv_cache, update_kv_cache, attn_mask=None):
#         B, Lq = x.shape[:2]
#
#         qkv = self.qkv_linear(x).reshape(B, Lq, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
#
#         q, curr_k, curr_v = qkv[0].clone(), qkv[1].clone(), qkv[2].clone()
#         q = self.q_norm(q)
#         curr_k = self.k_norm(curr_k)
#
#         if kv_cache[layer_index]["k"] is not None:
#             k = torch.cat([kv_cache[layer_index]["k"], curr_k], dim=2)
#             v = torch.cat([kv_cache[layer_index]["v"], curr_v], dim=2)
#         else:
#             k = curr_k
#             v = curr_v
#
#         if update_kv_cache:
#             kv_cache[layer_index]["k"] = k
#             kv_cache[layer_index]["v"] = v
#
#         return self._forward_sdpa(q, k, v, attn_mask)
#
#     def forward_with_cache(
#         self,
#         text_feat,          # (B, Lq, Dq)
#         scene_feat,         # (B, Lk, Dkv)
#         obj_to_frame,       # (Lk,)
#         frame_anchor: int,
#         kv_cache: dict,
#         layer_index: int,
#         update_kv_cache: bool = True,
#         mode: str = "predict"
#     ):
#         B, Lq, _ = text_feat.shape
#         _, Lk, _ = scene_feat.shape
#
#         text_proj = self.q_linear(text_feat)
#         scene_proj_k = self.k_linear(scene_feat)
#         x = torch.cat([text_proj, scene_proj_k], dim=1)  # (B, Lq + Lk, D)
#
#         obj_to_frame_full = torch.cat([
#             torch.full((Lq,), frame_anchor, device=text_feat.device),
#             obj_to_frame
#         ], dim=0)
#
#         # 初始化当前 layer 的缓存
#         if layer_index not in kv_cache:
#             kv_cache[layer_index] = {"k": None, "v": None, "obj_to_frame": None}
#
#         if update_kv_cache:
#             kv_cache[layer_index]["obj_to_frame"] = obj_to_frame_full
#         else:
#             if "obj_to_frame" not in kv_cache[layer_index] or kv_cache[layer_index]["obj_to_frame"] is None:
#                 raise RuntimeError(f"Missing `obj_to_frame` for layer {layer_index} in kv_cache. You must run with update_kv_cache=True first.")
#             obj_to_frame_full = kv_cache[layer_index]["obj_to_frame"]
#
#         # Create attention mask
#         total_len = Lq + Lk
#         attn_mask = torch.zeros((1, total_len, total_len), dtype=torch.bool, device=text_feat.device)
#         scene_start = Lq
#         scene_idx = torch.arange(scene_start, total_len, device=text_feat.device)
#         scene_q_frame = obj_to_frame_full[scene_idx].view(1, -1, 1)
#         k_frame_all = obj_to_frame_full.view(1, 1, -1)
#
#         if mode == "predict":
#             sub_mask = scene_q_frame >= k_frame_all
#         elif mode == "retrospect":
#             sub_mask = scene_q_frame <= k_frame_all
#         else:
#             raise ValueError("mode must be 'predict' or 'retrospect'")
#
#         attn_mask[:, scene_idx, :] = sub_mask
#
#         output = self._forward_kv_cache(
#             x, layer_index=layer_index, kv_cache=kv_cache,
#             update_kv_cache=update_kv_cache, attn_mask=attn_mask
#         )
#
#         text_out = output[:, :Lq]
#         scene_out = output[:, Lq:]
#         return text_out, scene_out













#
#
# class CausalUnifiedAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim, num_heads, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_linear = nn.Linear(dim_q, dim)
#         self.k_linear = nn.Linear(dim_kv, dim)
#         self.v_linear = nn.Linear(dim_kv, dim)
#
#         self.q_norm = norm_layer(self.head_dim)
#         self.k_norm = norm_layer(self.head_dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#     def _forward_sdpa(self, q, k, v, attn_mask=None):
#         B, H, Lq, _ = q.shape
#         q = q * self.scale
#         k = k.transpose(-2, -1)  # (B, H, D, Lk)
#         attn = torch.matmul(q, k)
#
#         if attn_mask is not None:
#             attn = attn.masked_fill(~attn_mask, float("-inf"))
#
#         attn = torch.softmax(attn, dim=-1)
#         x = torch.matmul(attn, v)
#         x = x.transpose(1, 2).reshape(B, Lq, -1)
#         return self.out_proj(x)
#
#     def forward(
#         self,
#         text_feat,      # (B, Lq, Dq)
#         scene_feat,     # (B, Lk, Dkv)
#         obj_to_frame,   # (Lk,)
#         frame_anchor: int,
#         mode: str = "predict"
#     ):
#         B, Lq, _ = text_feat.shape
#         _, Lk, _ = scene_feat.shape
#         total_len = Lq + Lk
#
#         # projection
#         q = self.q_linear(text_feat)  # (B, Lq, D)
#         k = self.k_linear(scene_feat)
#         v = self.v_linear(scene_feat)
#
#         # concat for unified attention
#         x = torch.cat([q, k], dim=1)  # (B, Lq + Lk, D)
#         qkv = x.reshape(B, total_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
#         q_split, k_split = qkv[:, :, :Lq], qkv[:, :, Lq:]
#         v_split = v.reshape(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lk, D)
#
#         # Norm and scale
#         q_split = self.q_norm(q_split)
#         k_split = self.k_norm(k_split)
#
#         # create attention mask
#         obj_to_frame_full = torch.cat([
#             torch.full((Lq,), frame_anchor, device=text_feat.device),
#             obj_to_frame
#         ], dim=0)
#         scene_start = Lq
#         scene_idx = torch.arange(scene_start, total_len, device=text_feat.device)
#         scene_q_frame = obj_to_frame_full[scene_idx].view(1, -1, 1)
#         k_frame_all = obj_to_frame_full.view(1, 1, -1)
#
#         attn_mask = torch.ones((1, total_len, total_len), dtype=torch.bool, device=text_feat.device)
#         if mode == "predict":
#             sub_mask = scene_q_frame >= k_frame_all
#         elif mode == "retrospect":
#             sub_mask = scene_q_frame <= k_frame_all
#         else:
#             raise ValueError("mode must be 'predict' or 'retrospect'")
#         attn_mask[:, scene_idx, :] = sub_mask
#         attn_mask = attn_mask.unsqueeze(1)  # (1, 1, Lq + Lk, Lq + Lk)
#
#         # attention
#         out = self._forward_sdpa(qkv, qkv, torch.cat([q, v], dim=1).reshape(B, total_len, self.num_heads, self.head_dim).transpose(1, 2), attn_mask=attn_mask)
#
#         # split text and scene output
#         text_out = out[:, :Lq]
#         scene_out = out[:, Lq:]
#         return text_out, scene_out
#
#
#
#
#
#





import torch.nn.functional as F

class BiCrossAttention(nn.Module):
    def __init__(self, dim_q_text=1024, dim_q_graph=640, dim_common=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_common = dim_common

        # Text attends to Graph
        self.q_text = nn.Linear(dim_q_text, dim_common)
        self.kv_graph = nn.Linear(dim_q_graph, dim_common * 2)
        self.out_text = nn.Linear(dim_common, dim_q_text)

        # Graph attends to Text
        self.q_graph = nn.Linear(dim_q_graph, dim_common)
        self.kv_text = nn.Linear(dim_q_text, dim_common * 2)
        self.out_graph = nn.Linear(dim_common, dim_q_graph)

    def attention(self, q, k, v):
        B, Lq, D = q.shape
        H = self.num_heads
        Dh = self.dim_common // H  # 每个 head 的维度

        q = q.view(B, Lq, H, Dh).transpose(1, 2)  # (B, H, Lq, Dh)
        k = k.view(B, -1, H, Dh).transpose(1, 2)  # (B, H, Lk, Dh)
        v = v.view(B, -1, H, Dh).transpose(1, 2)  # (B, H, Lk, Dh)

        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, Lq, Dh)
        out = out.transpose(1, 2).reshape(B, Lq, H * Dh)  # (B, Lq, D)
        return out

    def forward(self, text_feat, graph_feat):
        # text_feat: (B, Lt, 1024)
        # graph_feat: (B, Lg, 64)

        # === Text attends to Graph ===
        q_t = self.q_text(text_feat)                          # (B, Lt, dim_common)
        k_g, v_g = self.kv_graph(graph_feat).chunk(2, dim=-1) # (B, Lg, dim_common) x 2
        out_t = self.attention(q_t, k_g, v_g)
        out_t = self.out_text(out_t) + text_feat              # residual

        # === Graph attends to Text ===
        q_g = self.q_graph(graph_feat)
        k_t, v_t = self.kv_text(text_feat).chunk(2, dim=-1)
        out_g = self.attention(q_g, k_t, v_t)
        out_g = self.out_graph(out_g) + graph_feat

        return out_t, out_g








import torch
from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# bidirectional cross attention - have two sequences attend to each other with 1 attention step

class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.,
        talking_heads = False,
        prenorm = False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device
        x = self.norm(x)
        context = self.context_norm(context)
        # get shared query/keys and values for sequence and context
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)
        # split out head
        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))
        # get similarities
        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale
        # relative positional bias, if supplied
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        # mask
        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        # get attention along both sequence length and context length dimensions
        # shared similarity matrix
        attn = sim.softmax(dim = -1)
        context_attn = sim.softmax(dim = -2)
        # dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        # talking heads
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)
        # src sequence aggregates values from context, context aggregates values from src sequence
        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)
        context_out = self.context_to_out(context_out)
        if return_attn:
            return out, context_out, attn, context_attn
        return out, context_out







if __name__=="__main__":
    # 示例输入
    text_feat = torch.randn(1, 77, 1024)  # 文本特征
    graph_feat = torch.randn(1, 10, 64)   # 场景图特征

    # 模块初始化并运行
    model = BiCrossAttention()
    text_out, graph_out = model(text_feat, graph_feat)

    print(text_out.shape)  # (1, 77, 1024)
    print(graph_out.shape) # (1, 10, 64)



#
#
#
#
# if __name__=="__main__":
#     # 参数设置
#     dim_q = 1024
#     dim_kv = 64
#     dim = 256
#     num_heads = 4
#     Lq = 6  # 文本长度，例如6个token
#     Lk = 8  # 场景图节点数量，例如8个节点
#     B = 2  # batch size
#     frame_anchor = 5
#
#     # 初始化模型
#     model = CausalUnifiedAttention(dim_q=dim_q, dim_kv=dim_kv, dim=dim, num_heads=num_heads)
#
#     # 输入特征
#     text_feat = torch.randn(B, Lq, dim_q)
#     scene_feat = torch.randn(B, Lk, dim_kv)
#     obj_to_frame = torch.randint(low=0, high=10, size=(Lk,), dtype=torch.long)
#
#     # 模式为 "predict" 或 "retrospect"
#     mode = "predict"
#     text_out, scene_out = model(
#         text_feat=text_feat,
#         scene_feat=scene_feat,
#         obj_to_frame=obj_to_frame,
#         frame_anchor=frame_anchor,
#         mode=mode
#     )
#
#     print("Text Output Shape:", text_out.shape)  # [B, Lq, dim]
#     print("Scene Output Shape:", scene_out.shape)  # [B, Lk, dim]
#
