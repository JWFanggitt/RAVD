import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import copy
from .gcn import GraphTripleConvNet, _init_weights, build_mlp
from causl_attention import BiCrossAttention,BidirectionalCrossAttention
import torch.nn.functional as F
from .temporal_transformer import MultiframeIntegrationTransformer
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline


def create_tensor_by_assign_samples_to_img(samples, sample_to_img, max_sample_per_img, batch_size):
    dtype, device = samples.dtype, samples.device
    N = batch_size
    D = samples.shape[1]
    assert (sample_to_img.max() + 1) == N

    samples_per_img = []
    for i in range(N):
        s_idxs = (sample_to_img == i).nonzero().view(-1)
        sub_sample = samples[s_idxs]
        len_cur = sub_sample.shape[0]
        if len_cur > max_sample_per_img:
            sub_sample = sub_sample[:max_sample_per_img, :]
        if len_cur < max_sample_per_img:
            zero_vector = torch.zeros([1, D]).to(device)
            padding_vectors = torch.cat([copy.deepcopy(zero_vector) for _ in range(max_sample_per_img - len_cur)], dim=0) # [res, D]
            sub_sample = torch.cat([sub_sample, padding_vectors], dim=0)
        sub_sample = sub_sample.unsqueeze(0)
        samples_per_img.append(sub_sample)
    samples_per_img = torch.cat(samples_per_img, dim=0).to(device)

    return samples_per_img


def create_tensor_by_assign_samples_to_video_frames(samples, sample_to_img, max_sample_per_img, batch_size, num_frames):
    """
    samples: Tensor[N_total, D]，表示每个节点的表示
    sample_to_img: Tensor[N_total]，表示每个 sample 属于哪个 (batch_idx × T + frame_idx)
    max_sample_per_img: 每帧最大节点数
    batch_size: B
    num_frames: T
    return: Tensor[B, T, max_sample_per_img, D]
    """
    N_total, D = samples.shape
    device = samples.device
    samples_per_frame = []

    for b in range(batch_size):
        video_frames = []
        for t in range(num_frames):
            idx = b * num_frames + t  # 当前帧在 sample_to_img 中的标记
            s_idxs = (sample_to_img == idx).nonzero(as_tuple=True)[0]
            sub_sample = samples[s_idxs]
            len_cur = sub_sample.shape[0]

            if len_cur > max_sample_per_img:
                sub_sample = sub_sample[:max_sample_per_img, :]
            elif len_cur < max_sample_per_img:
                pad_len = max_sample_per_img - len_cur
                pad_vector = torch.zeros(1, D).to(device)
                padding = pad_vector.expand(pad_len, D)
                sub_sample = torch.cat([sub_sample, padding], dim=0)

            video_frames.append(sub_sample.unsqueeze(0))  # [1, N, D]

        video_frames = torch.cat(video_frames, dim=0)     # [T, N, D]
        samples_per_frame.append(video_frames.unsqueeze(0))  # [1, T, N, D]

    final_tensor = torch.cat(samples_per_frame, dim=0)  # [B, T, N, D]
    return final_tensor











class HookTool:
    def __init__(self):
        self.extract_fea_in = None
        self.extract_fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.extract_fea_in = fea_in
        self.extract_fea = fea_out

def get_linear_feas_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks

pretrained_model_path=r"/media/lotvs/WD_4T/checkpoint-30000"
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

def sample_noise(latents, noise_strength, use_offset_noise):
    # b,n,d=latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    # offset_noise = None
    #
    # if use_offset_noise:
    #     offset_noise = torch.randn(n, d,1, 1, device=latents.device)
    #     noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents





class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 2)  # 输出 scale 和 shift 两份
        )
        self.norm= nn.LayerNorm(embed_dim * 2)

    def forward(self, t):
        t = t.view(-1, 1).float()  # (B, 1)
        temb = self.time_mlp(t)
        temb=self.norm(temb)# (B, 2 * dim)
        return temb.chunk(2, dim=-1)  # 返回 scale, shift




class SceneVAEModel(nn.Module):
    """
    VAE-based network for scene generation and manipulation from a scene graph.
    It has a separate embedding of shape and bounding box latents.
    """

    def __init__(self, args, num_objs, num_rels,clip_model):
        super(SceneVAEModel, self).__init__()

        gconv_dim = args.embedding_dim  # 64
        gconv_hidden_dim = gconv_dim * 4  # 64 * 4
        box_embedding_dim = args.embedding_dim  # 64
        obj_embedding_dim = args.embedding_dim  # 64

        self.obj_embeddings_encoder = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.obj_embeddings_decoder = nn.Embedding(num_objs + 1, obj_embedding_dim)

        self.rel_embeddings_encoder = nn.Embedding(num_rels, args.embedding_dim * 2)
        self.rel_embeddings_decoder = nn.Embedding(num_rels, args.embedding_dim * 2)

        self.box_embeddings = nn.Linear(4, box_embedding_dim)

        self.use_angles=False
        self.angle_embedding_dim = int( args.embedding_dim / 4)
        self.class_num=7
        self.clip_model = clip_model
        self.relation_num=8
        self.Nangle = 24

        if self.use_angles:
            # angle prediction net
            self.angle_embeddings = nn.Embedding(self.Nangle, self.angle_embedding_dim)


        self.mlp_mean_var = build_mlp(
            [args.embedding_dim * 2 + 512, gconv_hidden_dim, args.embedding_dim * 2],
            batch_norm="batch",
            final_nonlinearity=True
        )
        self.mlp_mean = build_mlp(
            [args.embedding_dim * 2, box_embedding_dim],
            batch_norm="batch",
            final_nonlinearity=False
        )
        self.mlp_var = build_mlp(
            [args.embedding_dim * 2, box_embedding_dim],
            batch_norm="batch",
            final_nonlinearity=False
        )
        self.mlp_box = build_mlp(
            [args.embedding_dim * 2 + 512, gconv_hidden_dim, 4],
            batch_norm="batch",
            final_nonlinearity=False
        )

        self.s_rel_compress = nn.Linear(640, self.class_num)
        self.o_rel_compress = nn.Linear(640, self.class_num)
        self.r_rel_compress = nn.Linear(640, self.relation_num)
        self.all_compress = nn.Linear(640, self.class_num)

        self.time_emb = TimeEmbedding(64)
        # self.mlp_box = build_mlp(
        #     [args.embedding_dim * 2 + 512, gconv_hidden_dim, 4],
        #     batch_norm="batch",
        #     final_nonlinearity=True
        # )

        gconv_encoder_kwargs = {
            'input_dim_obj': gconv_dim * 2 + 512,
            'input_dim_pred': gconv_dim * 2 + 512,
            'hidden_dim': gconv_hidden_dim,
            'num_layers': 5,
            'pooling': 'avg',
            'mlp_normalization': 'batch',
            'residual': True  #
        }

        gconv_decoder_kwargs = {
            'input_dim_obj': gconv_dim * 2 + 512,
            'input_dim_pred': gconv_dim * 2 + 512,
            'hidden_dim': gconv_hidden_dim,
            'num_layers': 5,
            'pooling': 'avg',
            'mlp_normalization': 'batch',
            'residual': True
        }

        self.gconv_encoder = GraphTripleConvNet(**gconv_encoder_kwargs)
        self.gconv_decoder = GraphTripleConvNet(**gconv_decoder_kwargs)

        # initialization
        self.box_embeddings.apply(_init_weights)
        self.mlp_mean_var.apply(_init_weights)
        self.mlp_mean.apply(_init_weights)
        self.mlp_var.apply(_init_weights)
        self.mlp_box.apply(_init_weights)
        self.graph_projection = nn.Linear(640 * 2,640)
        self.max_sample_per_img=7
        self.temporal_layer=MultiframeIntegrationTransformer(T=16,embed_dim=512,layers=3)
        self.frame_anchor=6
        self.heads=8
        self.dim=1024
        self.graph_dim=64
        self.output_dim=1024

        # self.causal_model= CausalUnifiedAttention(self.dim, self.graph_dim, self.output_dim, self.heads)
        self.causal_model= BiCrossAttention()
        # self.causal_model=BidirectionalCrossAttention(
        #                 dim = 1024,
        #                 heads = 8,
        #                 dim_head = 64,
        #                 context_dim = 640
        #             )
        self.training=True
        self.model_pattern= "predict"
        self.kv_cache={0: {"k": None, "v": None}}
        # 定义两个线性层
        # self.scene_proj = nn.Sequential(
        #     nn.Linear(1024, 512),  # 第一步：降维
        #     nn.ReLU(),  # 可换成 nn.GELU() 或其他激活
        #     nn.Linear(512, 64)  # 第二步：输出最终维度
        # )




    def build_temporal_obj_mask(self,obj_to_frame, frame_anchor, mode="predict", B=1, L=1):
        """
        构造 (B, L, N) 的 attention mask,True 表示不允许 attend。
        """
        N = obj_to_frame.shape[0]
        if mode == "predict":
            valid = obj_to_frame > frame_anchor
        elif mode == "retrospect":
            valid = obj_to_frame < frame_anchor
        else:
            raise ValueError("mode must be 'predict' or 'retrospect'")
        mask = ~valid  # shape: (N,)
        return mask.unsqueeze(0).unsqueeze(0).expand(B, L, N)  # (B, Lq, Lk)




    def pool_samples(self, samples, obj_to_img, pooling='avg'):
        dtype, device = samples.dtype, samples.device
        O, D = samples.size()

        N = obj_to_img.data.max().item() + 1

        out = torch.zeros(N, D, dtype=dtype, device=device)
        idx = obj_to_img.view(O, 1).expand(O, D)
        out = out.scatter_add(0, idx, samples)

        if pooling == 'avg':
            ones = torch.ones(O, dtype=dtype, device=device)
            obj_counts = torch.zeros(N, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
            obj_counts = obj_counts.clamp(min=1)
            out = out / obj_counts.view(N, 1)
        elif pooling != 'sum':
            raise ValueError('Invalid pooling "%s"' % pooling)

        return out

    def build_graph_global_feature(self, obj_vecs, pred_vecs, obj_to_video, triple_to_video):
        obj_fea = self.pool_samples(obj_vecs, obj_to_video)
        pred_fea = self.pool_samples(pred_vecs, triple_to_video)
        graph_global_fea = self.graph_projection(torch.cat([obj_fea, pred_fea], dim=1))
        return F.normalize(graph_global_fea, dim=-1).unsqueeze(1)

    def reverse_denoise_graph_feature(self, noisy_graph_global_fea, prompt_feat, timesteps):
        denoised_graph = []

        with torch.no_grad():
            for batch_idx in range(noisy_graph_global_fea.shape[0]):
                sample = noisy_graph_global_fea[batch_idx:batch_idx + 1]
                text_cond = prompt_feat[batch_idx:batch_idx + 1]
                start_t = int(timesteps[batch_idx].item())

                for step_t in range(start_t, -1, -1):
                    _, scene_out = self.causal_model(text_cond, sample)
                    pred_noise = scene_out - sample
                    sample = noise_scheduler.step(
                        pred_noise.squeeze(1),
                        step_t,
                        sample.squeeze(1)
                    ).prev_sample.unsqueeze(1).to(noisy_graph_global_fea.dtype)

                denoised_graph.append(F.normalize(sample, dim=-1))

        return torch.cat(denoised_graph, dim=0)

    def encoder(self, objs, obj_clip_embs, boxes, triples, rel_clip_embs,angles):
        ##print(list(self.mlp_mean.parameters()))
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # Shape: (T, 1), s-subject, p-predicate (relation), o-object
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Shape: (T,)
        edges = torch.stack([s, o], dim=1)  # Shape: (T, 2)

        # Relation Embeding
        rel_embs = self.rel_embeddings_encoder(p)  # Shape: (T, embedding_dim * 2) = (T, 64 * 2)
        rel_embs = torch.cat([rel_clip_embs, rel_embs],
                             dim=1)  # Shape: (T, clip_dim + embedding_dim * 2) = (T, 512 + 64 * 2)

        # Node Embeding
        obj_embs = self.obj_embeddings_encoder(objs)  # Shape: (O, embedding_dim) = (O, 64)
        obj_embs = torch.cat([obj_clip_embs, obj_embs], dim=1)  # Shape: (O, clip_dim + embedding_dim) = (O, 512 + 64)
        box_embs = self.box_embeddings(boxes)  # Shape: (O, embedding_dim) = (O, 64)
        if self.use_angles:
            angle_vecs =self.angle_embeddings(angles)
            obj_embs = torch.cat([obj_embs,box_embs , angle_vecs], dim=1)
        obj_embs = torch.cat([obj_embs, box_embs],dim=1)
        #                      dim=1)  # Shape: (O, clip_dim + embedding_dim * 2) = (O, 512 + 64 * 2)
        # Encoding
        all_embs, _ = self.gconv_encoder(obj_embs, rel_embs,
                                         edges)  # Shape: (O, clip_dim + embedding_dim * 2) = (O, 512 + 64 * 2)
        all_embss = self.mlp_mean_var(all_embs)  # Shape: (O, embedding_dim * 2) = (O, 64 * 2)
        mu = self.mlp_mean(all_embss)  # Shape: (O, embedding_dim) = (O, 64)
        logvar = self.mlp_var(all_embss)  # Shape: (O, embedding_dim) = (O, 64)
        return all_embs, _,mu,logvar

    def encode_image_local_global(self, video):
        with torch.no_grad():
            batch, frames, channels, h, w = video.shape
            video = video.view(-1, channels,h, w )
            extract_linear_feas = get_linear_feas_by_hook(self.clip_model.visual)
            global_image_fea = self.clip_model.encode_image(video)
        global_image_fea=global_image_fea.view(batch,frames, global_image_fea.shape[1])
        global_image_fea=self.temporal_layer(global_image_fea)
        local_image_fea = extract_linear_feas[-1].extract_fea

        return local_image_fea.detach(), global_image_fea.detach()

        # return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



    def decoder(self, objs, obj_clip_embs, z, triples, rel_clip_embs,angles):
        s, p, o = triples.chunk(3, dim=1)  # Shape: (T, 1), s-subject, p-predicate, o-object
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Shape: (T,)
        edges = torch.stack([s, o], dim=1)  # Shape: (T, 2)

        # Relation Embeding
        rel_embs = self.rel_embeddings_decoder(p)  # Shape: (T, embedding_dim) = (T, embedding_dim)
        rel_embs = torch.cat([rel_clip_embs, rel_embs], dim=1)  # Shape: (T, clip_dim + embedding_dim) = (T, 512 + 64)

        # Node Embeding
        obj_embs = self.obj_embeddings_decoder(objs)  # Shape: (O, embedding_dim) = (O, 64)
        obj_embs = torch.cat([obj_clip_embs, obj_embs], dim=1)  # Shape: (O, clip_dim + embedding_dim) = (T, 512 + 64)
        obj_embs = torch.cat([obj_embs, z], dim=1)  # Shape: (O, clip_dim + embedding_dim * 2) = (T, 512 + 64 * 2)

        # Decoding
        obj_vecs,pred_vecs = self.gconv_decoder(obj_embs, rel_embs, edges)
        # box_pred = self.mlp_box(obj_vecs)

        # return obj_vecs[edges[:,0]],obj_vecs[edges[:,1]],pred_vecs,torch.sigmoid(box_pred)

        return obj_vecs,pred_vecs

    def forward_causal_attention(
            self, prompt_feat, noisy_z, obj_to_frame, frame_anchor, layer_index=0, mode="predict"
    ):

        # if self.training:
        #     # kv_cache = {layer_index: {"k": None, "v": None}}
        #     kv_cache={}
        #     update = True
        # else:
        #     kv_cache = self.kv_cache
        #     update = False

        return self.causal_model(
            text_feat=prompt_feat,
            scene_feat=noisy_z,
            obj_to_frame=obj_to_frame,
            frame_anchor=frame_anchor,
            mode=mode,
        )


    def forward(self, objs, obj_clip_embs, boxes, triples, rel_clip_embs,angles,obj_to_video,triple_to_video,video,prompt_feat,obj_to_frame,timesteps,task_type):

        obj_frames = obj_to_frame  # e.g., [0,0,1,1,...,15]

        if task_type == "predict":
            mask = obj_frames >= 6
        elif task_type == "backward":
            mask = obj_frames < 6
        else:
            raise ValueError("Unknown task_type")


        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        obj_vecs, pred_vecs,mu,logvar=self.encoder(objs, obj_clip_embs, boxes, triples, rel_clip_embs, angles)

        noise = sample_noise(obj_vecs, noise_strength=0.1, use_offset_noise=None).to(obj_vecs.dtype)
        noisy_obj_vecs = obj_vecs.clone()
        noisy_obj_vecs[mask] = noise_scheduler.add_noise(
            obj_vecs[mask],
            noise[mask],
            timesteps[obj_to_video[mask]]
        ).to(noisy_obj_vecs.dtype)
        
        # 4. Global pooling 构建全局图 latent
        noisy_graph_global_fea = self.build_graph_global_feature(
            noisy_obj_vecs, pred_vecs, obj_to_video, triple_to_video
        )
        if self.training:
            graph_global_fea = noisy_graph_global_fea
        else:
            graph_global_fea = self.reverse_denoise_graph_feature(
                noisy_graph_global_fea, prompt_feat, timesteps
            )

        # 5. 扩散模型前向，预测节点噪声
        text_out, scene_out = self.causal_model(prompt_feat, graph_global_fea)  # scene_out: [N, D]

        z = self.reparameterize(mu, logvar)
        p_obj_vecs,p_pred_vecs  = self.decoder(objs, obj_clip_embs, z, triples, rel_clip_embs,
                                                                         angles)


        return text_out,p_obj_vecs,noise,mask

        # return mu, logvar,torch.sigmoid(s_obj_vec),torch.sigmoid(o_obj_vec),torch.sigmoid(pred_vecs),new_box_pred, norm_global_image_features,global_image_features,norm_global_graph_features, alL_obj_vec,local_image_feature

    def sample_box(self, mean_est, cov_est, objs, obj_clip_embs, triples, rel_clip_embs, device):
        with torch.no_grad():
            # mean_est = np.zeros(64)
            # cov_est = np.eye(64)
            z = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, objs.size(0))).float().to(device)
            box_pred = self.decoder(objs, obj_clip_embs, z, triples, rel_clip_embs)
            return box_pred

    def collect_data_statistics(self, train_loader, device):
        pbar = tqdm(train_loader, file=sys.stdout)
        mean_cat = []
        for idx, batch in enumerate(pbar):
            # for idx, batch in enumerate(train_loader):

            imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch
            objs, triples, boxes = objs.to(device), triples.to(device), boxes.to(device)
            obj_clip_embs, rel_clip_embs = obj_clip_embs.to(device), rel_clip_embs.to(device)

            mean, logvar = self.encoder(objs, obj_clip_embs, boxes, triples, rel_clip_embs)
            mean, logvar = mean.cpu().clone(), logvar.cpu().clone()

            mean = mean.data.cpu().clone()
            mean_cat.append(mean)

        mean_cat = torch.cat(mean_cat, dim=0)
        mean_est = torch.mean(mean_cat, dim=0, keepdim=True)  # Shape: (1, embedding_dim) = (1, 64)
        cov_est = np.cov((mean_cat - mean_est).numpy().T)
        mean_est = mean_est[0]

        return mean_est, cov_est
