def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)



def initialize_latents_with_conditional(latents, cond_latents, m):
    """
    latents: (B, 4, F, H, W)
    cond_latents: (B, 4, |m|, H, W)
    m: int, >0 means using first m frames as condition, <0 means using last |m| frames
    """
    B, C, F, H, W = latents.shape

    if m > 0:
        cond_frame_indices = list(range(m))
        latents[:, :, :m, :, :] = cond_latents
    else:
        m_abs = abs(m)
        cond_frame_indices = list(range(F - m_abs, F))
        latents[:, :, -m_abs:, :, :] = cond_latents

    for t in range(F):
        if t in cond_frame_indices:
            continue

        prev_cond = max([i for i in cond_frame_indices if i < t], default=None)
        next_cond = min([i for i in cond_frame_indices if i > t], default=None)

        if prev_cond is not None and next_cond is not None:
            ratio = (t - prev_cond) / (next_cond - prev_cond)
            latents[:, :, t, :, :] = (
                (1 - ratio) * latents[:, :, prev_cond, :, :] +
                ratio * latents[:, :, next_cond, :, :]
            )
        elif prev_cond is not None:
            latents[:, :, t, :, :] = latents[:, :, prev_cond, :, :]
        elif next_cond is not None:
            latents[:, :, t, :, :] = latents[:, :, next_cond, :, :]

    return latents, cond_frame_indices

def style_transfer_to_nearest_cond(latents, cond_frame_indices):
    F = latents.shape[2]
    for f in range(F):
        if f in cond_frame_indices:
            continue
        nearest = min(cond_frame_indices, key=lambda x: abs(x - f))
        latents[:, :, f, :, :] = adaptive_instance_normalization(
            latents[:, :, f, :, :],
            latents[:, :, nearest, :, :]
        )
    return latents

# Inside __call__() method of LAMPPipeline, replace original latent initialization with:

# Automatic generation of m and cond_latents
max_m = min(6, video_length // 2)
m = random.choice(list(range(-max_m, 0)) + list(range(1, max_m + 1)))

B = batch_size * num_videos_per_prompt
C = num_channels_latents
F = video_length
H = height // self.vae_scale_factor
W = width // self.vae_scale_factor

raw_latents = torch.randn((B, C, F, H, W), device=device, dtype=text_embeddings.dtype)
raw_latents = raw_latents * self.scheduler.init_noise_sigma

if m > 0:
    cond_frame_indices = list(range(m))
    cond_latents = raw_latents[:, :, :m, :, :]
elif m < 0:
    cond_frame_indices = list(range(F + m, F))
    cond_latents = raw_latents[:, :, m:, :, :]
else:
    cond_frame_indices = []
    cond_latents = None

if cond_latents is not None and m != 0:
    latents, cond_frame_indices = initialize_latents_with_conditional(raw_latents.clone(), cond_latents, m)
else:
    latents = raw_latents.clone()
    cond_frame_indices = []

# In the denoising loop, after latents update, add:
if i > 30 and len(cond_frame_indices) > 0:
    latents = style_transfer_to_nearest_cond(latents, cond_frame_indices)