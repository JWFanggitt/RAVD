# patch_cached_download.py

import huggingface_hub
from huggingface_hub import hf_hub_download

def cached_download(url_or_filename, *args, **kwargs):
    """
    A minimal drop-in replacement for deprecated `cached_download`.
    Treats `url_or_filename` as `repo_id`, and assumes typical usage from diffusers.
    """
    revision = kwargs.pop("revision", "main")
    cache_dir = kwargs.pop("cache_dir", None)
    return hf_hub_download(
        repo_id=url_or_filename,
        filename=kwargs.pop("filename", None),
        subfolder=kwargs.pop("subfolder", None),
        cache_dir=cache_dir,
        revision=revision,
        **kwargs
    )

# 注入 monkey patch
huggingface_hub.cached_download = cached_download
