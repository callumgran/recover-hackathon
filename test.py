import torch, platform, os
print("arch:", platform.machine())
print("torch:", torch.__version__, "torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))