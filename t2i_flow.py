import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import wandb
import torchvision.utils as vutils

from model import UNetModelWithTextEmbedding
from dataset import CFMDataset

def tensor_to_image(tensor):
    return F.to_pil_image(tensor.clamp(0, 1))

# 🔹 加载数据
ds = load_from_disk("/root/mems_dataset").shuffle(seed=42).select(range(200))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder = CLIPTextModel.from_pretrained(
    "/root/.cache/modelscope/hub/models/openai-mirror/clip-vit-base-patch16"
).to(device)
text_encoder.eval()

tokenizer = CLIPTokenizer.from_pretrained(
    "/root/.cache/modelscope/hub/models/openai-mirror/clip-vit-base-patch16"
)

train_ds = CFMDataset(ds, text_encoder, tokenizer, device)
train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)

model = UNetModelWithTextEmbedding(
    dim=(3, 64, 64), num_channels=64, num_res_blocks=1,
    embedding_dim=512, dropout=0.05, num_heads=4
).to(device)

optimizer = torch.optim.AdamW(model.parameters())
n_epochs = 15000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# 🔹 开关：是否启用 W&B
use_wandb = True  # 调试时改成 False

if use_wandb:
    # 🔹 初始化 W&B
    wandb.init(project="cfm-image-generation", name="flow100",
            config={"epochs": n_epochs, "batch_size": 20})
else:
    # 提供一个空的 mock wandb，避免代码报错
    class DummyWandb:
        def log(self, *args, **kwargs): pass
        class Image: 
            def __init__(self, *args, **kwargs): pass
        class Video: 
            def __init__(self, *args, **kwargs): pass

    wandb = DummyWandb()

# 🔹 Euler 采样函数
def euler_method(model, text_embedding, t_steps, dt, noise):
    y = noise
    y_values = [y]
    with torch.no_grad():
        for t in t_steps[1:]:
            dy = model(t.to(device), y, text_embeddings=text_embedding)
            y = y + dy * dt
            y_values.append(y)
    return torch.stack(y_values)

def sample_and_log(epoch, text_embeddings, n_samples=20, save_path="sample.gif", tag="train"):
    noise = torch.randn((n_samples, 3, 64, 64), device=device)
    t_steps = torch.linspace(0, 1,100, device=device)
    dt = t_steps[1] - t_steps[0]

    # 🔹 always eval during sampling
    model.eval()
    with torch.no_grad():
        results = euler_method(model, text_embeddings, t_steps, dt, noise)
    model.train()

    # 取最后一步
    final_batch = results[-1]   # (n_samples, 3, 64, 64)

    # 拼成网格 (nrow=5 -> 4x5 grid)
    grid = vutils.make_grid(final_batch, nrow=5, normalize=True, value_range=(0,1))
    grid_img = tensor_to_image(grid.cpu())
    grid_img.save(f"sample_epoch{epoch}_{tag}_{len(t_steps)}.png")

    wandb.log({
        f"sample_grid_{tag}": wandb.Image(grid_img, caption=f"{tag}_epoch_{epoch}_{len(t_steps)}")
    })

    # 存 GIF
    frames = [tensor_to_image(results[idx, 0].cpu()) for idx in range(0, results.shape[0], 5)]
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=300, loop=0)

    # log 到 wandb
    wandb.log({
        f"sample_gif_{len(t_steps)}_{tag}": wandb.Video(save_path, fps=4, format="gif")
    })

# 🔹 训练循环
for epoch in tqdm(range(n_epochs)):
    losses = []
    for batch in train_loader:
        optimizer.zero_grad()
        x1 = batch["image"].to(device)
        text_embeddings = batch["caption_embedding"].to(device)

        x0 = torch.randn_like(x1).to(device)
        t = torch.rand(x0.shape[0], 1, 1, 1).to(device)

        xt = t * x1 + (1 - t) * x0
        ut = x1 - x0
        t = t.squeeze()

        vt = model(t, xt, text_embeddings=text_embeddings)
        loss = torch.mean(((vt - ut) ** 2))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})

    if (epoch + 1) % 3000 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
        sample_and_log(epoch, text_embeddings)

        input_prompt = "drive_freq:32600Hz,split:between_3.0%_and_5.0%,parasitic:less_than_5000Hz,x_stiffness:8000N/m,nonlinearity:low"
        text_embedding = train_ds.get_embed(input_prompt).unsqueeze(0)
        sample_and_log(epoch, text_embedding, n_samples=1, save_path=f"sample_prompt_epoch{epoch}.gif", tag="unseen")