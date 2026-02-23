from matplotlib import pyplot as plt
import torch


def show_pred(batch, pred, log_dir, key="nocs"):
    target = batch[key]
    if (target > 128).any():
        target = target / 255.0
    recon = pred["recon"]

    bs = len(recon)
    if bs > 8:
        bs = 8  # limit to 8 images
    fig, axs = plt.subplots(bs, 2, figsize=(10, 3 * bs))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        axs[i][0].imshow(target[i].permute(1, 2, 0).clip(0, 1).float().cpu().numpy())
        axs[i][0].set_title(key)
        axs[i][1].imshow(recon[i].permute(1, 2, 0).clip(0, 1).float().cpu().numpy())
        axs[i][1].set_title("recon")
        for j in range(2):
            axs[i][j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tokenizer_pred.png")
    plt.close()


def show_steps(batch, pred, log_dir, key="nocs"):
    target = batch[key]
    if (target > 128).any():
        target = target / 255.0
    recon = pred["recon"]
    steps = pred["steps"]

    bs = len(recon)
    if bs > 8:
        bs = 8  # limit to 8 images
    fig, axs = plt.subplots(bs, 2 + len(steps), figsize=(10, 3 * bs))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        axs[i][0].imshow(target[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][0].set_title(key)
        axs[i][1].imshow(recon[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][1].set_title("recon")
        for j in range(len(steps)):
            axs[i][2 + j].imshow(steps[j][i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
            axs[i][2 + j].set_title(f"step {j}")
        for j in range(2 + len(steps)):
            axs[i][j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tokenizer_steps.png")
    plt.close()
