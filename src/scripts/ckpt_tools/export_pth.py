import ipdb
import torch


# python -m src.scripts.ckpt_tools.export_pth
if __name__ == "__main__":

    # 加载ckpt文件
    checkpoint = torch.load('logs/checkpoints/coordar/last.ckpt', map_location='cpu', weights_only=False)
    ipdb.set_trace()


    # 方式1：如果是完整的checkpoint（包含模型状态字典）
    if 'state_dict' in checkpoint:
        torch.save(checkpoint['state_dict'], 'model.pth')
    # 方式2：如果ckpt本身就是状态字典
    else:
        torch.save(checkpoint, 'model.pth')