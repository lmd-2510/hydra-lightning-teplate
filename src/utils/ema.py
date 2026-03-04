from tqdm.auto import tqdm
import torch
from torch import nn

class EMA:
    def __init__(self,
                 model: nn.Module,
                 decay: float = 0.999,
                 device: str = None):
        self.decay = decay
        self.device = device
        self.num_updates = 0
        self.shadow = {}
        self.backup_params = {}
        self.backup_buffers = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().to(self.device)

    @torch.inference_mode()
    def update(self, model: nn.Module):
        self.num_updates += 1
        
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue     
            new_val = param.detach()
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(new_val, alpha=(1.0 - decay))
            else:
                self.shadow[name] = new_val.clone().to(self.device)


    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
            "num_updates": self.num_updates
        }

    def load_state_dict(self, state: dict):
        self.decay = float(state["decay"])
        self.shadow = {k: v.clone().to(self.device) for k, v in state["shadow"].items()}
        self.num_updates = int(state["num_updates"])

    @torch.inference_mode()
    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup_params[name] = param.detach().clone() 
            param.data.copy_(self.shadow[name])

        for name, buffer in model.named_buffers():
            self.backup_buffers[name] = buffer.detach().clone()

    @torch.inference_mode()
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}

        for name, buffer in model.named_buffers():
            if name in self.backup_buffers:
                buffer.data.copy_(self.backup_buffers[name])
        self.backup_buffers = {}

@torch.inference_mode()
def update_bn_stats(model, dataloader, device):
    if dataloader is None:
        return
    
    model.train()
    
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        colour="yellow",
        desc="Updating BatchNorm...",
        leave=False
    )

    for cnt, (X, _) in enumerate(pbar):
        X = X.to(device)
        model(X)

        if cnt + 1 == 200:
            break

    pbar.close()  
    model.eval()