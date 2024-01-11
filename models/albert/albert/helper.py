import torch
from einops import rearrange

"""
    @openai - VD-VAE
    https://github.com/openai/vdvae/blob/main/vae_helpers.py

    Nice helper module as calling super.__init__() gets annoying
"""
class HelperModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

"""
    @lucidrains - Phil Wang (nystrom-attention)
    https://github.com/lucidrains/nystrom-attention/blob/main/nystrom_attention/nystrom_attention.py
"""
def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 1):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for i in range(iters):
        xz = x @ z
        xz_1 = (7 * I - xz)
        xz_2 = (15 * I - (xz @ xz_1))
        del xz_1
        xz_3 = (13 * I - (xz @ xz_2))
        del xz
        del xz_2
        z = 0.25 * z @ xz_3
        del xz_3
    return z
