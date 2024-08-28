import numpy as np
import torch

    # Yash: [[[1, 0]], [[0, 1]]],  3, 1
def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))
    """
    Yash:
    (tensor([[[[[[[1.],
              [0.]]],

            [[[0.],
              [1.]]]]]]]),
    torch.Size([1, 1, 1, 2, 1, 2, 1]))
    """


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3) # Yash: Shape: num_samples*batch_size*num_components*2*2*2

    d = m.dim() # Yash: 6
    n = m.shape[-3] # Yash: 2
    siz0 = m.shape[:-3] # Yash: num_samples*batch_size*num_components
    siz1 = m.shape[-2:] # Yash: 2*2
    m2 = m.unsqueeze(-2) # Yash: Shape: num_samples*batch_size*num_components*2*2*1*2
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))
    """
    Yash: It take the first 2x2 matrix and puts it on the fist 2x2 diagnal (0,0 to 1,1) 
    and takes the second 2x2 matrix and puts it on the last 2x2 diagnal (2,2 to 3,3) keeping everythig else as 0
    """


def tile(a, dim, n_tile, device='cpu'):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)