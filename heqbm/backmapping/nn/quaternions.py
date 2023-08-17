import torch

def get_quaternions(batch: int, rot_axes: torch.Tensor, angles: torch.Tensor):
        real_parts = torch.cos(angles)
        imaginary_parts_multiplier = torch.sin(angles)
        if angles.size(0) < rot_axes.size(0): 
            real_parts = real_parts.repeat(batch, 1)
            imaginary_parts_multiplier = imaginary_parts_multiplier.repeat(batch, 1)
        assert real_parts.size(0) == rot_axes.size(0)
        imaginary_parts = imaginary_parts_multiplier * rot_axes
        q = torch.cat((real_parts, imaginary_parts), -1)
        return q

def q_conjugate(q: torch.Tensor):
    scaling = torch.tensor([1, -1, -1, -1], device=q.device)
    return scaling * q

def q_mult(q1: torch.Tensor, q2: torch.Tensor):
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return torch.stack((w, x, y, z), -1)

def qv_mult(q1: torch.Tensor, v2: torch.Tensor):
    if v2.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {v2.shape}.")
    real_parts = v2.new_zeros(v2.shape[:-1] + (1,))
    q2 = torch.cat((real_parts, v2), -1)
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[..., 1:]