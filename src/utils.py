import mrcfile
import numpy as np
import torch
from torch.nn import functional as F
# from sklearn.cluster import KMeans
import sys
import math

from openfold.np import residue_constants

def mrcread(fpath : str, iSlc = None):
    with mrcfile.mmap(fpath, permissive = True, mode = 'r') as mrc:
        data = mrc.data if iSlc is None or mrc.data.ndim == 2 else mrc.data[iSlc]
        return np.array(data, dtype = np.float64)


def get_atom_coords(atom_mask,atom_positions):
    """
    Extracts atom names and coordinates from a Protein instance.

    Args:
      prot: The protein to extract information from.

    Returns:
      A tuple containing:
      - A numpy array of shape (N, 3) with the coordinates of the atoms.
      - A numpy array of shape (N,) with the names of the atoms.
    """

    '''
    atom_coords = []
    for i in range(atom_positions.shape[0]):
        for pos, mask in zip(atom_positions[i], atom_mask[i]):
            if mask < 0.5:
                continue
            atom_coords.append(pos)
    atom_coords = torch.stack(atom_coords)
    '''
    # s = time.time()

    flat_positions = atom_positions.view(-1, 3)
    flat_mask = atom_mask.view(-1)
    atom_coord = flat_positions[flat_mask >= 0.5]

    # e = time.time()
    # print('mod get atom time',e-s)
    # print(torch.mean((atom_coords-atom_coord)**2))

    return atom_coord

def get_atom_weights(atom_mask):
    element_dict = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'P': 15.0, 'S': 16.0, 'W': 18.0, 'K': 19.0,
                'AU': 79.0}
    atom_types = residue_constants.atom_types

    atom_names = []

    for i in range(atom_mask.shape[0]):
        for atom_name, mask in zip(atom_types, atom_mask[i]):
            if mask < 0.5:
                continue
            atom_names.append(atom_name[0])
    atom_names = np.array(atom_names)

    def get_weight(atom_name):
        weight = element_dict.get(atom_name, np.nan)
        if np.isnan(weight):
            print(f"Unknown element: {atom_name}")
        return weight

    replace_func = np.vectorize(get_weight)
    atom_weights = replace_func(atom_names)
    
    return atom_weights


def compute_frc(proj, data, ctf, box_size = 240, max_freq=1):
    N = proj.shape[0]

    # Perform 2D Fourier Transform
    F1 = torch.fft.fftshift(torch.fft.fft2(proj), dim=(-2, -1))
    F1 = F1 * ctf
    F2 = torch.fft.fftshift(torch.fft.fft2(data), dim=(-2,-1))

    # Calculate the frequency grid in polar coordinates
    ny = box_size
    nx = box_size
    y, x = torch.meshgrid(torch.arange(-ny // 2, ny // 2), torch.arange(-nx // 2, nx // 2))
    
    freq_radius = torch.sqrt(x ** 2 + y ** 2).long()
    freq_radius = freq_radius.unsqueeze(0).unsqueeze(0).expand(N,1,box_size,box_size)
    # Number of frequency bins
    max_radius = int(box_size //2 * max_freq)
    frc = torch.zeros(N,max_radius)

    for radius in range(1, max_radius):
        mask = (freq_radius == radius)
        if mask.sum() == 0:
            continue

        # Sum over all angles θ for the current radius k
        P_k_theta = F1[mask].reshape(N,-1)
        I_k_theta = F2[mask].reshape(N,-1)

        # Calculate the dot product for this ring
        numerator = torch.sum(P_k_theta.real * I_k_theta.real + P_k_theta.imag * I_k_theta.imag,dim=-1)

        denominator = torch.sqrt(
            torch.sum(P_k_theta.real ** 2 + P_k_theta.imag ** 2,dim=-1) * torch.sum(I_k_theta.real ** 2 + I_k_theta.imag ** 2,dim=-1))

        frc[:,radius] = numerator / (denominator + 1e-8)  # Add small value to prevent division by zero

    # Normalize FRC
    # frc = 2 * frc / (1 + frc)

    frc = torch.sum(frc)/max_radius

    return frc

def compute_frc_simulate(proj, data, box_size = 240, max_freq=1):
    N = proj.shape[0]

    # Perform 2D Fourier Transform
    F1 = torch.fft.fftshift(torch.fft.fft2(proj), dim=(-2, -1))
    F2 = torch.fft.fftshift(torch.fft.fft2(data), dim=(-2,-1))

    # Calculate the frequency grid in polar coordinates
    ny = box_size
    nx = box_size
    y, x = torch.meshgrid(torch.arange(-ny // 2, ny // 2), torch.arange(-nx // 2, nx // 2))
    
    freq_radius = torch.sqrt(x ** 2 + y ** 2).long()
    freq_radius = freq_radius.unsqueeze(0).unsqueeze(0).expand(N,1,box_size,box_size)
    # Number of frequency bins
    max_radius = int(box_size //2 * max_freq)
    frc = torch.zeros(N,max_radius)

    for radius in range(1, max_radius):
        mask = (freq_radius == radius)
        if mask.sum() == 0:
            continue

        # Sum over all angles θ for the current radius k
        P_k_theta = F1[mask].reshape(N,-1)
        I_k_theta = F2[mask].reshape(N,-1)

        # Calculate the dot product for this ring
        numerator = torch.sum(P_k_theta.real * I_k_theta.real + P_k_theta.imag * I_k_theta.imag,dim=-1)

        denominator = torch.sqrt(
            torch.sum(P_k_theta.real ** 2 + P_k_theta.imag ** 2,dim=-1) * torch.sum(I_k_theta.real ** 2 + I_k_theta.imag ** 2,dim=-1))

        frc[:,radius] = numerator / (denominator + 1e-8)  # Add small value to prevent division by zero

    # Normalize FRC
    # frc = 2 * frc / (1 + frc)

    frc = torch.sum(frc)/max_radius

    return frc

def compute_frc_3d(vol1, vol2, box_size=240):
    # Perform 3D Fourier Transform
    F1 = torch.fft.fftshift(torch.fft.fftn(vol1), dim=(-3, -2, -1))
    F2 = torch.fft.fftshift(torch.fft.fftn(vol2), dim=(-3, -2, -1))

    # # Calculate the frequency grid in spherical coordinates
    ny, nx, nz = box_size, box_size, box_size
    z, y, x = torch.meshgrid(torch.arange(-nz // 2, nz // 2),
                             torch.arange(-ny // 2, ny // 2),
                             torch.arange(-nx // 2, nx // 2))

    freq_radius = torch.sqrt(x ** 2 + y ** 2 + z ** 2).long()
    freq_radius = freq_radius.unsqueeze(0).expand(vol1.shape[0], box_size, box_size, box_size)

    # Number of frequency bins
    max_radius = box_size // 2
    frc = torch.zeros(max_radius, dtype=torch.float32)
    frc[0] = 1
    for radius in range(max_radius):
        mask = (freq_radius == radius)
        if mask.sum() == 0:
            continue

        # Sum over all angles θ for the current radius k
        P_k_theta = F1[mask]
        I_k_theta = F2[mask]

        # Calculate the dot product for this ring
        numerator = torch.sum(P_k_theta.real * I_k_theta.real + P_k_theta.imag * I_k_theta.imag)

        denominator = torch.sqrt(
            torch.sum(P_k_theta.real ** 2 + P_k_theta.imag ** 2) * torch.sum(I_k_theta.real ** 2 + I_k_theta.imag ** 2))

        frc[radius] = numerator / (denominator + 1e-8)  # Add small value to prevent division by zero

    # Find resolution corresponding to FRC = 0.5 and 0.143
    resolution_0_5 = torch.where(frc >= 0.5)[0].max().item()
    resolution_0_143 = torch.where(frc >= 0.143)[0].max().item()
    resolution_0_5 = box_size/ (resolution_0_5+1)
    resolution_0_143 = box_size /(resolution_0_143+1)

    return frc, resolution_0_5, resolution_0_143

def discrete_radon_transform_3d(volume, rotation):
    volume = volume.expand(rotation.shape[0],1,volume.shape[-3],volume.shape[-2],volume.shape[-1])

    b = volume.shape[0]
    
    zeros = torch.zeros(b, 3, 1).to(volume.device)

    theta = torch.cat([rotation, zeros], dim=2)

    grid = F.affine_grid(theta, size=volume.shape)

    volume_rot = F.grid_sample(volume, grid, mode='bilinear')

    # volume_rot = volume_rot.permute(0, 1, 2, 3, 4)
    volume_rot = volume_rot.permute(0, 1, 3, 4, 2)
    proj = volume_rot.sum(dim=-1)
    
    return proj

def translation_2d(proj, trans):
    """
    Input:
        proj: Bx1xbsxbs tensor 
        trans: Bx2 tensor
    """
    
    b = trans.shape[0]
    
    eye = torch.eye(2).unsqueeze(0).repeat(b, 1, 1).to(proj.device)
    trans = trans.unsqueeze(-1)
    trans = trans * 2 / proj.shape[-1]
    theta = torch.cat([eye, trans], dim=2)

    grid = F.affine_grid(theta, size=proj.shape)
    proj_trans = F.grid_sample(proj, grid, mode='bicubic')
    
    return proj_trans

def compute_mse(proj, data, box_size = 240, max_freq=1):
    # Perform 2D Fourier Transform
    F1 = torch.fft.fftshift(torch.fft.fft2(proj), dim=(-2, -1))
    F2 = torch.fft.fftshift(torch.fft.fft2(data), dim=(-2,-1))
    # Calculate the frequency grid in polar coordinates
    ny = box_size
    nx = box_size
    y, x = torch.meshgrid(torch.arange(-ny // 2, ny // 2), torch.arange(-nx // 2, nx // 2))

    freq_radius = torch.sqrt(x ** 2 + y ** 2).long()
    freq_radius = freq_radius.unsqueeze(0).unsqueeze(0).expand(proj.shape[0],1,box_size,box_size)
    # Number of frequency bins
    max_radius = int(box_size //2 * max_freq)
    mask = (freq_radius <= max_radius)
    mask = mask.to(proj.device)
    Fproj = torch.fft.ifftshift(F1*mask, dim=(-2, -1))
    proj_mask = torch.fft.ifft2(Fproj)
    proj_mask = proj_mask.real

    Fdata = torch.fft.ifftshift(F2*mask, dim=(-2, -1))
    data_mask = torch.fft.ifft2(Fdata)
    data_mask = data_mask.real

    return F.mse_loss(data_mask,proj_mask,reduction='mean')
    dists = torch.cdist(x, y)
    return dists.min(1)[0].mean() + dists.min(0)[0].mean()


def analyze_storage_size(data_dict: dict):
   
    def get_size(obj):
        if isinstance(obj, torch.Tensor):
            return obj.nelement() * obj.element_size()
        elif isinstance(obj, dict):
            return sum(get_size(k) + get_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return sum(get_size(item) for item in obj)
        else:
            return sys.getsizeof(obj)

    def format_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    sizes = {key: get_size(value) for key, value in data_dict.items()}
    
    total_size = sum(sizes.values())

    if total_size == 0:
        print("The dictionary is empty or contains only empty objects.")
        return

    sorted_sizes = sorted(sizes.items(), key=lambda item: item[1], reverse=True)

    print("-" * 60)
    print(f"{'Key':<25} | {'Size':>15} | {'Percentage':>15}")
    print("-" * 60)
    for key, size in sorted_sizes:
        percentage = (size / total_size) * 100 if total_size > 0 else 0
        print(f"{key:<25} | {format_size(size):>15} | {percentage:14.2f}%")
    print("-" * 60)
    print(f"{'TOTAL':<25} | {format_size(total_size):>15} | {'100.00%':>15}")
    print("-" * 60)


def deep_clone(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: deep_clone(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deep_clone(item) for item in obj)
    else:
        return obj