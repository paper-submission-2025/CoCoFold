import time
import numpy as np
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ['BMP_DUPLICATE_LIB_OB'] = 'TRUE'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def sum_of_gaussians_2d_torch(centers, coef, sdev, maxrange, matrices, batch_size=1000,k=5):
    device = centers.device
    maxrange = torch.tensor(maxrange, device=device).to(coef.dtype)
    sdev = sdev.to(device)
    B, N, _ = centers.shape
    H, W = matrices.shape[1:]

    # Prepare the range for grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=centers.dtype, device=device),
        torch.arange(W, dtype=centers.dtype, device=device),
        indexing='ij'
    )
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(coef.dtype)  # Shape [1, 1, H, W]
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(coef.dtype)  # Shape [1, 1, H, W]

    centers = centers.unsqueeze(-1).unsqueeze(-1)  # Shape [B, N, 2, 1, 1]
    sdev = sdev.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape [1, N, 2, 1, 1]

    # Initialize density
    density = torch.zeros_like(matrices)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        centers_batch = centers[:, start:end, :, :, :]
        sdev_batch = sdev[:, start:end, :, :, :]
        coef_batch = coef[start:end].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        coef_batch = coef_batch.expand(B, end - start, 1, 1)

        # Calculate the bounds dynamically based on centers and sdev
        x_min = torch.clamp(
            (centers_batch[..., 0, :, :] - k * sdev_batch[..., 0, :, :]).min().floor().int(), min=0
        )
        x_max = torch.clamp(
            (centers_batch[..., 0, :, :] + k * sdev_batch[..., 0, :, :]).max().ceil().int(),
            max=grid_x.shape[-1],
        )
        y_min = torch.clamp(
            (centers_batch[..., 1, :, :] - k * sdev_batch[..., 1, :, :]).min().floor().int(), min=0
        )
        y_max = torch.clamp(
            (centers_batch[..., 1, :, :] + k * sdev_batch[..., 1, :, :]).max().ceil().int(),
            max=grid_y.shape[-2],
        )

        # Subset grid for the relevant region
        grid_x_sub = grid_x[:, :, y_min:y_max, x_min:x_max]
        grid_y_sub = grid_y[:, :, y_min:y_max, x_min:x_max]

        # Compute the distance and Gaussian function within the subset grid
        dy = (grid_y_sub - centers_batch[..., 1, :, :]) / (sdev_batch[..., 1, :, :] + 1e-8)
        dx = (grid_x_sub - centers_batch[..., 0, :, :]) / (sdev_batch[..., 0, :, :] + 1e-8)
        d2 = dy**2 + dx**2

        # Mask and calculate Gaussians
        gaussians = coef_batch * torch.exp(-0.5 * d2)

        # Accumulate density into the main matrices
        density[:, y_min:y_max, x_min:x_max] += torch.sum(gaussians, dim=1)

    matrices += density

    return matrices

def centers_rotation(coords, rotations):
    """
    Output:
        rotated_coords[..., :2]: BxNx2
    """

    coords = coords.transpose(1, 2)
    rotated_coords = torch.matmul(rotations, coords)
    rotated_coords = rotated_coords.transpose(1, 2)
    return rotated_coords[..., :2]


def translation_2d(proj, trans, box_size, apix,density_center):
    """
    Inputs:
        proj: Bx1xbsxbs tensor
        trans: Bx2 tensor
    Output:
        proj_trans: Bx1xbsxbs tensor
    """
    B, _, H, W = proj.shape
    b = trans.shape[0]

    y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W))
    y_indices = y_indices.flatten().to(trans.dtype)
    x_indices = x_indices.flatten().to(trans.dtype)

    y_indices = y_indices.to(proj.device)
    x_indices = x_indices.to(proj.device)
    flat_images = proj.view(B, -1)
    # Compute the total intensity for each image
    total_intensity = flat_images.sum(dim=1)
    #if (total_intensity == 0).any():
    #    print("Warning: total_intensity contains zero values")

    flat_images /= total_intensity.view(B,1)

    # Compute the weighted sum of coordinates
    x_weighted_sum = (flat_images * x_indices).sum(dim=1)
    y_weighted_sum = (flat_images * y_indices).sum(dim=1)

    # Compute the centroids
    centroid_x = x_weighted_sum
    centroid_y = y_weighted_sum

    centroids = torch.stack([centroid_x, centroid_y], dim=1)

    eye = torch.eye(2).unsqueeze(0).repeat(b, 1, 1).to(proj.device).to(proj.dtype)
    # trans *= apix

    trans -= (density_center - centroids)

    # trans -= (box_size / 2)

    trans = trans.unsqueeze(-1)
    trans = trans * 2 / box_size
    theta = torch.cat([eye, trans], dim=2)

    grid = F.affine_grid(theta, size=proj.shape)

    proj_trans = F.grid_sample(proj, grid, mode='bicubic')


    return proj_trans


def pdb2img(atoms_coord,
            resolution,
            atoms_weight,
            rotation,
            trans,
            density_center,
            box_size=256,
            cutoff_range=5,  # in standard deviations
            sigma_factor=1 / (np.pi * np.sqrt(2)),  # standard deviation / resolution)
            apix=1,
            sdevs = None,
            masks = None,
            affine_matricies = None,
            is_multimer=False,
            ):
    """
    Projection of 3D GMM without molmap
    Inputs:
        atoms_coord: BxNx3 tensor
        resolution: float
        atoms_weight: Nx1 tensor
        rotation: Bx2x3 tensor, only first two rows are needed since z will be integrated
        trans: Bx2 tensor
        box_size: int
        cutoff_range: int
        sigma_factor: float
        is_multimer: bool, if True, enables parameter sharing for multimers
    Output:
        img: Bx1xbsxbs
    """

    if is_multimer:
        num_total_atoms = atoms_coord.shape[1]
        num_param_atoms = atoms_weight.shape[0]

        if num_total_atoms > num_param_atoms and num_total_atoms % num_param_atoms == 0:
            # print("INFO: Multimer mode enabled. Reusing atom weights and sdevs for each subunit.")
            num_subunits = num_total_atoms // num_param_atoms

            atoms_weight = atoms_weight.repeat(num_subunits)
            if sdevs is not None:
                sdevs = sdevs.repeat(num_subunits, 1)

        elif num_total_atoms != num_param_atoms:
            raise ValueError(
                f"Multimer mode is enabled, but atom coordinate count ({num_total_atoms}) "
                f"is not a clean multiple of the parameter count ({num_param_atoms})."
            )
        
    # get parameters for GMM
    _, N, _ = atoms_coord.shape
    B,_,_ = rotation.shape
    pad = 3 * resolution
    step = (1. / 3) * resolution
    sdev = resolution * sigma_factor

    for i in range(len(masks)):
        affine_matrix = torch.from_numpy(affine_matricies[i]).to(atoms_coord.device).to(atoms_coord.dtype)
        mask_indices = masks[i]
        atoms_coord[:,mask_indices,:] = torch.matmul(atoms_coord[:,mask_indices ,:], affine_matrix[:,:3].T) + affine_matrix[:,3]

    atoms_coord = atoms_coord/apix
    # rotation
    proj_rot = centers_rotation(atoms_coord, rotation)

    # transform xy to the grid ij and make it into the box
    origin = torch.min(proj_rot, dim=1, keepdim=True).values
    proj_rot[..., 0] = proj_rot[..., 0] / step - origin[..., 0] / step
    proj_rot[..., 1] = proj_rot[..., 1] / step - origin[..., 1] / step
    proj_rot += pad

    # projection
    img = sum_of_gaussians_2d_torch(centers=proj_rot.to(proj_rot.device), coef=atoms_weight.to(proj_rot.device), sdev=sdevs.to(proj_rot.device), maxrange=cutoff_range,matrices=torch.zeros(B, box_size, box_size).to(proj_rot.device).to(proj_rot.dtype))


    normalization = torch.pow(2 * torch.pi, torch.tensor(-1)) * torch.pow(sdev, torch.tensor(-2))
    img *= normalization

    # scale to integrate z
    img /= step
    img = img.unsqueeze(1)
    # move it to the center first and then modify the translation
    img = translation_2d(img, trans / step, box_size, apix,density_center.to(img.device))


    return img

def sum_of_gaussians_3d_torch(centers, coef, sdev, maxrange, matrices, batch_size=1):
    device = centers.device
    maxrange = torch.tensor(maxrange, device=device)
    sdev = sdev.to(device)
    B, N, _ = centers.shape
    D, H, W = matrices.shape[1:]
    
    density = torch.zeros_like(matrices)
    for c in range(N):

        sd = sdev[c]
        cf = coef[c]
        center = centers[0,c,...]

        ijk_min = torch.ceil(center - maxrange * sd).to(torch.int)
        ijk_max = torch.floor(center + maxrange * sd).to(torch.int)
        ijk_min[ijk_min<=0] = 0
        ijk_min[ijk_max>=D] = D-1
        ijk_max[ijk_max<=0] = 0
        ijk_max[ijk_max>=D] = D-1
        z = torch.arange(ijk_min[2], ijk_max[2] + 1).to(device)
        y = torch.arange(ijk_min[1], ijk_max[1] + 1).to(device)
        x = torch.arange(ijk_min[0], ijk_max[0] + 1).to(device)
        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

        dz = (Z - center[2]) / sd[2]
        dy = (Y - center[1]) / sd[1]
        dx = (X - center[0]) / sd[0]

        d2 = dz ** 2 + dy ** 2 + dx ** 2
        gauss = cf * torch.exp(-0.5 * d2)

        density[0, ijk_min[2]:ijk_max[2] + 1, ijk_min[1]:ijk_max[1] + 1, ijk_min[0]:ijk_max[0] + 1] += gauss
    matrices+=density

    return matrices


def translation_center(proj, box_size):
    """
    Inputs:
        proj: Bx1xDxHxW tensor
        box_size: int (size of the box)
    Output:
        proj_trans: Bx1xDxHxW tensor
    """
    B, _, D, H, W = proj.shape

    z_indices, y_indices, x_indices = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij')
    z_indices = z_indices.flatten().float()
    y_indices = y_indices.flatten().float()
    x_indices = x_indices.flatten().float()

    z_indices = z_indices.to(proj.device)
    y_indices = y_indices.to(proj.device)
    x_indices = x_indices.to(proj.device)
    flat_images = proj.view(B, -1)
    # Compute the total intensity for each image
    total_intensity = flat_images.sum(dim=1)
    if (total_intensity == 0).any():
        print("Warning: total_intensity contains zero values")

    # Compute the weighted sum of coordinates
    x_weighted_sum = (flat_images * x_indices).sum(dim=1)
    y_weighted_sum = (flat_images * y_indices).sum(dim=1)
    z_weighted_sum = (flat_images * z_indices).sum(dim=1)

    # Compute the centroids
    centroid_x = x_weighted_sum / (total_intensity + 1e-10)
    centroid_y = y_weighted_sum / (total_intensity + 1e-10)
    centroid_z = z_weighted_sum / (total_intensity + 1e-10)

    centroids = torch.stack([centroid_x, centroid_y, centroid_z], dim=1)

    eye = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(proj.device)

    trans = -(box_size / 2 - centroids)

    trans = trans.unsqueeze(-1)
    trans = trans * 2 / box_size
    theta = torch.cat([eye, trans], dim=2)

    # Create the affine grid
    grid = F.affine_grid(theta, size=proj.shape, align_corners=True)
    # Sample the original image with the grid to get the translated image
    proj_trans = F.grid_sample(proj, grid, mode='nearest', align_corners=True)

    return proj_trans

def pdb2mrc(atoms_coord,
            resolution,
            atoms_weight,
            rotation=None,
            box_size=256,
            sdevs = None,
            affine_matrix1 = None,
            cutoff_range=5,  # in standard deviations
            sigma_factor=1 / (np.pi * np.sqrt(2)),  # standard deviation / resolution)
            apix=1,
            ):
    """
    Projection of 3D GMM without molmap
    Inputs:
        atoms_coord: BxNx3 tensor
        resolution: float
        atoms_wight: Nx1 tensor
        rotation: Bx2x3 tensor, only first two rows are needed since z will be integrated
        trans: Bx2 tensor
        box_size: int
        cutoff_range: int
        sigma_factor: float
    Output:
        img: Bx1xbsxbs
    """

    # get parameters for GMM
    B, N, _ = atoms_coord.shape

    pad = 3 * resolution
    step = (1. / 3) * resolution
    sdev = resolution * sigma_factor
    if sdevs is None:
        sdevs = torch.zeros(N, 3)
        sdevs += sdev / step
    else:
        temp = torch.zeros(N,3).to(sdevs.device)
        temp += sdev/step
        temp[:,:2] = sdevs
        temp[:,2] = torch.mean(sdevs,dim=1)
        sdevs = temp

    # rotation
    if rotation is not None:
        atoms_coord = atoms_coord.transpose(1, 2)
        rotated_coords = torch.matmul(rotation, atoms_coord)
        rotated_coords = rotated_coords.transpose(1, 2)
    else:
        rotated_coords = atoms_coord 
    rotated_coords/=apix
    # print('atom coord xyz', torch.max(atoms_coord),torch.min(atoms_coord))
    # transform xy to the grid ij and make it into the box
    origin = torch.min(rotated_coords, dim=1, keepdim=True).values
    rotated_coords[..., 0] = rotated_coords[..., 0] - origin[..., 0]
    rotated_coords[..., 1] = rotated_coords[..., 1] - origin[..., 1]
    rotated_coords[..., 2] = rotated_coords[..., 2] - origin[..., 2]
    rotated_coords += pad

    grid = sum_of_gaussians_3d_torch(centers=rotated_coords, coef=atoms_weight, sdev=sdevs, maxrange=cutoff_range,
                                    matrices=torch.zeros(B, box_size, box_size, box_size).to(rotated_coords.device))

    normalization = torch.pow(2 * torch.pi, torch.tensor(-1.5)) * torch.pow(sdev, torch.tensor(-3))
    grid *= normalization

    # move it to the center first and then modify the translation
    grid = translation_center(grid.unsqueeze(1), box_size)

    return grid
