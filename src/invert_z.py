import numpy as np
import mrcfile
import torch
import torch.nn.functional as F
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


def load_mrc(filename):
    with mrcfile.open(filename, permissive=True) as mrc:
        data = mrc.data.copy()
        voxel_size = mrc.voxel_size
    return data, voxel_size


def save_mrc(data, filename, voxel_size):
    with mrcfile.new(filename, overwrite=True) as mrc:

        mrc.set_data(data)
        mrc.voxel_size = voxel_size


def apply_transformation(data, transformation_matrix):

    # Calculate the inverse of the transformation matrix
    inverse_matrix = np.linalg.inv(transformation_matrix)
    # inverse_matrix = transformation_matrix
    volume = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    rotation = torch.tensor(inverse_matrix, dtype=torch.float32).unsqueeze(0)

    b = volume.shape[0]

    zeros = torch.zeros(b, 3, 1).to(volume.device)

    theta = torch.cat([rotation, zeros], dim=2)

    grid = F.affine_grid(theta, size=volume.shape)

    volume_rot = F.grid_sample(volume, grid, mode='bilinear')

    return volume_rot.numpy()[0, 0, ...]


def main(args):
    # Load the MRC file
    input_mrc = str(args.input_mrc)
    data, voxel_size = load_mrc(input_mrc)

    # Define the rotation matrix and translation vector (example values)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]
                                ])

    # Create the transformation matrix
    transformation_matrix = np.eye(3)
    transformation_matrix[:3, :3] = np.linalg.inv(rotation_matrix)

    # Apply the transformation to the MRC data
    transformed_data = apply_transformation(data, transformation_matrix)

    # Save the transformed MRC data to a new file
    output_mrc = str(args.output_mrc) if args.output_mrc else 'transformed_volume.mrc'
    save_mrc(transformed_data, output_mrc, voxel_size)

    print("Transformation applied and saved to", output_mrc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_mrc", type=str,
        help="""Path to the input MRC file containing the volume data"""
    )
    parser.add_argument(
        "--output_mrc", type=str,
        help="""Path to save the transformed MRC file"""
    )
    args = parser.parse_args()
    print(args)
    main(args)