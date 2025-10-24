import argparse
from particledataset import ParticleDataset
from cryo_model import LoRADecoder
import torch
import numpy as np
from utils import get_atom_coords, get_atom_weights
from pts2img import pdb2mrc

from heads import AuxiliaryHeads
from feats import atom14_to_atom37
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.script_utils import prep_output
from openfold.np import protein
from structure_module import StructureModule

def main(args):


    device0 = args.device0
    device1 = args.device1

    structure_path = str(args.structure_dir)
    loaded_data = torch.load(structure_path, map_location=device0,weights_only=False)
    outputs = loaded_data['outputs']

    feats = loaded_data['feats']
    inplace_safe = loaded_data['inplace_safe']
    type = loaded_data['type']
    offload_inference = loaded_data['offload_inference']

    state = loaded_data['model_state']
    init_params = loaded_data['init_params']

    structure_module = StructureModule(
        c_s=init_params['c_s'],
        c_z=init_params['c_z'],
        c_ipa=init_params['c_ipa'],
        c_resnet=init_params['c_resnet'],
        no_heads_ipa=init_params['no_heads_ipa'],
        no_qk_points=init_params['no_qk_points'],
        no_v_points=init_params['no_v_points'],
        dropout_rate=init_params['dropout_rate'],
        no_blocks=init_params['no_blocks'],
        no_transition_layers=init_params['no_transition_layers'],
        no_resnet_blocks=init_params['no_resnet_blocks'],
        no_angles=init_params['no_angles'],
        trans_scale_factor=init_params['trans_scale_factor'],
        epsilon=init_params['epsilon'],
        inf=init_params['inf'],
        is_multimer=init_params['is_multimer'],
    ).to(device0)
    structure_module.load_state_dict(state)

    del loaded_data

    loaded_data = torch.load(str(args.head_dir),weights_only=False,map_location=device0)
    config_heads = loaded_data['heads']
    aux_heads = AuxiliaryHeads(config_heads).to(device0)
    try:
        aux_heads.load_state_dict(loaded_data['heads_state'])
    except:
        aux_heads.load_state_dict(loaded_data['model_state'])
    outputs['asym_id'] = loaded_data['asym_id']
    outputs["num_recycles"] = loaded_data['num_recycles']
    del loaded_data

    outputs["final_atom_mask"] = feats["atom37_atom_exists"]
    atom_weights = get_atom_weights(outputs["final_atom_mask"])
    atom_weights = torch.from_numpy(atom_weights).to(torch.float).to(device1)

    structure_module.to(device0)


    dataset = ParticleDataset(str(args.star_data_dir),str(args.mrc_data_dir),1)

    print(f'The dataset contains {len(dataset)} particles.')
    box_size = int(args.boxsize)
    z_dim = 10
    conf_regressor_params = {'z_dim': z_dim, 'variational': False, 'std_z_init': 0.1, 'init':str(args.z_init)}

    model = LoRADecoder(n_particles_dataset=len(dataset), conf_regressor_params=conf_regressor_params, sonly = args.sonly).to(device1)
    model = model.float()
    structure_module = structure_module.float()

    loaded_data = torch.load(str(args.model_dir), map_location=device1,weights_only=False)

    model_state_dict = model.state_dict()
    partial_state_dict = {k: v for k, v in loaded_data['model_state'].items() if k in model_state_dict}

    model_state_dict.update(partial_state_dict)
    model.load_state_dict(model_state_dict)

    structure_module.eval()
    model.eval()
    with torch.no_grad():

        hidden = model.forward_s()
        hidden = hidden.view(384,192,3).to(device0)
        if args.af:
            hidden = None
            print('Alphafold predict!!!!!')

        outputs["sm"] = structure_module(
            outputs,
            feats["aatype"],
            mask = feats["seq_mask"].to(dtype=type).to(torch.float),
            h = hidden,
            inplace_safe=inplace_safe,
            _offload_inference=offload_inference,
        )
            
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        if 'rigids' in outputs['sm']:
            outputs['sm']['rigids'] = torch.tensor([1.])
        if "asym_id" in feats:
            outputs["asym_id"] = feats["asym_id"]


        sdevs = loaded_data['sdevs']
        atom_weights = loaded_data['atom_weights']

        atom_coord = get_atom_coords(outputs["final_atom_mask"],outputs["final_atom_positions"])
        atom_coord = atom_coord.unsqueeze(0).to(device1)


        if args.affine_matrices_csv is not None:
            affine_raw = np.loadtxt(args.affine_matrices_csv, delimiter=",")
            if affine_raw.shape[1] != 4 or affine_raw.shape[0] % 3 != 0:
                raise ValueError("CSV file must have 4 columns and rows as multiples of 3.")
            affine_matrices = [affine_raw[i:i+3, :] for i in range(0, affine_raw.shape[0], 3)]
        else:
            raise ValueError("Please provide --affine_matrices_csv to load affine matrices.")
        print('affine_matrices',affine_matrices)

        if args.mask_split_csv is not None:
            split_points = np.loadtxt(args.mask_split_csv, dtype=int)
            if np.ndim(split_points) == 0:
                split_points = [int(split_points)]
            else:
                split_points = split_points.tolist()
            
            split_points.append(len(atom_weights))
            masks = []
            for i in range(len(split_points) - 1):
                start, end = split_points[i], split_points[i + 1]
                mask = np.zeros(len(atom_weights), dtype=bool)
                mask[start:end] = True
                masks.append(mask)
        else:
            raise ValueError("Please provide --mask_split_csv to load masks.")
        for k in range(len(masks)):
            affine_matrix = torch.from_numpy(affine_matrices[k]).to(atom_coord.device).to(torch.float)
        
            mask_indices = masks[k]

            atom_coord[:,mask_indices,:] = torch.matmul(atom_coord[:,mask_indices ,:], affine_matrix[:,:3].T) + affine_matrix[:,3]



        sdevs_np = sdevs.cpu().numpy()

        np.savetxt(args.output_dir+'_sdevs.csv', sdevs_np, delimiter=',')

        sdevs = torch.abs(sdevs)
        predict = pdb2mrc(atom_coord,float(args.resolution),atom_weights,rotation=None,sdevs=sdevs,box_size=box_size,apix=float(args.apix))
        predict = predict.cpu().numpy().reshape(box_size,box_size,box_size)

        import mrcfile
        with mrcfile.new(args.output_dir+'.mrc', overwrite=True) as mrc:

            mrc.set_data(np.asarray(predict, dtype=np.float32))
            mrc.voxel_size = float(args.apix)


        outputs.update(aux_heads(outputs))

        out = tensor_tree_map(lambda x: np.array(x.to(torch.float).cpu()), outputs)
        protein_data = torch.load(str(args.prot_dir),weights_only=False,map_location=device0)
        
        unrelaxed_protein = prep_output(
            out,
            protein_data['processed_feature_dict'],
            protein_data['feature_dict'],
            protein_data['feature_processor'],
            protein_data['config_preset'],
            protein_data['multimer_ri_gap'],
            protein_data['subtract_plddt']
        )
        with open(str(args.output_dir)+'.pdb', 'w') as fp:
            fp.write(protein.to_pdb(unrelaxed_protein))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--star_data_dir", type=str,
        help="""Path to the STAR file containing the particle data"""
    )
    parser.add_argument(
        "--mrc_data_dir", type=str,
        help="""Path to the MRC file containing the particle data"""
    )
    parser.add_argument(
        "--structure_dir", type=str,
        help="""Path to the structure module file containing the initial structure module state"""
    )
    parser.add_argument(
        "--model_dir",type=str, default=None,
        help="""Path to the trained GMM model file"""
    )
    parser.add_argument(
        "--head_dir", type=str,
        help="""Path to the auxiliary heads file containing the initial auxiliary heads state"""
    )
    parser.add_argument(
        "--prot_dir",type=str,
        help="""Path to the protein data file containing the initial protein data state"""
    )
    parser.add_argument(
        "--device0", type=str, default='cuda:0',
        help="""Device to run the structure model on, e.g., 'cuda:0' or 'cuda:1'"""
    )
    parser.add_argument(
        "--device1", type=str, default='cuda:0',
        help="""Device to run the GMM model on, e.g., 'cuda:0' or 'cuda:1'"""
    )
    parser.add_argument(
        "--output_dir",type=str,
        help="""Directory to save the output files, including the PDB and GMM MRC files"""
    )
    parser.add_argument(
        "--z_init",default='normal',
        help="""ignore this argument, it is not used in the current implementation"""
    )
    parser.add_argument(
        "--sonly",action="store_true", default=False,
        help="""Just keep true"""
    )
    parser.add_argument(
        "--af",action="store_true", default=False,
        help="""Use Alphafold for prediction if the initial saved structure model is given"""
    )
    parser.add_argument(
        "--apix", type=float, default=1.0,
        help="""Pixel size in Angstroms, default is 1.0"""
    )
    parser.add_argument(
        "--boxsize", type=int, default=256,
        help="""Size of the box for the particle data, default is 256"""
    )
    parser.add_argument(
        "--resolution",type=float, default=3.0,
        help="""Initial resolution for the GMM model, ignore it if you use the trained model"""
    )
    parser.add_argument(
    "--affine_matrices_csv", type=str, default=None,
    help="""Path to a CSV file containing affine matrices (each matrix is 3 rows x 4 cols)"""
    )
    parser.add_argument(
    "--mask_split_csv", type=str, default=None,
    help="""Path to CSV file that defines split indices for masks. Each line is a split point (e.g., 0, 12, 28, ...)"""
    )
    args = parser.parse_args()
    main(args)




