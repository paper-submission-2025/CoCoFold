import argparse
from particledataset import ParticleDataset
from torch.utils.data import DataLoader
from cryo_model import LoRADecoder
import torch
import numpy as np
from ctf import compute_ctf
from utils import get_atom_coords, get_atom_weights,compute_frc, discrete_radon_transform_3d, translation_2d, compute_frc_simulate, deep_clone
from pts2img import pdb2img
import time
import pandas as pd
import sys

from heads import AuxiliaryHeads
from feats import atom14_to_atom37
from structure_module import StructureModule


def main(args):
        
    torch.manual_seed(0)
    np.random.seed(0)
    box_size = int(args.boxsize)

    device0 = args.device0
    device1 = args.device1
    apix =float(args.apix)
    particle_sign = int(args.particle_sign)
    structure_path = str(args.structure_dir)
    loaded_data = torch.load(structure_path,weights_only=False,map_location=device0)
    outputs = loaded_data['outputs']
    config = loaded_data['config']
    feats = loaded_data['feats']
    inplace_safe = loaded_data['inplace_safe']
    dtype = loaded_data['type']
    offload_inference = loaded_data['offload_inference']
    m = loaded_data['m']
    x_prev = loaded_data['x_prev']
    seq_mask = loaded_data['seq_mask']
    feats["seq_mask"] = feats["seq_mask"].to(dtype=dtype).to(torch.float)
    state = loaded_data['model_state']
    init_params = loaded_data['init_params']

    if args.density_center is None:
        density_center =  torch.tensor([box_size/2,box_size/2], dtype=torch.float32).to(device0)
    else:
        density_center = torch.tensor(args.density_center, dtype=torch.float32).to(device0)
        density_center = density_center.unsqueeze(0)
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
    )
    

    structure_module = structure_module.to(device0)
    structure_module.load_state_dict(state)

    del loaded_data

    loaded_data = torch.load(str(args.head_dir),weights_only=False,map_location=device0)
    config_heads = loaded_data['heads']
    aux_heads = AuxiliaryHeads(config_heads).to(device0)
    aux_heads.load_state_dict(loaded_data['heads_state'])

    outputs['asym_id'] = loaded_data['asym_id']
    outputs["num_recycles"] = loaded_data['num_recycles']
    del loaded_data

    hidden = torch.zeros(384,192,3).to(torch.float)

    structure_module = structure_module.float()
    aux_heads = aux_heads.float()
    for key in outputs:
        if isinstance(outputs[key], torch.Tensor):
            outputs[key] = outputs[key].float()

    with torch.no_grad():
        outputs["sm"] = structure_module(
                            outputs,
                            feats["aatype"],
                            mask = feats["seq_mask"].to(dtype=dtype).to(torch.float),
                            h = hidden.to(device0),
                            inplace_safe=inplace_safe,
                            _offload_inference=offload_inference,
                        )
                    
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )

        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]

        outputs.update(aux_heads(outputs))
        atom_coord = get_atom_coords(outputs["final_atom_mask"],outputs["final_atom_positions"])
        atom_weights = get_atom_weights(outputs["final_atom_mask"])
        atom_weights = torch.from_numpy(atom_weights).to(torch.float).to(device1)

        resolution = torch.tensor([float(args.resolution)]).to(torch.float).to(device1)
        sdevs = torch.zeros(atom_weights.shape[0], 2)
        sdevs += 3/(np.pi * np.sqrt(2))
        sdevs = sdevs.to(torch.float).to(device1)

    
    if args.invert_z :
        invert_z = np.array([[1,0,0],[0,1,0],[0,0,-1]]).reshape(3,3)
    else:
        invert_z = None

    if args.affine_matrices_csv is not None:
        affine_raw = np.loadtxt(args.affine_matrices_csv, delimiter=",")
        if affine_raw.shape[1] != 4 or affine_raw.shape[0] % 3 != 0:
            raise ValueError("CSV file must have 4 columns and rows as multiples of 3.")
        affine_matrices = [affine_raw[i:i+3, :] for i in range(0, affine_raw.shape[0], 3)]
    else:
        raise ValueError("Please provide --affine_matrices_csv to load affine matrices.")
    print('affine_matrices',affine_matrices)
    '''
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
    '''
    if args.mask_split_csv is not None:
        split_points = np.loadtxt(args.mask_split_csv, dtype=int)
        if np.ndim(split_points) == 0:
            split_points = [int(split_points)]
        else:
            split_points = split_points.tolist()
        monomer_atom_count = len(atom_weights) 

        if args.is_multimer:
            num_subunits = len(split_points)
            total_atom_count = monomer_atom_count * num_subunits
        else:
            num_subunits = 1
            total_atom_count = monomer_atom_count

        masks = []
        split_points.append(total_atom_count)
        
        for i in range(len(split_points)-1): 
            start_mono, end_mono = split_points[i], split_points[i+1]
            total_mask = np.zeros(total_atom_count, dtype=bool)
            total_mask[start_mono : end_mono] = True
            masks.append(total_mask)
    
    else:
        raise ValueError("Please provide --mask_split_csv to load masks.")

    batch_size = int(args.batch_size)
    mini_batch_size = int(args.mini_batch_size)
    print('batch_size',batch_size,'mini_batch_size',mini_batch_size)
    dataset = ParticleDataset(str(args.star_data_dir),str(args.mrc_data_dir),apix,transR=invert_z,norm = args.norm)

    print(f'The dataset contains {len(dataset)} particles.')

    z_dim = 10
    epochs = int(args.epochs)

    conf_regressor_params = {'z_dim': z_dim, 'variational': False, 'std_z_init': 0.1, 'init':str(args.z_init)}

    model = LoRADecoder(n_particles_dataset=len(dataset), conf_regressor_params=conf_regressor_params, n_layers=3,sonly=args.sonly).to(device1)
    model = model.float()

    if args.input_model_dir is not None:
        loaded_data = torch.load(str(args.input_model_dir),weights_only=False,map_location=device0)
        model_state_dict = model.state_dict()
        partial_state_dict = {k: v for k, v in loaded_data['model_state'].items() if k in model_state_dict and k != 'conf_table.table_conf'}
        model_state_dict.update(partial_state_dict)
        model.load_state_dict(model_state_dict)

        atom_weights = loaded_data['atom_weights'] 
        sdevs = loaded_data['sdevs']


    print(model)

    lr_rate1 = 1e-3
    lr_rate2 = 1e-4
    lr_sdevs = 5e-3
    lr_atoms_weights = 1e-2
    
    limit = [0.1,0.8,1,20]
    print('gmm bound',limit)

    params = list(model.named_parameters())

    table_conf_params = [p for n, p in params if 'conf_table' in n]
    other_params = [p for n, p in params if 'conf_table' not in n]
    structure_params = list(structure_module.parameters())


    optimizer = torch.optim.Adam([ {'params': other_params, 'lr': lr_rate1}, 
                                {'params': table_conf_params, 'lr': 1e-2}])
    optimizer.add_param_group({'params':structure_params,'lr':lr_rate2})

   
    if args.learn_weight:
        atom_weights.requires_grad = True
        optimizer.add_param_group({'params': atom_weights, 'lr': lr_atoms_weights})
        print('lr_atom_weight',lr_atoms_weights)

    if args.learn_width:
        sdevs.requires_grad = True
        optimizer.add_param_group({'params': sdevs, 'lr': lr_sdevs})
        print('lr_sdevs',lr_sdevs)

    flag_map = False
    if args.em_map is not None:
        start = time.time()
        em_map = args.em_map
        import mrcfile
        with mrcfile.open(em_map) as mrc:
            em_map =  np.array(mrc.data)
        em_map = torch.from_numpy(em_map).to(device1).to(torch.float)
        em_map = em_map.reshape(1,1,em_map.shape[0],em_map.shape[1],em_map.shape[2])
        flag_map = True

    step_size=100
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.89)
    print('scheduler step',step_size)
    print('lr_rate',lr_rate1)
    print('stru rate',lr_rate2)

    freqs = (
            np.stack(
                np.meshgrid(
                    np.linspace(-0.5, 0.5, box_size, endpoint=False),
                    np.linspace(-0.5, 0.5, box_size, endpoint=False),
                ),
                -1,
            )
            / apix
            )
    
    freqs = freqs.reshape(-1, 2)
    freqs = torch.from_numpy(freqs).unsqueeze(0).to(torch.float)
    apix = torch.tensor([float(args.apix)]).to(torch.float).to(device1)


    for param in structure_module.parameters():
        param.requires_grad = True

    for name, param in structure_module.named_parameters():
        print(f"{name}: {param.device}, {param.dtype}, {param.requires_grad}")
    if args.input_model_dir is not None:
        loaded_opt_state = loaded_data['opt_state']
        current_opt_state = optimizer.state_dict()
        if len(loaded_opt_state['param_groups']) == len(current_opt_state['param_groups']):
            optimizer.load_state_dict(loaded_opt_state)
        else:
            print("⚠️ Skip optimizer param_groups loading")
        del loaded_data

    print('training')
    max_freq = torch.tensor([float(args.max_freq)]).to(torch.float)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for num, batch in enumerate(dataloader):
            start = time.time()
            data,para,trans,R,R2,index = batch
            data = data.unsqueeze(1)
            data = data.to(device1).to(torch.float)
            trans = trans.to(device1).to(torch.float)
            R = R.to(device1).to(torch.float)
            R2 = R2.to(device1).to(torch.float)
            para = para.to(torch.float)
            voltage, defocusU, defocusV,astigmatism, Cs, amplitude, phase_shift, pixel_size = para.T
            voltage = voltage.unsqueeze(1)
            defocusU = defocusU.unsqueeze(1)
            defocusV = defocusV.unsqueeze(1)
            astigmatism = astigmatism.unsqueeze(1)
            Cs = Cs.unsqueeze(1)
            amplitude = amplitude.unsqueeze(1)
            phase_shift = phase_shift.unsqueeze(1)
            #s = time.time()
            ctf = compute_ctf(
                freqs=freqs,
                dfu=defocusU,
                dfv=defocusV,
                dfang=astigmatism,
                volt=voltage,
                cs=Cs,
                w=amplitude,
                phase_shift=phase_shift,
                bfactor=None 
            )

            ctf = ctf.reshape(ctf.shape[0],1,box_size,box_size).to(device1)
            ctf = ctf.to(torch.float)

            losses = 0
            penalties = 0
            for num_start in range(0, data.shape[0], mini_batch_size):
                num_end = min(num_start + mini_batch_size,  data.shape[0])
                hidden = model.forward_s()
                hidden = hidden.view(384,192,3).to(device0)

                outputs["sm"] = structure_module(
                    outputs,
                    feats["aatype"],
                    mask = feats["seq_mask"],
                    h = hidden,
                    inplace_safe=inplace_safe,
                    _offload_inference=offload_inference,
                )

                outputs["final_atom_positions"] = atom14_to_atom37(
                    outputs["sm"]["positions"][-1], feats
                )
                outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
                atom_coord = get_atom_coords(outputs["final_atom_mask"],outputs["final_atom_positions"])
                atom_coord = atom_coord.unsqueeze(0).to(device1)

                del hidden

                if args.is_multimer:
                    num_subunits = len(masks)
                    if num_subunits > 0:
                        atom_coord = atom_coord.repeat(1, num_subunits, 1)
                    sdevs_for_penalty = sdevs.repeat(num_subunits, 1)          
                    atom_weights_for_penalty = atom_weights.repeat(num_subunits)
                else:
                    sdevs_for_penalty = sdevs
                    atom_weights_for_penalty = atom_weights

                proj = pdb2img(atom_coord,
                resolution,
                atom_weights,
                R[num_start:num_end],
                trans[num_start:num_end],
                density_center=density_center,
                box_size=box_size,
                cutoff_range=5,  # in standard deviations
                sigma_factor=1 / (np.pi * np.sqrt(2)),  # standard deviation / resolution)
                apix = apix,
                sdevs = sdevs,
                masks = masks,
                affine_matricies = affine_matrices,
                is_multimer=args.is_multimer
                )

                proj *= particle_sign


                # FRC support image size for powers of 2 only using torch.float

                if flag_map:

                    em_map_proj = discrete_radon_transform_3d(em_map,R2[num_start:num_end])
                    em_map_proj = translation_2d(em_map_proj, trans[num_start:num_end])
                    em_map_proj = em_map_proj.reshape(proj.shape[0],proj.shape[1],proj.shape[2],proj.shape[3])
                    em_map_proj *= particle_sign

                    loss = -compute_frc_simulate(proj=proj.to(torch.float),data=em_map_proj.to(torch.float),box_size=box_size,max_freq = max_freq)/data.shape[0]
                    losses += loss

                else:
                    loss = -compute_frc(proj=proj.to(torch.float),data=data[num_start:num_end].to(torch.float),ctf=ctf[num_start:num_end].to(torch.float),box_size=box_size,max_freq = max_freq)/data.shape[0]
                    losses += loss


                for i in range(len(masks)):
                    mask_i = masks[i]
                    penalty = torch.mean((torch.relu(sdevs_for_penalty[mask_i,:].to(torch.float) - limit[1])) + 
                                            torch.relu(limit[0] - sdevs_for_penalty[mask_i,:].to(torch.float))) + torch.mean((torch.relu(atom_weights_for_penalty[mask_i].to(torch.float) - limit[3])) + torch.relu(limit[2] - atom_weights_for_penalty[mask_i].to(torch.float)))

                    penalties += penalty.to('cpu')
                    loss += penalty.to('cpu')

                loss.backward()


            optimizer.step()
            print("peak memory:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
            optimizer.zero_grad()

            end = time.time()
            print('epoch',epoch,' batch num',num)
            print('loss',losses)
            print('penalty',penalties)
            print('time',end-start)

            sys.stdout.flush()  

        scheduler.step()

        structure_path = str(args.output_structure_dir)+str(epoch+1)+'.pth'
        print('structure_path',structure_path)
        structure_data = {
                    'outputs':outputs,
                    'model_state':structure_module.state_dict(),
                    'feats':feats,
                    'type':dtype,
                    'config':config,
                    'm': m,
                    'x_prev': x_prev,
                    'inplace_safe':inplace_safe,
                    'offload_inference':offload_inference,
                    'seq_mask':seq_mask,
                    'init_params': {
                        'c_s': structure_module.c_s,
                        'c_z': structure_module.c_z,
                        'c_ipa': structure_module.c_ipa,
                        'c_resnet': structure_module.c_resnet,
                        'no_heads_ipa': structure_module.no_heads_ipa,
                        'no_qk_points': structure_module.no_qk_points,
                        'no_v_points': structure_module.no_v_points,
                        'dropout_rate': structure_module.dropout_rate,
                        'no_blocks': structure_module.no_blocks,
                        'no_transition_layers': structure_module.no_transition_layers,
                        'no_resnet_blocks': structure_module.no_resnet_blocks,
                        'no_angles': structure_module.no_angles,
                        'trans_scale_factor': structure_module.trans_scale_factor,
                        'epsilon': structure_module.epsilon,
                        'inf': structure_module.inf,
                        'is_multimer': structure_module.is_multimer,
                        }
                    }
        cloned_data_to_save = deep_clone(structure_data)
        torch.save(cloned_data_to_save, structure_path)
        
        # torch.save(structure_data, structure_path)

        model_path = str(args.output_model_dir)+str(epoch+1)+'.pth'
        print('model_path',model_path)
        model_data = {
                    'model_state':model.state_dict(),
                    'opt_state':optimizer.state_dict(),
                    'atom_weights':atom_weights,
                    'sdevs':sdevs,
                    }
        torch.save(model_data, model_path)


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
        "--input_model_dir", type=str, default=None,
        help="""Path to the trained GMM model file"""
    )
    parser.add_argument(
        "--head_dir", type=str,
        help="""Path to the auxiliary heads file containing the initial auxiliary heads state"""
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
        "--output_structure_dir",type=str,
        help="""Directory to save the output structure module files"""
    )
    parser.add_argument(
        "--output_model_dir",type=str,
        help="""Directory to save the output GMM model files"""
    )
    parser.add_argument(
        "--z_init",default='normal',
        help="""ignore this argument, it is not used in the current implementation"""
    )
    parser.add_argument(
        "--max_freq", type=float, default=1.0,
        help="""Maximum frequency for the FRC computation, default is 1.0"""
    )
    parser.add_argument(
        "--sonly",action="store_true", default=False,
        help="""Just keep true"""
    )
    parser.add_argument(
        "--particle_sign", type=int, default=-1,
        help="""Sign of the particle data, default is -1. Set to 1 for positive particles."""
    )    
    parser.add_argument(
        "--boxsize", type=int, default=256,
        help="""Size of the box for the particle data, default is 256"""
    )
    parser.add_argument(
        "--apix", type=float, default=1.0,
        help="""Pixel size in Angstroms, default is 1.0"""
    )
    parser.add_argument(
        "--norm",action="store_true", default=False,
        help="""Whether to normalize the particle data, default is False"""
    )
    parser.add_argument(
        "--learn_weight",action="store_true", default=False,
        help="""Whether to learn the atom weights during training, default is False"""
    )
    parser.add_argument(
        "--learn_width",action="store_true", default=False,
        help="""Whether to learn the atom's gaussian width during training, default is False"""
    )
    parser.add_argument(
        "--resolution",type=float, default=3.0,
        help="""Initial resolution for the FRC computation, default is 3.0"""
    )
    parser.add_argument(
        "--density_center",default=None,type=float,nargs=2
    )
    parser.add_argument(
        "--em_map",default=None,
    )
    parser.add_argument(
        "--batch_size",type=int, default=32,
        help="""Batch size for training, default is 32"""
    )
    parser.add_argument(
        "--mini_batch_size",type=int, default=8,
        help="""Mini batch size to decompose batch size (gradient accumulation), default is 8"""
    )
    parser.add_argument(
        "--epochs",type=int, default=10,
        help="""Number of epochs to train the model, default is 10"""
    )
    parser.add_argument(
        "--invert_z", action="store_true", default=False,
        help="""Whether to invert the Z-axis in the particle data, default is False"""
    )
    parser.add_argument(
    "--affine_matrices_csv", type=str, default=None,
    help="""Path to a CSV file containing affine matrices (each matrix is 3 rows x 4 cols)"""
    )
    parser.add_argument(
    "--mask_split_csv", type=str, default=None,
    help="""Path to CSV file that defines split indices for masks. Each line is a split point (e.g., 0, 12, 28, ...)"""
    )
    parser.add_argument(
        "--is_multimer", action="store_true", default=False,
        help="""Enable multimer mode. In this mode, monomer coordinates are replicated to match the number of affine matrices,
                and a single set of atom weights/sdevs is shared across all subunits.
                Set this flag when refining symmetric multimers."""
    )
    args = parser.parse_args()
    print(args)
    main(args)




