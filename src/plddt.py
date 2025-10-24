import os
from Bio import PDB
import argparse


def filter_pdb_by_plddt(input_pdb, output_pdb, plddt_threshold=30):
    """
    Filter atoms in a PDB file based on pLDDT score (stored in B-factor field).
    Only atoms with pLDDT >= threshold will be retained.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    class PLDDTFilter(PDB.Select):
        def accept_atom(self, atom):
            return atom.bfactor >= plddt_threshold

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, select=PLDDTFilter())


def batch_filter_pdb_files(input_dir, output_dir, plddt_threshold=30, file_extension=".pdb"):
    """
    Batch filter PDB files in a directory by pLDDT threshold.
    Args:
        input_dir (str): Path to input directory containing PDB files.
        output_dir (str): Path to output directory.
        plddt_threshold (int): Minimum pLDDT score to keep an atom.
        file_extension (str): File extension to filter (default: ".pdb").
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(file_extension):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                filter_pdb_by_plddt(input_path, output_path, plddt_threshold)
                print(f"[OK] {filename}")
            except Exception as e:
                print(f"[ERROR] {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter atoms in PDB files by pLDDT threshold.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input PDB files.")
    parser.add_argument("--output_dir", required=True, help="Directory to store filtered PDB files.")
    parser.add_argument("--threshold", type=float, default=30, help="pLDDT threshold (default: 30).")
    parser.add_argument("--ext", type=str, default=".pdb", help="File extension to filter (default: .pdb).")
    args = parser.parse_args()

    batch_filter_pdb_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        plddt_threshold=args.threshold,
        file_extension=args.ext
    )