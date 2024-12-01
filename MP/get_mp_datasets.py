import os
import pandas as pd
from tqdm import tqdm
from func_timeout import FunctionTimedOut
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices


def generate_slab(structure, miller_index, file):
    """
    Function to generate slab and save as CIF file.
    """
    try:
        slabs = SlabGenerator(structure,
                              miller_index=miller_index,
                              min_slab_size=8.0,  # Minimum slab thickness
                              min_vacuum_size=15.0,  # Minimum vacuum size
                              lll_reduce=True)
        slab = slabs.get_slab(shift=0)
        name = ''.join(map(str, miller_index))
        slab.to(filename=os.path.join(file, f"{name}.cif"))
    except Exception as e:
        print(f"Error generating slab for {miller_index}: {e}")


def process_materials(mpids, base_dir):
    """
    Process materials, generate slabs and save their information.
    """
    for mpid in tqdm(mpids):
        structure_file = os.path.join('mp_cif', f'{mpid}.cif')
        structure = Structure.from_file(structure_file)

        mpid_dir = os.path.join(base_dir, mpid)
        os.makedirs(mpid_dir, exist_ok=True)

        # Get the symmetrically distinct Miller indices
        hkl_indices = get_symmetrically_distinct_miller_indices(structure, max_index=2)

        # Generate slabs for each Miller index
        for hkl in hkl_indices:
            generate_slab(structure, hkl, mpid_dir)


def collect_data(mpids, base_dir):
    """
    Collect data about each CIF file and save it into a DataFrame.
    """
    data = pd.DataFrame(columns=['index', 'ID', 'crystal_face_id', 'composition', 'Space_Group'])
    idx = 1
    for mpid in tqdm(mpids):
        folder_path = os.path.join(base_dir, mpid)
        cif_files = os.listdir(folder_path)

        # Loop over each CIF file in the material folder
        for cif_file in cif_files:
            cif_path = os.path.join(folder_path, cif_file)
            structure = Structure.from_file(cif_path)

            # Collect data
            formula = structure.formula
            space_group = structure.get_space_group_info()[1]

            data = pd.concat([data, pd.DataFrame({
                'index': [idx],
                'ID': [mpid],
                'crystal_face_id': [cif_file.split('.')[0]],  # Remove extension
                'composition': [formula],
                'Space_Group': [space_group],
            })], ignore_index=True)

            idx += 1
    return data


def main():
    # Load material IDs from the Excel file
    data = pd.read_excel('stable_ternary.xlsx', engine='openpyxl').iloc
    mpids = list(data.material_id)

    base_dir = "mp_facet_structure"

    # Step 1: Process materials and generate slabs
    process_materials(mpids, base_dir)

    # Step 2: Collect data from generated CIF files
    collected_data = collect_data(mpids, base_dir)

    # Step 3: Save the collected data to an Excel file
    collected_data.to_excel("../data/mp_data.xlsx", index=False)
    print("Datasets saved successfully.")


if __name__ == "__main__":
    main()
