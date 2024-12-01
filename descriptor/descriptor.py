import os
import re
import pandas as pd
from pymatgen.core import Structure
from matminer.featurizers.structure import XRDPowderPattern
import tqdm


def get_project_root():
    """
    Returns the absolute path of the project root directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def read_excel_file(file_name, folder_name):
    """
    Reads an Excel file.

    Args:
        file_name (str): The name of the Excel file to read.
        folder_name (str): The folder where the file is located.

    Returns:
        pd.DataFrame: DataFrame containing the contents of the Excel file.
    """
    project_root = get_project_root()
    file_path = os.path.join(project_root, folder_name, file_name)

    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"File {file_name} not found in the {folder_name} folder.")
        return None


def calculate_num(formula):
    """
    Calculate the number of unique elements in the chemical formula.

    Args:
        formula (str): The chemical formula as a string.

    Returns:
        int: The number of unique elements in the formula.
    """
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    return len(matches)


def calculate_natoms(formula):
    """
    Calculate the total number of atoms in the chemical formula.

    Args:
        formula (str): The chemical formula as a string.

    Returns:
        int: The total number of atoms in the formula.
    """
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    return sum([int(item[1]) if item[1] else 1 for item in matches])


def calculate_properties(element_properties, formula):
    """
    Calculate properties of elements in the chemical formula, such as sum, min, max, and range.

    Args:
        element_properties (pd.DataFrame): DataFrame containing properties of elements.
        formula (str): The chemical formula as a string.

    Returns:
        dict: Dictionary containing the calculated properties (sum, min, max, range) for each element property.
    """
    properties_dict = {}
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    # Loop through each matched element in the formula
    for match in matches:
        element_name, count_str = match
        if element_name not in element_properties.index:
            return None
        count = int(count_str) if count_str else 1
        element_attributes = element_properties.loc[element_name]

        # Store the properties for the current element
        for attribute in element_attributes.index:
            attribute_value = element_attributes[attribute]
            if attribute in properties_dict:
                properties_dict[attribute].append([attribute_value, count])
            else:
                properties_dict[attribute] = [[attribute_value, count]]

    result_dict = {}
    # Calculate statistics for each attribute (sum, min, max, range)
    for attribute, values in properties_dict.items():
        min_value = min(item[0] for item in values)
        max_value = max(item[0] for item in values)
        range_value = max_value - min_value
        total_sum = sum(item[0] * item[1] for item in values)

        result_dict[f'sum_{attribute}'] = total_sum
        result_dict[f'min_{attribute}'] = min_value
        result_dict[f'max_{attribute}'] = max_value
        result_dict[f'range_{attribute}'] = range_value

    return result_dict


def process_compositions(element_file, formula_file, CSV=False):
    """
    Processes chemical formulas and calculates element properties.

    Args:
        element_file (str): The name of the Excel file containing element properties.
        formula_file (str): The name of the Excel file containing chemical formulas.

    Returns:
        pd.DataFrame: DataFrame containing the chemical formula and calculated properties.
        pd.DataFrame: The original DataFrame containing the chemical formulas.
    """
    element_properties_df = read_excel_file(element_file, 'data')
    if element_properties_df is None:
        return None, None

    element_properties_df = element_properties_df.set_index('Elements')

    chemical_formulas_df = read_excel_file(formula_file, 'data')
    if chemical_formulas_df is None:
        return None, None

    result_columns = ['index', 'Chemical_Formula']
    for attribute in element_properties_df.columns:
        for stat in ['sum', 'min', 'max', 'range']:
            result_columns.append(f"{stat}_{attribute}")

    compositions_df = pd.DataFrame(columns=result_columns)

    # Iterate through each row, calculate properties, and store the results
    for index, row in chemical_formulas_df.iterrows():
        formula_name = row['composition']
        properties = calculate_properties(element_properties_df, formula_name)

        if properties:
            new_row = pd.DataFrame([[row['index'], formula_name, *properties.values()]], columns=result_columns)
            compositions_df = pd.concat([compositions_df, new_row], ignore_index=True)

    # Add additional columns for species and atom counts
    compositions_df['nspecies'] = compositions_df['Chemical_Formula'].apply(calculate_num)
    compositions_df['natoms'] = compositions_df['Chemical_Formula'].apply(calculate_natoms)

    return compositions_df, chemical_formulas_df


def process_xrd_features(base_folder='data/cifs', two_theta_range=(0, 90), bw_method=0.05):
    """
    Processes the X-ray diffraction (XRD) features of CIF files and returns a DataFrame containing the features.

    Args:
        base_folder (str): The relative path to the directory containing the CIF files from the project root (default is 'data/cifs').
        two_theta_range (tuple): The range of two-theta angles for the XRD calculation (default is (0, 90)).
        bw_method (float): The bandwidth method used for XRD feature calculation (default is 0.05).

    Returns:
        pd.DataFrame: DataFrame containing the calculated XRD features.
    """

    project_root = get_project_root()
    base_path = os.path.join(project_root, base_folder)

    XRD = XRDPowderPattern(two_theta_range=two_theta_range, bw_method=bw_method)

    # Define the structure columns, including 'index' and 'theta_{i}' for each two-theta value
    structure_columns = ['index']
    for i in range(0, 91):
        structure_columns.append(f"theta_{i}")

    structure_df = pd.DataFrame(columns=structure_columns)

    # Iterate over CIF files (assuming they are named 1.cif, 2.cif, ..., 10.cif)
    for i in tqdm.tqdm(range(1, 6352)):
        path = os.path.join(base_path, f'{i}.cif')
        structure = Structure.from_file(path)

        try:
            xray_diffraction_information = XRD.featurize(structure)
            new_row = pd.DataFrame([[i, *xray_diffraction_information]], columns=structure_columns)
            structure_df = pd.concat([structure_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Exception for structure {i}: {str(e)}")
            continue

    return structure_df


if __name__ == "__main__":
    """
    Main execution block to process compositions, XRD features, and combine the results.
    """
    # Process XRD features
    structure_df = process_xrd_features(base_folder='data\\cifs')

    # Process compositions and element properties
    compositions_df, origin = process_compositions('element.xlsx', 'compositions.xlsx')

    final_df = pd.merge(compositions_df.drop('Chemical_Formula', axis=1),structure_df, on='index')
    final_df = pd.merge(origin.drop('composition', axis=1),final_df, on='index')
    final_df.to_excel('../data/data_all.xlsx', index=False)
