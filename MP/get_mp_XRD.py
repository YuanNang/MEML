from multiprocessing import Pool
import pandas as pd
import os
from pymatgen.core import Structure
from matminer.featurizers.structure import XRDPowderPattern

from tqdm import tqdm
import warnings
import time
warnings.filterwarnings("ignore")

def read_file(index_and_row):
    index, row = index_and_row
    path = os.path.join(base_path, row['ID'], row['crystal_face_id'] + '.cif')
    structure = Structure.from_file(path)
    return index, structure


def process_structure(index_and_structure):
    index, Structure = index_and_structure

    print('Run task %s (%s)...' % (index + 1, os.getpid()))
    start = time.time()

    try:
        xray_diffraction_information = XRD.featurize(Structure)

        end = time.time()
        print('Task %s runs %0.2f seconds.' % (index + 1, (end - start)))

        return [index + 1, *xray_diffraction_information]
    except Exception as e:
        print(f"Exception for structure {index + 1}: {str(e)}")
        # Optionally, you can return some default value or None here
        return None


if __name__ == '__main__':

    df = pd.read_csv("mp_data.csv", index_col=0)

    base_path = 'mp_facet_structure'

    # 创建一个新的 DataFrame 以存储结果
    result_columns = ['index'] + [f"theta_{i}" for i in range(91)]
    XRD = XRDPowderPattern(two_theta_range=(0, 90), bw_method=0.05)

    # 使用进程池
    Structures = []
    for index_and_row in tqdm(df.iterrows(), total=len(df)):
        Structures.append(read_file(index_and_row))

    with Pool(processes=12) as pool:
        results = list(tqdm(pool.imap(process_structure, Structures), total=len(df)))

    # 过滤掉 None 结果
    result_data = [result for result in results if result is not None]
    results_df = pd.DataFrame(result_data, columns=result_columns)
    results_df.to_csv('mp_XRD.csv', index=False)