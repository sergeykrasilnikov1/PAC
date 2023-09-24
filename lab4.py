import sys
import numpy as np


def read_data(path):
    with open(path, 'r') as file:
        data = file.read().strip()
        return np.array(data.split(), dtype=int)
    

def mix_arrays(array1, array2, p):
    mask = np.random.choice([0, 1], size=len(array1), p=[1-p, p])
    mixed_array = np.where(mask, array2, array1)
    return mixed_array


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python <script_name>.py <file_1_path> <file_2_path> <P>")
        sys.exit(1)

    file_1_path = sys.argv[1]
    file_2_path = sys.argv[2]
    p = float(sys.argv[3])

    array1 = read_data(file_1_path)
    array2 = read_data(file_2_path)

    if len(array1) != len(array2):
        print("Error: Arrays must have the same length.")
        sys.exit(1)

    mixed_array = mix_arrays(array1, array2, p)

    print(mixed_array)