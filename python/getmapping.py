import numpy as np


def getmapping(samples, mappingtype):
    """
    Returns a structure containing a mapping table for LBP codes.

    Parameters:
    - samples: Number of sampling points
    - mappingtype: Type of mapping ('u2', 'ri', or 'riu2')

    Returns:
    - mapping: A dictionary containing the mapping table, number of samples, and number of bins
    """

    table = np.arange(2**samples, dtype=np.int32)
    newMax = 0  # number of patterns in the resulting LBP code
    index = 0

    if mappingtype == 'u2':  # Uniform 2
        newMax = samples * (samples - 1) + 3
        for i in range(2**samples):
            i_bin = format(i, f'0{samples}b')
            j_bin = i_bin[-1] + i_bin[:-1]  # rotate left
            numt = sum(i_bin[k] != j_bin[k] for k in range(samples))
            if numt <= 2:
                table[i] = index
                index += 1
            else:
                table[i] = newMax - 1

    elif mappingtype == 'ri':  # Rotation invariant
        tmpMap = np.full(2**samples, -1, dtype=np.int32)
        for i in range(2**samples):
            rm = i
            r = i
            r_bin = format(r, f'0{samples}b')

            for j in range(1, samples):
                r = int(r_bin[-j:] + r_bin[:-j], 2)  # rotate left
                if r < rm:
                    rm = r

            if tmpMap[rm] < 0:
                tmpMap[rm] = newMax
                newMax += 1
            table[i] = tmpMap[rm]

    elif mappingtype == 'riu2':  # Uniform & Rotation invariant
        newMax = samples + 2
        for i in range(2**samples):
            i_bin = format(i, f'0{samples}b')
            j_bin = i_bin[-1] + i_bin[:-1]  # rotate left
            numt = sum(i_bin[k] != j_bin[k] for k in range(samples))
            if numt <= 2:
                table[i] = bin(i).count('1')
            else:
                table[i] = samples + 1

    else:
        raise ValueError("Invalid mapping type. Choose 'u2', 'ri', or 'riu2'.")

    mapping = {
        'table': table,
        'samples': samples,
        'num': newMax
    }

    return mapping


# Example usage
if __name__ == "__main__":
    # Example: Generate a uniform LBP mapping for 8 sampling points
    mapping_u2 = getmapping(8, 'u2')
    print("Uniform LBP mapping:")
    print(f"Number of samples: {mapping_u2['samples']}")
    print(f"Number of bins: {mapping_u2['num']}")
    print(f"Mapping table (first 10 elements): {mapping_u2['table'][:10]}")

    # Example: Generate a rotation-invariant uniform LBP mapping for 16 sampling points
    mapping_riu2 = getmapping(16, 'riu2')
    print("\nRotation-invariant uniform LBP mapping:")
    print(f"Number of samples: {mapping_riu2['samples']}")
    print(f"Number of bins: {mapping_riu2['num']}")
    print(f"Mapping table (first 10 elements): {mapping_riu2['table'][:10]}")
