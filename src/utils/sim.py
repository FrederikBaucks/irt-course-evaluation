"""Contains some helper functions for simulations."""
import os

def save_resp_matrix(matrix, path):
    """Save numpy response matrix as csv."""
    # create folder if necessary
    folder = path[:path.rfind("/")]
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # parse numpy to csv
    csv_data = ",".join(["C" + str(i) for i in range(matrix.shape[1])]) + "\n"
    for vec in matrix:
        row = ""
        for i, c in enumerate(vec):
            if i >= 1:
                row += ","
            if c != -1:
                row += str(c)
        csv_data += row + "\n"

    with open(path, "w") as file:
        n = file.write(csv_data)
