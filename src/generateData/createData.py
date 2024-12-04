import os
import numpy as np
import csv
from enum import Enum

SIZE = 1000000
BLOCK_SIZE = 1


class Distribution(Enum):
    LOGNORMAL = 5


# store path
filePath = {
    Distribution.LOGNORMAL: "./lognormal1M.csv"
}


def create_data(distribution, data_size=SIZE):
    if distribution == Distribution.LOGNORMAL:
        data = np.random.lognormal(0, 2, data_size)

    res_path = filePath[distribution]
    data.sort()

    # Create the 'data' directory if it doesn't exist
    os.makedirs(os.path.dirname(res_path), exist_ok=True)

    with open(res_path, 'w', newline='') as csvFile:
        csv_writer = csv.writer(csvFile)
        for i, d in enumerate(data):
            csv_writer.writerow([int(d * 10000), i / BLOCK_SIZE])


if __name__ == "__main__":
    create_data(Distribution.LOGNORMAL)