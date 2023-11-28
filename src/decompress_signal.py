import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb
import sys


def transform_signals(signals):
    return np.array(signals[:, 0] * 0.1, signals[:, 1] * 5)


if __name__ == "__main__":
    print("Parsing arguments.")
    # parse arguments from commandline args
    readpath = os.path.abspath(sys.argv[1])
    writepath = os.path.abspath(sys.argv[2])
    print(f"Decompressing {os.path.basename(readpath)} -> {os.path.basename(writepath)}.")

    # Load the record
    print("Loading record.")
    record = wfdb.rdrecord(readpath)

    # Get signal column titles & data
    print("Retrieving column titles & signals.")
    col_titles = record.sig_name
    raw_signals = record.p_signal

    # Transform into physical units
    print("Transforming signals to physical units.")
    signals = transform_signals(raw_signals)

    print("Writing to output.")
    # Write out to new file
    with open(writepath, "w") as file:
        file.write(",".join(col_titles) + "\n")
        for i in range(signals.shape[0]):
            file.write(",".join([str(x) for x in signals[i, :]]) + "\n")
    print("Finished.")

    exit(0)
