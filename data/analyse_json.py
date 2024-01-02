import json
import os
import numpy as np


def analyse_params(results):
    non_params = ["rmse", "init_time", "train_time", "forecast_time"]
    param_keys = [key for key in results[0] if key not in non_params]

    out_doc = {key: {param: [] for param in param_keys} for key in non_params}

    for measurement in non_params:
        for param in param_keys:
            values = {}
            for param_set in results:
                # Pass if measurement value, e.g. RMSE, is None
                if not param_set[measurement]:
                    continue

                if param_set[param] not in values:
                    values[param_set[param]] = [
                        param_set[measurement]
                        if measurement == "rmse"
                        else np.float64(param_set[measurement][0:-1])
                    ]
                else:
                    values[param_set[param]].append(
                        param_set[measurement]
                        if measurement == "rmse"
                        else np.float64(param_set[measurement][0:-1])
                    )

            out_doc[measurement][param] = {
                value: {
                    "mean": np.mean(array),
                    "std": np.std(array, ddof=1),
                    "N": len(array),
                    "max": max(array),
                    "min": min(array),
                }
                for value, array in values.items()
            }

    return out_doc


def analyse_cross_file_params(json_files):
    return


def get_best_n(json_results, n):
    best_n = {i: None for i in range(1, n + 1)}
    count = 0
    for params in json_results:
        if count < n:
            count += 1
            best_n[count] = params
        else:
            for i in range(1, n + 1):
                if params["rmse"] and params["rmse"] < best_n[i]["rmse"]:
                    best_n[i] = params
                    for j in range(n, i, -1):
                        best_n[j] = best_n[j - 1]
                    break
    return best_n


def get_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


basepath = os.path.abspath(os.path.join("..", "data", "metadata"))
filenames = [
    "17_measured_100.json",
    "22_measured_100.json",
    "25_measured_100.json",
    "28_measured_100.json",
]
json_files = [(file, get_json(os.path.join(basepath, file))) for file in filenames]

n = 5
for filename, json_file in json_files:
    print(f"{filename}")
    best_n = get_best_n(json_file, n)
    for rank, params in best_n.items():
        print(f"\t{rank}:")
        for key, value in params.items():
            print(f"\t\t- {key}: {value}")


# n = 3
# for filename, json_file in json_files:
#    print(f"{filename}:")
#    best_n = get_best_n(json_file, n)
#    for rank, params in best_n.items():
#        print(f"\t{rank}:")
#        for key, value in params.items():
#            print(f"\t\t- {key}: {value}")

# x = analyse_params(json_files[0][1])

# for param, data in x["rmse"].items():
#    print(param)
#    for value, stats in data.items():
#        print(f"\t{value}:")
#        for stat, val in stats.items():
#            print(f"\t\t{stat}: {val}")
