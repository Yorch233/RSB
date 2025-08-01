import argparse
import concurrent
import json
import os
from glob import glob
from os.path import join
import librosa
import numpy as np
import pandas as pd
from soundfile import read
from tabulate import tabulate
from tqdm import tqdm

from RSB.evaluate import MetricRegister


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing configuration parameters for audio quality metrics calculation.

    Arguments:
        clean_dir (str): Directory path where the clean audio files are stored.
        noisy_dir (str): Directory path where the noisy audio files are stored.
        enhanced_dir (str): Directory path where the enhanced audio files are located.
        sample_rate (int): Sample rate of the audio files (default: 16000).
        max_workers (int): Max number of workers (default: 8).
    """
    parser = argparse.ArgumentParser(
        description='Calculate audio quality metrics between clean and enhanced audio files.'
    )
    
    parser.add_argument(
        "--clean_dir",
        type=str,
        required=True,
        help='Directory path where the clean audio files are stored.'
    )
    parser.add_argument(
        "--noisy_dir",
        type=str,
        required=True,
        help='Directory path where the noisy audio files are stored.'
    )
    parser.add_argument(
        "--enhanced_dir",
        type=str,
        required=True,
        help='Directory path where the enhanced audio files are located.'
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help='Sample rate of the audio files.'
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help='Max number of workers. Defaults to number of CPU cores.'
    )
    
    return parser.parse_args()

def mean_std(data):
    """
    Calculate the mean and standard deviation of the data, ignoring NaN values.

    Args:
        data (np.ndarray): Array of numerical values, potentially containing NaNs.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the non-NaN values.
    """
    data = data[~np.isnan(data)]  # Remove NaN values
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def evaluate(metrics: dict,
             clean_dir: str,
             noisy_dir: str,
             enhanced_dir: str,
             filename: str,
             sample_rate: int = 16000):
    """
    Evaluate audio quality metrics for a single file.

    Args:
        metrics (dict): Dictionary of metric calculator instances.
        clean_dir (str): Directory containing clean reference audio files.
        noisy_dir (str): Directory containing noisy input audio files.
        enhanced_dir (str): Directory containing enhanced (processed) audio files.
        filename (str): Name of the audio file to evaluate.
        sample_rate (int): Sample rate for loading audio files. Defaults to 16000.

    Returns:
        dict: A dictionary containing the filename and calculated metric values.
    """
    clean_audio_path = join(clean_dir, filename)
    noisy_audio_path = join(noisy_dir, filename)
    enhanced_audio_path = join(enhanced_dir, filename)

    # Use librosa for efficient audio loading
    x, _ = librosa.load(clean_audio_path, sr=sample_rate)  # Clean signal
    y, _ = librosa.load(noisy_audio_path, sr=sample_rate)  # Noisy signal
    x_hat, _ = librosa.load(enhanced_audio_path,
                            sr=sample_rate)  # Enhanced signal

    # Ensure all signals have the same length by trimming to the shortest
    len_min = min(x.size, y.size, x_hat.size)
    x, y, x_hat = x[:len_min], y[:len_min], x_hat[:len_min]

    result = {"filename": filename}
    # Calculate each metric for the current file
    for metric in metrics.values():
        # Pass reference, degraded (enhanced), noise, and path to the metric calculator
        metric_res = metric.calculate(ref_wav=x,
                                      deg_wav=x_hat,
                                      noise_wav=y - x,
                                      sample_rate=sample_rate,
                                      wav_path=enhanced_audio_path)
        result.update(
            metric_res)  # Add metric results to the output dictionary
    return result


def calc_metrics(clean_dir: str,
                 noisy_dir: str,
                 enhanced_dir: str,
                 sample_rate: int = 16000,
                 max_workers: int = 0,
                 overwrite: bool = True):
    """
    Calculate audio quality metrics for all files in specified directories.

    Args:
        clean_dir (str): Directory containing clean reference audio files.
        noisy_dir (str): Directory containing noisy input audio files.
        enhanced_dir (str): Directory containing enhanced (processed) audio files.
        sample_rate (int): Sample rate of the audio files. Defaults to 16000.
        max_workers (int): Maximum number of worker threads for parallel processing.
                           Defaults to 0 (uses ThreadPoolExecutor default, typically number of CPU cores).
        overwrite (bool): If True, re-calculate metrics even if results file exists. Defaults to True.
    """
    print(f"Clean Speech: {clean_dir}")
    print(f"Noisy Speech: {noisy_dir}")
    print(f"Enhanced Speech: {enhanced_dir}")

    output_json_path = join(enhanced_dir, "metrics.json")
    results = None
    # Load existing results if available and not overwriting
    if os.path.exists(output_json_path) and not overwrite:
        with open(output_json_path, "r") as json_file:
            results = json.load(json_file)
        print(
            f"Results are loaded from existing JSON file `{output_json_path}`")
    else:
        # Find all clean audio files efficiently using os.scandir
        clean_files = sorted(
            [f.path for f in os.scandir(clean_dir) if f.name.endswith('.wav')])
        file_names = [os.path.basename(f)
                      for f in clean_files]  # Extract filenames

        print(f"Number of files: {len(file_names)}")

        # Fetch and instantiate the required metric calculators
        # MetricRegister.fetch(['pesq', 'estoi', 'composite', 'energy_ratios', 'dnsmos'])
        metric_fns = MetricRegister.fetch(
            ['pesq', 'estoi', 'composite', 'energy_ratios', 'dnsmos'])

        for metric_name in metric_fns.keys():
            metric_fns[metric_name] = metric_fns[metric_name](
            )  # Instantiate each metric

        # Dictionary to store results for all files
        data = {"filename": []}

        # Use ThreadPoolExecutor for parallel processing of files
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as executor:
            # Submit evaluation tasks for all files
            future_to_name = {
                executor.submit(evaluate, metric_fns, clean_dir, noisy_dir, enhanced_dir, filename, sample_rate):
                filename
                for filename in file_names
            }
            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_name),
                               desc='Calculating',
                               total=len(file_names),
                               miniters=50):
                name = future_to_name[future]
                try:
                    result = future.result(
                    )  # Get the result of the evaluation
                except Exception as exc:
                    print(f"{name} generated an exception: {exc}")
                else:
                    # Append results for this file to the main data dictionary
                    for item_key in result.keys():
                        if item_key not in data.keys():
                            data[item_key] = [
                            ]  # Initialize list for new metric
                        data[item_key].append(
                            result[item_key])  # Append metric value

        # Convert collected data to a Pandas DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(join(enhanced_dir, "_results.csv"), index=False)

        # Calculate mean and std for each metric and prepare summary results
        results = {"Metric": [], "Mean": [], "Std": []}

        for metric_key in [k for k in data.keys()
                           if k != 'filename']:  # Iterate through metric keys
            mean, std = mean_std(
                df[metric_key].to_numpy(dtype=np.float64))  # Calculate stats
            results["Metric"].append(metric_key.upper())  # Store metric name
            results["Mean"].append(np.float64(mean))  # Store mean
            results["Std"].append(np.float64(std))  # Store standard deviation

        # Save summary results to a JSON file
        with open(output_json_path, "w") as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Results have been saved to {output_json_path}")

    # Print the final summary table using tabulate
    headers = ["Metric", "Mean", "Std"]
    print("\nSummary of Results:")
    print(tabulate(results, headers=headers, floatfmt=".4f", numalign="right"))

    # Print a one-line summary of means and stds
    summary_line = " | ".join([
        f"{mean:.4f} | {std:.4f}" for metric, mean, std in zip(
            results["Metric"], results["Mean"], results["Std"])
    ])
    print(summary_line)

if __name__ == '__main__':
    # Entry point of the script
    args = parse_args()  # Parse command-line arguments
    # Call the main calculation function with provided arguments
    calc_metrics(clean_dir=args.clean_dir,
                 noisy_dir=args.noisy_dir,
                 enhanced_dir=args.enhanced_dir,
                 sample_rate=args.sample_rate,
                 max_workers=args.max_workers)
