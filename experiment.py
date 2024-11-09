from itertools import repeat
import numpy as np
import psutil
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Dict, List, Tuple

NUM_CORES = psutil.cpu_count(logical=False)

def run_experiment(
    strategies: Dict[str, Callable], batch_size: int, n_values, m: int, T: int
) -> None:
    gaps = np.zeros((len(strategies), len(n_values)))
    stddevs = np.zeros_like(gaps)
    # Run in parallel
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = executor.map(
            compute_gaps,
            n_values,
            repeat(batch_size),
            repeat(m),
            repeat(T),
            repeat(list(strategies.values()))
        )

        # Extract results and populate gaps and stddevs
        for idx, (mean_gaps, std_gaps) in enumerate(results):
            gaps[:, idx] = mean_gaps
            stddevs[:, idx] = std_gaps
    return gaps, stddevs
    
# Helper function for running one single trial
def run_trial(trial, gaps, batch_size, n, m, strategies):
    bins = np.zeros((len(strategies), m))    
    for _ in range(n // batch_size):
        for i, func in enumerate(strategies):
            if isinstance(func, tuple):
                func[0](bins[i], batch_size, m, func[1])
            else:
                func(bins[i], batch_size, m)
                
    for i in range(len(strategies)):
        gaps[i, trial] = compute_gap(bins[i], n, m)
    
def compute_gaps(
    n: int, batch_size: int, m: int, T: int, strategies: List[Callable]
) -> Tuple[np.ndarray, np.ndarray]:
    gaps = np.zeros((len(strategies), T))
    for trial in range(T):
        run_trial(trial, gaps, batch_size, n, m, strategies)
    return np.mean(gaps, axis=1), np.std(gaps, axis=1)



def compute_gap(bins: np.ndarray, n: int, m: int) -> float:
    """
    Computes the gap, defined as the difference between the maximum bin load and the average load across all bins.

    Args:
        bins (numpy.ndarray): Array representing the current load of each bin.
        n (int): Total number of balls distributed across bins.
        m (int): Number of bins.

    Returns:
        float: The gap between the maximum load and the average load.
    """
    return np.max(bins) - n / m

def plot(plot_type, avgs, stddevs, x, labels, x_max, x_step, y_max, y_step, title, filename):
    sns.set_theme(style="whitegrid")     
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        match(plot_type.lower()):
            case "average":
                    sns.lineplot(x=x, y=avgs[i, :], marker=".", label=label)
                    plt.fill_between(
                        x, avgs[i, :] - stddevs[i, :], avgs[i, :] + stddevs[i, :], alpha=0.2
                    )
            case "standard deviation":
                sns.lineplot(x=x, y=stddevs[i, :], marker=".", label=label)
    plt.xticks(np.arange(0, x_max, x_step), rotation=45) 
    plt.yticks(np.arange(0, y_max, y_step))
    plt.xlabel("Number of Balls (n)")
    plt.ylabel(f"{plot_type} of Gap (G_n)")
    plt.title(title)
    plt.savefig(filename, bbox_inches="tight")
    plt.legend()
    plt.show()
