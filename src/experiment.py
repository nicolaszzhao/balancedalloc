import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run(strategies, b, n, m, T):
    means = []
    stddevs = []
    for strategy in strategies.values():
        strategy_results = []
        for _ in range(T):
            trial_results = __run_trial(
                b,
                n,
                m,
                strategy,
            )
            strategy_results.append(trial_results)
        means.append(np.mean(strategy_results, axis=0))
        stddevs.append(np.std(strategy_results, axis=0))
    return np.array(means), np.array(stddevs)


def plot(
    plot_type,
    avgs,
    stddevs,
    x,
    labels,
    x_max,
    x_step,
    y_max,
    y_step,
    save_fig,
    filename,
):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        match (plot_type.lower()):
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
    plt.legend(loc="upper left", fontsize="small")
    if save_fig:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def __run_trial(b, n, m, strategy):
    bin = np.zeros(m)
    result = []
    chunk_size = m if b // m > 0 else 1
    for i in range(0, n, b):
        func, arg = strategy
        bin_outdated = np.copy(bin)
        for j in range(0, b, chunk_size):
            func(bin, bin_outdated, chunk_size, m, arg)
            if b == 1 and i % m != 0:
                continue
            gap = __compute_gap(bin, i + j + 1, m)
            result.append(gap)
    return result


def __compute_gap(bins: np.ndarray, n: int, m: int) -> float:
    return np.max(bins) - n / m
