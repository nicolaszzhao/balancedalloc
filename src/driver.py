import numpy as np
import experiment
from pathlib import Path
from strategies import d_choice, beta_choice, k_partial_information

# Number of bins
m = 100
# Number of balls
n = m**2
# Number of trials
T = 10

# Save generated plots as PDFs?
SAVE_FIGS = True


def run(name, strategies, batches, n, m, T, save_fig=False, save_path=Path("../report/assets")):
    print(f"\t- Running experiments for {batches} batches, n={n}, m={m}, T={T}.")
    avgs, stddevs = zip(*(experiment.run(strategies, b, n, m, T) for b in batches))
    avgs_array = np.concatenate([np.ravel(avg) for avg in avgs])
    stddevs_array = np.concatenate([np.ravel(stddev) for stddev in stddevs])

    # Calculate steps and max values for plotting
    x_step = n // 10
    x_max = n + x_step
    max_avg_stddevs = (avgs_array + stddevs_array).max()
    y_step_avg = max_avg_stddevs // 10 + 1
    y_max_avg = (max_avg_stddevs // y_step_avg) * y_step_avg
    max_stddev = stddevs_array.max()
    y_step_stddev = max_stddev // 10 + 1
    y_max_stddev = (max_stddev // y_step_stddev) * y_step_stddev

    labels = list(strategies.keys())
    for avg, stddev, b in zip(avgs, stddevs, batches):
        xs = range(m, n + m, m)
        if save_fig:
            save_path.mkdir(exist_ok=True)
        avg_path = save_path / Path(f"avg-{n}_n-{b}_batch-{name}.pdf")
        stddev_path = save_path / Path(f"stddev-{n}_n-{b}_batch-{name}.pdf")

        experiment.plot(
            "Average",
            avg,
            stddev,
            xs,
            labels,
            x_max,
            x_step,
            y_max_avg,
            y_step_avg,
            save_fig,
            avg_path,
        )
        experiment.plot(
            "Standard Deviation",
            avg,
            stddev,
            xs,
            labels,
            x_max,
            x_step,
            y_max_stddev,
            y_step_stddev,
            save_fig,
            stddev_path,
        )
        if save_fig:
            print(f"\t- Generated: {avg_path}")
            print(f"\t- Generated: {stddev_path}")

def main():
    ds = [1, 2, 3, 4, 5]
    batches = [1, 500, 2000]

    d_choices_strategies = {f"{d}-Choice": (d_choice, d) for d in ds}
    experiment_name = "d-choices"
    print(f"* Processing: {experiment_name} plots.")
    run("d-choices", d_choices_strategies, batches, n, m, T, SAVE_FIGS)
    batches = [5000]
    run("d-choices", d_choices_strategies, batches, n, m, T, SAVE_FIGS)
    print(f"* Completed: {experiment_name} plots.")


    betas = [0.25, 0.5, 0.75]
    ds = [1, 2]
    batches = [1, 500, 2000] 

    d_choices_strategies = {f"{d}-Choice": (d_choice, d) for d in ds}


    beta_choices_strategies = {
        f"(1+{beta})-Choice": (beta_choice, beta) for beta in betas
    }

    strategies = {
        **beta_choices_strategies,
        **d_choices_strategies,
    }

    experiment_name = "d-choices vs beta-choices"
    print(f"* Processing: {experiment_name} plots.")
    run("dchoice-vs-beta", strategies, batches, n, m, T, SAVE_FIGS)
    batches = [5000] 
    run("dchoice-vs-beta", strategies, batches, n, m, T, SAVE_FIGS)
    print(f"* Completed: {experiment_name} plots.")

    ds = [2]
    ks = [1, 2]
    betas = [0.25, 0.75]
    batches = [1, 500 ,2000] 

    d_choices_strategies = {f"{d}-Choice": (d_choice, d) for d in ds}
    beta_choices_strategies = {
        f"(1+{beta})-Choice": (beta_choice, beta) for beta in betas
    }
    partial_info_strategies = {
        f"Partial info k={k}": (k_partial_information, k) for k in ks
    }

    all_strategies = {
        **d_choices_strategies,
        **beta_choices_strategies,
        **partial_info_strategies,
    }

    experiment_name = "All in One"
    print(f"* Processing: {experiment_name} plots.")
    run("all", all_strategies, batches, n, m, T, SAVE_FIGS)

    batches = [5000] 
    run("all", all_strategies, batches, n, m, T, SAVE_FIGS)
    print(f"* Completed: {experiment_name} plots.")

if __name__ == "__main__":
    main()
