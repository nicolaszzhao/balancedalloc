import numpy as np

def d_choice(bins: np.ndarray, read_bins: np.ndarray, batch_size: int, m: int, d: int) -> None:
    choices = np.random.choice(m, (batch_size, d))
    loads = read_bins[choices]
    min_loads = np.min(loads, axis=1, keepdims=True)
    min_indices = (loads == min_loads)
    cum_counts = np.cumsum(min_indices, axis=1) * min_indices
    chosen_idx = np.argmax((cum_counts == np.random.randint(1, cum_counts.max(axis=1, keepdims=True) + 1)), axis=1)
    updates = choices[np.arange(batch_size), chosen_idx]
    np.add.at(bins, updates, 1)


def beta_choice(bins: np.ndarray, read_bins,batch_size: int, m: int, beta: float) -> None:
    random_probs = np.random.rand(batch_size)

    two_choice_indices = np.where(random_probs >= beta)[0]
    one_choice_indices = np.where(random_probs < beta)[0]

    if two_choice_indices.size > 0:
        d_choice(bins, read_bins, two_choice_indices.size, m, 2)

    if one_choice_indices.size > 0:
        d_choice(bins, read_bins, one_choice_indices.size, m, 1)


def k_partial_information(bins: np.ndarray, read_bins,  batch_size: int, m: int, k: int) -> None:
    choices = np.random.choice(m, (batch_size, 2))
    median_load = np.median(read_bins)
    is_above_median = read_bins[choices] > median_load

    # Step 1: Choose bins based on median comparison
    primary_choice = np.where(
        is_above_median[:, 0] != is_above_median[:, 1],
        choices[np.arange(batch_size), np.argmin(is_above_median, axis=1)],
        -1 if k == 2 else np.random.choice(m, batch_size),
    )

    if k == 2:
        # For k=2, check the 75th percentile if both choices are either above or below median
        needs_secondary_check = primary_choice == -1
        top_75_percentile = np.percentile(read_bins, 75)
        is_in_top_75 = read_bins[choices[needs_secondary_check]] > top_75_percentile

        # Secondary choice based on top 75% comparison
        secondary_choice = np.where(
            is_in_top_75[:, 0] != is_in_top_75[:, 1],
            choices[needs_secondary_check, np.argmin(is_in_top_75, axis=1)],
            np.random.choice(
                m, needs_secondary_check.sum()
            ),  # Random choice if both are the same
        )

        # Replace -1 entries in primary_choice with secondary_choice results
        primary_choice[needs_secondary_check] = secondary_choice

    # Apply the final selected bin indices to increment the chosen bins' load
    np.add.at(bins, primary_choice, 1)
