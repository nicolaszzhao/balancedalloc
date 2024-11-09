import numpy as np

def k_choices(bins: np.ndarray, batch_size: int, m: int, k: int) -> None:
    """
    Implements the two-choice strategy where each ball is placed in the less loaded of two randomly selected bins.

    Args:
        bins (numpy.ndarray): Array representing the current load of each bin.
        batch_size (int): Number of balls placed in each iteration.
        m (int): Number of bins.
        k (int): Number of balls that we can choose.

    Returns:
        None. Modifies the `bins` array in-place by updating bin loads.
    """
    choices = np.random.choice(m, (batch_size, k))
    loads = bins[choices]
    min_loads = np.min(loads, axis=1, keepdims=True)
    min_indices = (loads == min_loads)
    cum_counts = np.cumsum(min_indices, axis=1) * min_indices
    chosen_idx = np.argmax((cum_counts == np.random.randint(1, cum_counts.max(axis=1, keepdims=True) + 1)), axis=1)
    updates = choices[np.arange(batch_size), chosen_idx]
    np.add.at(bins, updates, 1)


def beta_choice(bins: np.ndarray, batch_size: int, m: int, beta: float) -> None:
    """
    Implements the beta-choice strategy where each ball is placed in a bin based on a probability `beta`:
    - With probability `beta`, a single bin is chosen randomly.
    - With probability `1 - beta`, the ball is placed in the lesser loaded of two randomly selected bins.

    Args:
        bins (numpy.ndarray): Array representing the current load of each bin.
        batch_size (int): Number of balls placed in each iteration.
        m (int): Number of bins.
        beta (float): Probability that a ball is placed in a single randomly selected bin.

    Returns:
        None. Modifies the `bins` array in-place by updating bin loads.
    """
    random_probs = np.random.rand(batch_size)

    one_choice_indices = np.where(random_probs < beta)[0]
    two_choice_indices = np.where(random_probs >= beta)[0]

    if two_choice_indices.size > 0:
        k_choices(bins, two_choice_indices.size, m, 2)

    if one_choice_indices.size > 0:
        k_choices(bins, one_choice_indices.size, m, 1)


def k_partial_information(bins: np.ndarray, batch_size: int, m: int, k: int) -> None:
    """
    Implements a partial-information strategy where balls are placed based on load comparisons relative to
    the median or 75th percentile (for k=2) thresholds:
    - If k=1: Compares bin loads to the median load to select a bin.
    - If k=2: First compares to the median; if inconclusive, compares to the 75th percentile.

    Args:
        bins (numpy.ndarray): Array representing the current load of each bin.
        batch_size (int): Number of balls placed in each iteration.
        m (int): Number of bins.
        k (int): The threshold comparison depth (1 for median only, 2 for median and 75th percentile).

    Returns:
        None. Modifies the `bins` array in-place by updating bin loads.
    """
    choices = np.random.choice(m, (batch_size, 2))
    median_load = np.median(bins)
    is_above_median = bins[choices] > median_load

    # Step 1: Choose bins based on median comparison
    primary_choice = np.where(
        is_above_median[:, 0] != is_above_median[:, 1],
        choices[np.arange(batch_size), np.argmin(is_above_median, axis=1)],
        -1 if k == 2 else np.random.choice(m, batch_size),
    )

    if k == 2:
        # For k=2, check the 75th percentile if both choices are either above or below median
        needs_secondary_check = primary_choice == -1
        top_75_percentile = np.percentile(bins, 75)
        is_in_top_75 = bins[choices[needs_secondary_check]] > top_75_percentile

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
