import numpy as np

def compute_gap(bins, n, m):
    return np.max(bins) - n / m


def one_choice(bins, batch_size, m):
    choices = np.random.randint(0, m, size=batch_size)
    np.add.at(bins, choices, 1)


def two_choice(bins, batch_size, m):
    choices = np.random.choice(m, (batch_size, 2))
    selected_bins = np.argmin(bins[choices], axis=1)
    updates = choices[np.arange(batch_size), selected_bins]
    np.add.at(bins, updates, 1)


def beta_choice(bins, beta, batch_size, m):
    random_probs = np.random.rand(batch_size)

    one_choice_indices = np.where(random_probs < beta)[0]
    two_choice_indices = np.where(random_probs >= beta)[0]

    if two_choice_indices.size > 0:
        two_choice(bins, two_choice_indices.size, m)

    if one_choice_indices.size > 0:
        one_choice(bins, one_choice_indices.size, m)


def partial_information_k1(bins, batch_size, m):
    choices = np.random.choice(m, (batch_size, 2))
    median_load = np.median(bins)
    is_above_median = bins[choices] > median_load

    # Select bins: if one of the two bins is below the median, choose the lower-load bin
    selected_bins = np.where(
        is_above_median[:, 0] != is_above_median[:, 1],
        choices[np.arange(batch_size), np.argmin(is_above_median, axis=1)],
        np.random.choice(m, batch_size),  # Random choice if both are above median
    )
    np.add.at(bins, selected_bins, 1)


def partial_information_k2(bins, batch_size, m):
    choices = np.random.choice(m, (batch_size, 2))
    median_load = np.median(bins)
    top_75_percentile = np.percentile(bins, 75)
    is_above_median = bins[choices] > median_load

    # Step 1: Choose bins based on median comparison
    primary_choice = np.where(
        is_above_median[:, 0] != is_above_median[:, 1],
        choices[np.arange(batch_size), np.argmin(is_above_median, axis=1)],
        -1,  # -1 as a placeholder if both are the same
    )

    # Step 2: For balls where both choices are either above or below median, check 75th percentile
    needs_secondary_check = primary_choice == -1
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
    np.add.at(bins, primary_choice, 1)
