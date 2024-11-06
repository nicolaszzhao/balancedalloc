import numpy as np

# Function to compute the maximum load gap G_n
def compute_gap(bins, n, m):
    avg_load = n / m
    max_load = np.max(bins)
    return max_load - avg_load

# d-batched full per bin knowledge strategies
def one_choice(bins, m,  batch_size):
    choices = np.random.randint(0, m, size=batch_size)
    np.add.at(bins, choices, 1)

def two_choice(bins, m,  batch_size):
    snapshot = bins.copy()
    batch_updates = np.zeros_like(bins)
    
    for _ in range(batch_size):
        choices = np.random.choice(m, 2, replace=False)
        chosen_bin = choices[np.argmin(snapshot[choices])]
        batch_updates[chosen_bin] += 1
    
    bins += batch_updates

def beta_choice(bins, beta, m, batch_size):
    snapshot = bins.copy()
    batch_updates = np.zeros_like(bins)
    
    for _ in range(batch_size):
        if np.random.rand() < beta:
            choice = np.random.randint(0, m)
            batch_updates[choice] += 1
        else:
            choices = np.random.choice(m, 2, replace=False)
            chosen_bin = choices[np.argmin(snapshot[choices])]
            batch_updates[chosen_bin] += 1
    
    bins += batch_updates


# d-batched partial information strategies
def partial_information_k1(bins, m, batch_size):
    snapshot = bins.copy()
    batch_updates = np.zeros_like(bins)

    for _ in range(batch_size):
        choices = np.random.choice(m, 2, replace=True)
        median_load = np.median(snapshot)
        is_above_median = snapshot[choices] > median_load
        if is_above_median[0] != is_above_median[1]:
            chosen_bin = choices[np.argmin(is_above_median)]
        else:  
            chosen_bin = np.random.choice(choices)
        batch_updates[chosen_bin] += 1  
    bins += batch_updates

def partial_information_k2(bins, m, batch_size):
    snapshot = bins.copy()
    batch_updates = np.zeros_like(bins)

    for _ in range(batch_size):
        choices = np.random.choice(m, 2, replace=True)
        median_load = np.median(snapshot)
        is_above_median = snapshot[choices] > median_load

        if is_above_median[0] != is_above_median[1]:
            chosen_bin = choices[np.argmin(is_above_median)]
        else:
            top_75_percentile = np.percentile(bins, 75)
            is_in_top_75 = snapshot[choices] > top_75_percentile
            if is_in_top_75[0] != is_in_top_75[1]:
                chosen_bin = choices[np.argmin(is_in_top_75)]
            else:
                chosen_bin = np.random.choice(choices)
        batch_updates[chosen_bin] += 1
    bins += batch_updates
