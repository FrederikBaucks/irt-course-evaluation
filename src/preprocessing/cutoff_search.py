import numpy as np


def entropy_continuous(X, threshold):
    # Discretize the data based on the threshold
    binary = X <= threshold

    # Calculate the proportions of data points in each bin
    p0 = np.mean(binary)
    p1 = 1 - p0

    # Calculate the entropy
    if p0 > 0 and p1 > 0:
        entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
    else:
        entropy = 0

    return entropy

def best_split_continuous(X, entropy=False):
    if entropy:
        min_entropy = float('inf')
        best_split = None

        # Sort the unique values
        values = np.sort(np.unique(X))
        print(values)
        # Calculate midpoints
        midpoints = (values[:-1] + values[1:]) / 2

        # Try all possible splits
        for midpoint in midpoints:
            entropy1 = entropy_continuous(X, midpoint)
            entropy2 = entropy_continuous(X, midpoint)

            # Calculate the weighted average of the entropies
            avg_entropy = (len(X[X <= midpoint]) * entropy1 + len(X[X > midpoint]) * entropy2) / len(X)

            if avg_entropy < min_entropy:
                min_entropy = avg_entropy
                best_split = midpoint

        return best_split, min_entropy
    elif entropy == False:
        min_variance = float('inf')
        best_split = None

        # Sort the unique values
        values = np.sort(np.unique(X))

        # Calculate midpoints
        midpoints = (values[:-1] + values[1:]) / 2

        # Try all possible splits
        for midpoint in midpoints:
            split1 = X <= midpoint
            split2 = X > midpoint

            variance1 = np.var(X[split1])
            variance2 = np.var(X[split2])

            # Calculate the weighted average of the variances
            avg_variance = (len(X[split1]) * variance1 + len(X[split2]) * variance2) / len(X)

            if avg_variance < min_variance:
                min_variance = avg_variance
                best_split = midpoint
        return best_split, min_variance
    else:
        #split using average
        return np.mean(X), 0

if __name__ == '__main__':

    # Example usage:
    X = np.array([1.2, 3.4, 5.6, 2.1, 3.3, 5.5, 2.3, 4.5, 1.1, 2.2, 5.4])

    split, entropy = best_split_continuous(X)

    print("Best split:", split)
    print("Entropy of the best split:", entropy)