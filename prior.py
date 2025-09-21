import numpy as np
from sklearn import preprocessing
import torch
from scipy.stats import invwishart
import random
device = torch.device("cuda")

def sort( x, y,X_true, centers):
    distances = np.linalg.norm(x, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_x = x[sorted_indices]
    sorted_y = y[sorted_indices]
    sorted_X_true = X_true[sorted_indices]
    mapping = {}
    storage = set()
    curr = 0
    for i in range(len(sorted_y)):
        if len(mapping) == centers:
            break
        if sorted_y[i] not in storage:
            mapping[sorted_y[i]] = curr
            storage.add(sorted_y[i])
            curr += 1

    y_mapped = np.array([mapping[number] for number in sorted_y])
    indices = np.random.permutation(len(sorted_x))
    shuffled_x = sorted_x[indices]
    shuffled_y = y_mapped[indices]
    shuffled_X_true = sorted_X_true[indices]
    return shuffled_x, shuffled_y, shuffled_X_true


def generate_bayesian_gmm_data(
        batch_size=32,
        start_seq_len = 100,
        seq_len=500,
        num_features=2,
        min_classes=1,
        num_classes=10,
        weight_concentration_prior=1.0,  # Dirichlet prior for mixture weights
        mean_prior=0.0,  # Mean of prior over cluster means
        mean_precision_prior=0.01,  # Precision (confidence) over means
        degrees_of_freedom_prior=None,  # Degrees of freedom for Wishart prior
        covariance_prior=None,  # Scale matrix for Wishart prior
        seed=None,
        nan_frac = None,
        **kwargs
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    #features = random.randint(2, num_features)
    features = num_features
    if covariance_prior is None:
        covariance_prior = np.eye(features)

    if degrees_of_freedom_prior is None:
        degrees_of_freedom_prior = features

    seq_len = random.randint(start_seq_len, seq_len)
    clusters_x = np.zeros((batch_size, seq_len, num_features))
    clusters_y = np.zeros((batch_size, seq_len))
    clusters_x_true = np.zeros((batch_size, seq_len, num_features))
    batch_classes = []
    for i in range(batch_size):
        n_components = np.random.randint(min_classes, num_classes + 1)

        # Sample weights from Dirichlet prior
        weights = np.random.dirichlet(np.ones(n_components) * weight_concentration_prior)

        # Assign points to components proportionally
        counts = np.random.multinomial(seq_len, weights)

        # Sample cluster parameters
        means = []
        covariances = []
        for _ in range(n_components):
            Sigma = invwishart.rvs(df=degrees_of_freedom_prior, scale=covariance_prior)
            mu = np.random.multivariate_normal(np.full(features, mean_prior), Sigma / mean_precision_prior)
            means.append(mu)
            covariances.append(Sigma)
        # Sample points from GMM
        X = []
        y = []
        for k in range(n_components):
            n_k = counts[k]
            X_k = np.random.multivariate_normal(means[k], covariances[k], size=n_k)
            X.append(X_k)
            y.append(np.full(n_k, k))

        X = np.vstack(X)
        y = np.concatenate(y)
        X = np.pad(X, ((0, 0), (0, num_features - features)), mode='constant')

        x = preprocessing.MinMaxScaler().fit_transform(X)
        n_samples, n_features = seq_len, features

        if nan_frac:
            fraction_missing = np.random.uniform(0, nan_frac)
            n_elements = n_samples * n_features
            n_missing = int(fraction_missing * n_elements)

            if n_missing > 0:
                all_indices = np.arange(n_elements)
                np.random.shuffle(all_indices)
                chosen = all_indices[:n_missing]

                rows, cols = np.unravel_index(chosen, (n_samples, n_features))

                # Ensure no row is fully NaN
                full_rows = [r for r in np.unique(rows) if np.sum(rows == r) == n_features]
                to_keep = []
                for r in full_rows:
                    row_mask = rows == r
                    row_positions_in_chosen = np.where(row_mask)[0]
                    keep_pos = np.random.choice(row_positions_in_chosen)
                    to_keep.append(keep_pos)
                # Remove all "to_keep" indices from chosen at once
                chosen = np.delete(chosen, to_keep)
                rows, cols = np.unravel_index(chosen, (n_samples, n_features))
                x[rows, cols] = -2

        x, y, X_true = sort(x, y, X, n_components)
        clusters_x[i] = x
        clusters_y[i] = y
        clusters_x_true[i] = X_true
        batch_classes.append(n_components)

    clusters_x_true = torch.tensor(clusters_x_true, dtype=torch.float32).permute(1, 0, 2)
    clusters_x = torch.tensor(clusters_x, dtype=torch.float32).permute(1, 0, 2)
    clusters_y = torch.tensor(clusters_y, dtype=torch.float32).permute(1, 0)
    batch_classes = torch.tensor(batch_classes).unsqueeze(0)

    return clusters_x.to(device), clusters_y.to(device), clusters_x_true.to(device), batch_classes.to(device)


def match_mask(train_X, X_true):
    # Copy to avoid modifying original
    masked_X = X_true.clone()

    # Wherever train_X == -2, set X_true to NaN
    masked_X[train_X == -2] = float('nan')

    return masked_X


