import math
from torch.optim.lr_scheduler import LambdaLR
import itertools
import torch
from sklearn.cluster import KMeans
import os
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import cluster
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda")
os.environ["OMP_NUM_THREADS"] = "1"

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def permute(num_classes):
    numbers = np.arange(num_classes)
    permutation = list(itertools.permutations(numbers, num_classes))
    return permutation


def map_labels(permutation, y): # y is of shape S, B and needs to be in shape S*B
    mapping = {key: value for key, value in enumerate(permutation)}
    mapping_tensor = torch.empty(max(mapping.keys()) + 1, dtype=torch.long, device=device)
    for key, value in mapping.items():
        mapping_tensor[key] = value
    y= y.long()
    targets = mapping_tensor[y].unsqueeze(1)
    targets = targets.reshape(-1).type(torch.LongTensor).to(device)
    return targets



def compute_accuracy_distribution(X, y, batch_classes,model=None,model_type=None, **kwargs):
    if model_type == 'transformer':
        return compute_accuracy_distribution_transformer(X, y,batch_classes,model, ** kwargs)
    elif model_type == 'gmm' or 'kmeans':
        return compute_accuracy_distribution_sklearn(X, y,batch_classes, model_type, **kwargs)
    else:
        print(f"Model type {model_type} does not exist")


def compute_accuracy_distribution_transformer(X, y, batch_classes, model, **kwargs):
    accuracy_buckets = np.zeros(11)
    # logits of shape (S,B,num_classes)
    logits, cluster_count_output = model(X.to(device),batch_classes)
    logits = logits.cpu()
    batch_classes = batch_classes.permute(1, 0)
    for batch_index, num_class in enumerate(batch_classes):
        targets = y[:, batch_index]
        prediction = logits[:, batch_index, :].argmax(dim=-1).cpu().numpy()  # logits of shape S,1
        unique_pred_classes = np.unique(prediction)
        if len(unique_pred_classes) == num_class:
            mapping  = {val: i for i, val in enumerate(unique_pred_classes)}
            mapped_prediction = [mapping[i] for i in prediction]
            mapped_labels = optimal_label_mapping(targets.cpu().numpy(), mapped_prediction)
            accuracy = accuracy_score(targets.cpu().numpy(), mapped_labels) * 100
        else:
            accuracy = 0
            permutations = permute(num_classes=num_class.cpu().numpy().item())
            for permutation in permutations:
                target = map_labels(permutation, targets)
                curr_accuracy = accuracy_score(target.cpu().numpy(), prediction) * 100
                if curr_accuracy > accuracy:
                    accuracy = curr_accuracy
            #accuracy = accuracy_score(targets.cpu().numpy(),prediction)
        acc_bucket = int(accuracy / 10)
        accuracy_buckets[acc_bucket] += 1
    return accuracy_buckets

def compute_accuracy_distribution_sklearn(X, y, batch_classes, model_type, **kwargs):
    accuracy_buckets = np.zeros(11)
    batch_classes = batch_classes.permute(1, 0)

    for batch_index, num_class in enumerate(batch_classes):
        num_class = num_class.cpu().item()
        if model_type == 'gmm':
            model =  GaussianMixture(n_components=num_class, random_state=42)
        elif model_type == 'kmeans':
            model = KMeans(n_clusters=num_class, random_state=42)
        else:
            print("Model not found")
        input_fit = X[: , batch_index, :].cpu()
        model.fit(input_fit)
        targets = y[:, batch_index].cpu().numpy()
        labels = model.predict(input_fit)
        mapped_labels = optimal_label_mapping(targets, labels)
        accuracy = accuracy_score(targets, mapped_labels) * 100
        acc_bucket = int(accuracy / 10)
        accuracy_buckets[acc_bucket] += 1
    return accuracy_buckets



def plot_accuracy_metric(accuracy_buckets, model_name):
    buckets = [f"{i * 10}-{(i + 1) * 10 -1}%" for i in range(10)]
    buckets.append("100%")
    plt.figure(figsize=(8, 5))
    plt.bar(buckets, accuracy_buckets, color='skyblue', edgecolor='black')
    plt.xlabel("Accuracy Ranges (%)")
    plt.ylabel("Frequency")
    plt.title(f"Accuracy Distribution Across Buckets {model_name}")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def optimal_label_mapping(true_labels, pred_labels):
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)
    n_classes = max(len(true_classes), len(pred_classes))

    cost_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    for i, true_class in enumerate(true_classes):
        for j, pred_class in enumerate(pred_classes):
            cost_matrix[i, j] = np.sum((true_labels == true_class) & (pred_labels == pred_class))

    cost_matrix = -cost_matrix  # Because we want to maximize matches

    # Solve assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create mapping: predicted class -> true class
    mapping = {pred_classes[j]: true_classes[i] for i, j in zip(row_ind, col_ind)}
    mapped_preds = np.array([mapping[label] for label in pred_labels])
    return mapped_preds



def compute_internal_metrics(X, y, metric):
    # X is of shape S,B,F
    # y is of Shape S, B
    batch_size = X.shape[1]
    scores = []
    for i in range(batch_size):
        curr_dataset = X[:,i, :]
        curr_labels = y[:, i]
        if np.unique(curr_labels).size != 1:
            if metric == 'dbi':
                scores.append(davies_bouldin_score(curr_dataset, curr_labels))
            elif metric == 'silhouette':
                scores.append(silhouette_score(curr_dataset, curr_labels))
            elif metric == 'ch':
                scores.append(calinski_harabasz_score(curr_dataset, curr_labels))
            else:
                print("metric not found/implemented")
    # plt.figure(figsize=(8, 5))
    # plt.hist(scores, bins=15, edgecolor="black", alpha=0.7)
    # plt.xlabel(f"{metric} index")
    # plt.ylabel("Frequency")
    # plt.title(f"{metric} Distribution Across {batch_size} Datasets")
    # plt.axvline(np.median(scores), color='red', linestyle='dashed', linewidth=2, label=f"median {metric}")
    # plt.legend()
    # plt.show()
    return np.array(scores)

def compute_external_metrics(y_true, y, metric):
    # purity, rand index, f-measure,
    batch_size = y_true.shape[1]
    scores = []

    for i in range(batch_size):
        y_true_curr = y_true[:, i]
        y_curr = y[:, i]
        if metric == 'purity':
            matrix = cluster.contingency_matrix(y_true_curr, y_curr)
            scores.append(np.sum(np.amax(matrix, axis=0)) / np.sum(matrix))
        elif  metric == 'rand_index':
            scores.append(rand_score(y_true_curr, y_curr))
        elif metric == 'fmi':
            scores.append(fowlkes_mallows_score(y_true_curr, y_curr))
        elif metric == 'nmi':
            scores.append(normalized_mutual_info_score(y_true_curr, y_curr))
        elif metric == 'ami':
            scores.append(adjusted_mutual_info_score(y_true_curr, y_curr))
        else:
            print("metric not found/implemented")

    # plt.figure(figsize=(8, 5))
    # plt.hist(scores, bins=15, edgecolor="black", alpha=0.7)
    # plt.xlabel(f"{metric} ")
    # plt.ylabel("Frequency")
    # plt.title(f"{metric} Distribution Across {batch_size} Datasets")
    # plt.axvline(np.median(scores), color='red', linestyle='dashed', linewidth=1, label=f"median {metric}")
    # plt.axvline(np.mean(scores), color='black', linestyle='dotted', linewidth=1, label=f"mean {metric}")
    #
    # plt.text(np.median(scores), plt.ylim()[1] * 0.9, f"Median: {np.median(scores):.2f}", color='red', ha='center', fontsize=10)
    # plt.text(np.mean(scores), plt.ylim()[1] * 0.85, f"Mean: {np.mean(scores):.2f}", color='black', ha='center', fontsize=10)
    # plt.legend()
    #plt.show()
    return np.array(scores)
def get_labels_bayesian_gmm(X,batch_classes=None,clusters = None, n_components=10,weight_concentration_prior=1.0,
                            mean_prior=None, mean_precision_prior=None,degrees_of_freedom_prior=None,n_init=None,
        covariance_prior=None,random_state=42):

    batch_size = X.shape[1]
    labels = np.zeros((X.shape[0], batch_size))
    for batch in range(batch_size):
        X_curr = X[:,batch,:]
        mask = (X_curr != 0).any(axis=0)  # Boolean mask for columns that are not all zero
        X_curr = X_curr[:, mask]
        num_features = X_curr.shape[1]
        curr_mean_prior = [0.0] * num_features
        curr_covariance_prior = np.eye(num_features)
        curr_degrees_of_freedom_prior = num_features
        cluster_num = clusters[batch]
        model = BayesianGaussianMixture(n_components=cluster_num,
                                        weight_concentration_prior_type='dirichlet_distribution',
                                        weight_concentration_prior=weight_concentration_prior,
                                        mean_prior=curr_mean_prior, mean_precision_prior=mean_precision_prior,
                                        degrees_of_freedom_prior=curr_degrees_of_freedom_prior,max_iter=1000,
                                        covariance_prior=curr_covariance_prior, random_state=random_state,n_init=n_init)
        model.fit(X_curr)
        prediction = model.predict(X_curr)
        labels[:, batch] = prediction
    return labels

def get_labels_gmm(X, batch_classes, random_state=42):
    batch_classes = batch_classes.squeeze(0)
    batch_size = X.shape[1]
    labels = np.zeros((X.shape[0], batch_size))
    for batch_index in range(len(batch_classes)):
        X_curr = X[:, batch_index, :]
        batch = batch_classes[batch_index].item()
        model = GaussianMixture(random_state=random_state, n_components = batch)
        model.fit(X_curr)
        prediction = model.predict(X_curr)
        labels[:, batch_index] = prediction
    return labels


def remap_labels(labels, probs):
    unique_labels = torch.unique(labels)
    label_mapping = {old.item(): new for new, old in enumerate(unique_labels)}

    remapped_labels = torch.tensor([label_mapping[int(l)] for l in labels], dtype=torch.long)
    remapped_probs = probs[:, unique_labels]
    return remapped_labels, remapped_probs



def get_cluster_scores_gmm(X, random_state=42):
    batch_size = X.shape[1]
    bic_scores = []
    aic_scores = []
    silhouette_scores = []
    for batch_index in range((batch_size)):
        X_curr = X[:, batch_index, :]
        best_bic = float("inf")
        best_bic_num = -1
        best_aic = float("inf")
        best_aic_num =-1
        best_sillhouette = float("-inf")
        best_sillhouette_num = -1
        for component in range(2, 11):
            model = GaussianMixture(random_state=random_state, n_components = component)
            model.fit(X_curr)
            predictions = model.predict(X_curr)
            current_bic  = model.bic(X_curr)
            current_aic  = model.aic(X_curr)
            current_sil  = silhouette_score(X_curr, predictions)
            if  current_bic < best_bic:
                best_bic = current_bic
                best_bic_num = component

            if current_aic < best_aic:
                best_aic = current_aic
                best_aic_num = component

            if current_sil > best_sillhouette:
                best_sillhouette = current_sil
                best_sillhouette_num = component

        bic_scores.append(best_bic_num)
        aic_scores.append(best_aic_num)
        silhouette_scores.append(best_sillhouette_num)

    return bic_scores, aic_scores, silhouette_scores

def get_clusters_bayesian_gmm(X,batch_classes=None, n_components=10,weight_concentration_prior=1.0,
                            mean_prior=None, mean_precision_prior=None,degrees_of_freedom_prior=None,
        covariance_prior=None, random_state=42, n_init = None):
    batch_size = X.shape[1]
    clusters = []
    for batch in range(batch_size):
        X_curr = X[:,batch,:]
        mask = (X_curr != 0).any(axis=0)  # Boolean mask for columns that are not all zero
        X_curr = X_curr[:, mask]
        features = X_curr.shape[1]
        #print(f"features: {features} , batch size: {batch_size}, X shape: {X_curr.shape}")
        curr_mean_prior = [0.0] * features
        curr_covariance_prior = np.eye(features)
        best_elbo = float('-inf')
        best_cluster = -1
        comps = n_components
        for component in range(1, comps + 1):
            model = BayesianGaussianMixture(n_components=component,
                                weight_concentration_prior_type='dirichlet_distribution',
                                weight_concentration_prior=weight_concentration_prior,
                                mean_prior=curr_mean_prior, mean_precision_prior=mean_precision_prior,
                                degrees_of_freedom_prior=degrees_of_freedom_prior,
                                covariance_prior=curr_covariance_prior,random_state=42)
            model.fit(X_curr)
            lower_bound = model.lower_bound_
            if lower_bound - np.log(math.factorial(component))> best_elbo:
                best_elbo = lower_bound - np.log(math.factorial(component))
                best_cluster = component
        clusters.append(best_cluster)
    return clusters

def get_nll_bayes(X,batch_classes, true_labels,clusters=None, n_components=10, weight_concentration_prior=1.0,
                  mean_prior=None, mean_precision_prior=None,
                  degrees_of_freedom_prior=None, covariance_prior=None,n_init = None,
                  random_state=42):
    batch_size = true_labels.shape[1]
    criterion = nn.NLLLoss()
    losses = []
    for batch_index in range(batch_size):
        X_curr = X[:, batch_index, :]
        mask = (X_curr != 0).any(axis=0)  # Boolean mask for columns that are not all zero
        X_curr = X_curr[:, mask]
        num_features = X_curr.shape[1]
        true_label = true_labels[:, batch_index].long()
        curr_mean_prior = [0.0] * num_features
        curr_covariance_prior = np.eye(num_features)
        curr_degrees_of_freedom_prior = num_features
        num_true_classes = len(torch.unique(true_label))
        cluster_num = clusters[batch_index]
        if cluster_num < num_true_classes:
            losses.append(float("inf"))
        else:
            model = BayesianGaussianMixture(n_components=cluster_num,
                                            weight_concentration_prior_type='dirichlet_distribution',
                                            weight_concentration_prior=weight_concentration_prior,
                                            mean_prior=curr_mean_prior, mean_precision_prior=mean_precision_prior,
                                            degrees_of_freedom_prior=curr_degrees_of_freedom_prior,max_iter=1000,
                                            covariance_prior=curr_covariance_prior, random_state=random_state,n_init=n_init)
            model.fit(X_curr)
            predicted_probs = model.predict_proba(X_curr)
            best_loss = float("inf")
            all_perms = list(itertools.permutations(range(cluster_num), num_true_classes))
            for perm in all_perms:
                permuted = torch.as_tensor(predicted_probs[:, perm])
                log_t = torch.log(permuted)
                loss = criterion(log_t, true_label).item()
                if loss < best_loss:
                    best_loss = loss
            losses.append(best_loss)
    return np.array(losses)


def get_nll_transformer(model, X, true_labels,batch_classes, n_components=5, condition=False):
    batch_size = true_labels.shape[1]
    criterion = nn.NLLLoss()
    losses = []
    for batch_index in range(batch_size):
        train_x = X[:, batch_index].unsqueeze(1)
        train_y = true_labels[:, batch_index]
        train_y = train_y.long()
        batch = batch_classes[batch_index].unsqueeze(-1)
        logits, cluster_output = model(train_x, torch.full(batch.shape,0, dtype=torch.long))
        cluster_output = cluster_output.view(-1, cluster_output.shape[2])
        predicted_index = cluster_output.argmax(dim=1).long().item()
        logits, cluster_output = model(train_x, torch.full(batch.shape, predicted_index + 1, dtype=torch.long))
        logits = logits.squeeze(1)

        predicted_probs = F.log_softmax(logits, dim=-1)
        num_true_classes = len(torch.unique(train_y))
        best_loss = float("inf")
        all_perms = list(itertools.permutations(range(n_components), num_true_classes))
        for perm in all_perms:
            permuted_part = torch.as_tensor(predicted_probs[:, perm])
            remaining_indices = [i for i in range(n_components) if i not in perm]
            remaining_part = torch.as_tensor(predicted_probs[:, remaining_indices])
            new_tensor = torch.cat([permuted_part, remaining_part], dim=1)
            loss_val = criterion(new_tensor, train_y).item()
            best_loss = min(best_loss, loss_val)
        losses.append(best_loss)

    return np.array(losses)