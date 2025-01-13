import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from utils.defines import *

def get_unlabeled_index(tags):
    idxs_unlabeled = []
    for i, tag in enumerate(tags): 
        if tag==UNLABELED:
            idxs_unlabeled.append(i)
    return np.array(idxs_unlabeled)

def LeastConfidence(predictions, embeddings, tags, n):
    idxs_unlabeled = get_unlabeled_index(tags)
    predictions = predictions[idxs_unlabeled]
    U = np.max(predictions,axis=1)
    return idxs_unlabeled[np.argsort(U)[:n]]

def Entropy(predictions, embeddings, tags, n):
    idxs_unlabeled = get_unlabeled_index(tags)
    predictions = predictions[idxs_unlabeled]
    log_probs = np.log(predictions)
    U = (predictions*log_probs).sum(1)
    return idxs_unlabeled[np.argsort(U)[:n]]

def Margin(predictions, embeddings, tags, n):
    idxs_unlabeled = get_unlabeled_index(tags)
    predictions = predictions[idxs_unlabeled]
    probs_sorted = np.sort(predictions)[::-1]
    U = probs_sorted[:, 0] - probs_sorted[:,1]
    return idxs_unlabeled[np.argsort(U)[:n]]

def Coreset(predictions, embeddings, tags, n):
    idxs_unlabeled = get_unlabeled_index(tags)
    predictions = predictions[idxs_unlabeled]
    def furthest_first(X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs
    idxs_labeled = [i for i in np.arange(len(embeddings)) if i not in idxs_unlabeled ]
    chosen = furthest_first(embeddings[idxs_unlabeled, :], embeddings[idxs_labeled, :], n)
    return idxs_unlabeled[chosen]

def KMeansSampling(predictions, embeddings, tags, n):
    idxs_unlabeled = get_unlabeled_index(tags)
    embedding = embeddings[idxs_unlabeled]
    cluster_learner = KMeans(n_clusters=n)
    cluster_learner.fit(embedding)
    
    cluster_idxs = cluster_learner.predict(embedding)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embedding - centers)**2
    dis = dis.sum(axis=1)
    q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

    return idxs_unlabeled[q_idxs]
