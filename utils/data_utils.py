from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE

def tsne(embedding):
    tsne = TSNE(n_components=2)
    tsne.fit_transform(embedding)
    embedding = tsne.embedding_
    return embedding

def kmeans_selection(embedding,n):
    cluster_learner = KMeans(n_clusters=n)
    cluster_learner.fit(embedding)
    
    cluster_idxs = cluster_learner.predict(embedding)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embedding - centers)**2
    dis = dis.sum(axis=1)
    q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
    return q_idxs

