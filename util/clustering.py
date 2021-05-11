import numpy as np
import torch
from tqdm import tqdm

# Modified from https://github.com/subhadarship/kmeans_pytorch

# ToDo: Can't choose a cluster if two points are too close to each other, that's where the nan come from

def clustering(gathered_features, device=torch.device('cpu'), threshold=1, K=20, Niter=300, tol=1e-4, distance='cosine', cluster_centers=[], step=0, min_cls=5):

    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    x = gathered_features # torch.cat([x.to(device) for x in gathered_features])
    cluster_ids_x, cluster_centers = kmeans(
            X=x, num_clusters=K, distance=distance, iter_limit=Niter, device=device, cluster_centers=cluster_centers, tol=tol)

    var = filter_centroid(gathered_features, cluster_ids_x, cluster_centers, K)

    print("{} feature's kmeans done.".format(len(x)))
    return cluster_ids_x, cluster_centers, var #mask

def filter_centroid(feature, cluster_ids_x, cluster_centers, K):
    distances = []
    for k in range(K):
        feat = feature[cluster_ids_x==k]
        cos_sim = feat * cluster_centers[k].view(1, -1)
        cos_sim = cos_sim / (feat.norm(p=2,dim=-1) * cluster_centers[k].norm(p=2,dim=-1) + 1e-16).view(-1,1)
        cos_sim = cos_sim.sum(-1)
        cos_dist = 1-cos_sim
        distances.append(cos_dist.sum() / (cos_dist.numel() + 1e-6))
    distances = torch.as_tensor(distances, device=distances[0].device)
    return distances

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers = [],
        tol=1e-4,
        tqdm_flag=True,
        iter_limit=0,
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)
    # initialize
    if type(cluster_centers) == list: #ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters)
    else:
        print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state) # should run on cpu o.w. OOM.

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            if selected.sum() == 0:
                continue
            initial_state[index] = selected.mean(dim=0)

            if torch.isnan(initial_state.sum()):
                print(initial_state, selected, choice_cluster)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster, initial_state


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / (A.norm(dim=-1, keepdim=True)  + 1e-6)
    B_normalized = B / (B.norm(dim=-1, keepdim=True) + 1e-6)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis
