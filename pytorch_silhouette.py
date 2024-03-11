
import torch
import torch.nn.functional as F


def get_silhouette_score(labels, all_dists):
    '''
    A Pytorch implementation of the Silhouette score.
    Accelerating the calculation by parallelizing using linear algebra.

    Modified from https://github.com/KevinMusgrave/pytorch-adapt/blob/1366df7c02
    0e3fbc4a44f4f35c13cfcc76334868/src/pytorch_adapt/layers/silhouette_score.py#L4

    Space Complexity Analysis:
        Notations: N is the number of samples.
        
        all_dists: O(N^2)
        cluster_mask: O(N^2)

    '''
    unique_labels = torch.unique(labels)
    num_samples = len(labels)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")
    # Although number of unique labels may not equal to `num_clusters`, each sample
    # always has a cluster label.
    assert all_dists.size(0) == all_dists.size(1)
    assert all_dists.size(0) == labels.size(0)

    # `cluster_mask` is crucial for Parallelization
    cluster_mask_bool = (labels.unsqueeze(1) == unique_labels)
    cluster_mask = F.normalize(1.0*cluster_mask_bool, p=1, dim=0) 

    scores = torch.zeros_like(labels).float()
    ## calculate the `b`
    sample2clu_dists = all_dists @ cluster_mask  # "num_samples" x "number of unique labels"
    sample2clu_dists_mask_self = sample2clu_dists + 1e5*cluster_mask_bool  # "num_samples" x "number of unique labels"
    sample_min_other_dists = torch.min(sample2clu_dists_mask_self, dim=1)[0]
    b = sample_min_other_dists  # `b` follows the standard definition in Silhouette Score
    ## calculate the `a` for the clusters with more than 1 elements
    sample2clu_dists_mask_other = sample2clu_dists[torch.nonzero(cluster_mask_bool, as_tuple=True)]
    num_elements = cluster_mask_bool.sum(dim=0)
    num_elements_sample_wise = (cluster_mask_bool * num_elements.unsqueeze(0)).sum(dim=1)
    mt1_mask = num_elements_sample_wise > 1  # "mt1" means more than 1
    # In line 3 below, we substract 1 from the number to exclude self distance
    sample2clu_dists_mask_other[mt1_mask] = sample2clu_dists_mask_other[mt1_mask] \
                                            * num_elements_sample_wise[mt1_mask] \
                                            / (num_elements_sample_wise[mt1_mask] - 1)
    a = sample2clu_dists_mask_other
    ## calculate the Silhouette Score
    # Set score=0 for all the samples assigned in the clusters which num_elements=0.
    # Here, do nothing because of the zero initialization of `scores`.
    scores_temp = (b - a) / (torch.maximum(a, b))
    scores[mt1_mask] = scores_temp[mt1_mask]
    silhouette = torch.mean(scores).item()

    return silhouette
