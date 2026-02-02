import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor, i_num):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.i_num = i_num

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], torch.randint(0, self.i_num, ())

    def __len__(self):
        return self.user_tensor.size(0)


class BPRMF(nn.Module):
    def __init__(self, params, sys_params):
        super(BPRMF, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = 64

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)
        self.f = nn.Sigmoid()
        self.sys_params = sys_params

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        user_vec = self.user_embedding(user_indices)
        pos_item_vec = self.item_embedding(pos_item_indices)
        neg_item_vec = self.item_embedding(neg_item_indices)

        pos_scores = self.f(torch.mul(user_vec, pos_item_vec).sum(dim=1))
        neg_scores = self.f(torch.mul(user_vec, neg_item_vec).sum(dim=1))

        cf_loss = torch.mean((-1.0) * F.logsigmoid(pos_scores - neg_scores))

        return cf_loss

    def get_user_ratings(self, user_indices):

        return torch.matmul(self.user_embedding(user_indices), self.item_embedding.weight.T)

