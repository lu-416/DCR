import torch
import torch.nn as nn
import torch.nn.functional as F
class DCR_MF(nn.Module):
    def __init__(self, train_records, params, sys_params):
        super(DCR_MF, self).__init__()

        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = 64
        self.sys_params = sys_params

        self.user_int_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_int_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)

        nn.init.normal_(self.user_int_embedding.weight, std=0.01)
        nn.init.normal_(self.item_int_embedding.weight, std=0.01)

        self.user_inf_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_inf_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)

        nn.init.normal_(self.user_inf_embedding.weight, std=0.01)
        nn.init.normal_(self.item_inf_embedding.weight, std=0.01)

        self.pop = cal_global_nov(train_records, self.num_items)[1].cpu()
        self.pop[self.pop == 0] = 1
        self.pop = F.normalize(self.pop.float(), dim=0)

        self.pop_sens = cal_global_nov(train_records, self.num_items)[2].cpu()
        self.pop_sens[self.pop_sens == 0] = 1
        self.pop_sens = F.normalize(self.pop_sens.float(), dim=0)

        self.p = self.pop_sens.view(-1, 1) * self.pop

        self.gamma = sys_params.gamma
        self.pop_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim // 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )
        self.psens_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim // 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )

        self.reg_loss_fn = nn.MSELoss()
        self.reg_coe = sys_params.reg_coe
        self.c_coe = sys_params.c_coe

    def contrastive_loss(self, u_emb, pi_emb):
        sim_matrix = torch.matmul(u_emb, pi_emb.T)

        pos_sim = torch.exp(torch.diag(sim_matrix))
        neg_sim = torch.mean(torch.exp(sim_matrix), dim=1)

        total_loss = -torch.mean(torch.log(pos_sim / neg_sim))
        return total_loss


    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        u_int_emb = self.user_int_embedding(user_indices)
        pi_int_emb = self.item_int_embedding(pos_item_indices)
        ni_int_emb = self.item_int_embedding(neg_item_indices)

        u_inf_emb = self.user_inf_embedding(user_indices)
        pi_inf_emb = self.item_inf_embedding(pos_item_indices)
        ni_inf_emb = self.item_inf_embedding(neg_item_indices)

        u_emb = torch.cat((u_int_emb, u_inf_emb), dim=1)
        pi_emb = torch.cat((pi_int_emb, pi_inf_emb), dim=1)
        ni_emb = torch.cat((ni_int_emb, ni_inf_emb), dim=1)

        pos_scores = torch.mul(u_emb, pi_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, ni_emb).sum(dim=1)

        pos_pop = self.pop_pred(pi_inf_emb)
        neg_pop = self.pop_pred(ni_inf_emb)
        p_sens = self.psens_pred(u_inf_emb)

        pos_scores = pos_scores * pos_pop * p_sens
        neg_scores = neg_scores * neg_pop * p_sens

        cf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        pop_reg_loss = (self.reg_loss_fn(pos_pop, self.pop[pos_item_indices.cpu()].to(self.device)) +
                           self.reg_loss_fn(neg_pop, self.pop[neg_item_indices.cpu()].to(self.device))) / 2

        psens_reg_loss = self.reg_loss_fn(p_sens, self.pop_sens[user_indices.cpu()].to(self.device))

        reg_loss = pop_reg_loss + psens_reg_loss

        c_loss_t = self.contrastive_loss(u_int_emb, pi_int_emb)
        c_loss_f = self.contrastive_loss(u_inf_emb, pi_inf_emb)

        c_loss = c_loss_t + c_loss_f

        return cf_loss + reg_loss * self.reg_coe + self.c_coe * c_loss, u_int_emb, u_inf_emb, pi_int_emb, pi_inf_emb


    def get_user_ratings(self, user_indices):
        u_int_emb = self.user_int_embedding(user_indices)
        i_int_emb = self.item_int_embedding.weight

        u_inf_emb = self.user_inf_embedding(user_indices)
        i_inf_emb = self.item_inf_embedding.weight

        s = self.pop_sens[user_indices.cpu()].to(self.device)
        s = s.unsqueeze(1).expand(u_inf_emb.shape[0], 64)

        p = self.pop.to(self.device)
        p = p.unsqueeze(1).expand(i_inf_emb.shape[0], 64)

        u_inf_emb = u_inf_emb / ( 1 * torch.exp(s))
        i_inf_emb = i_inf_emb / ( 1 * torch.exp(p))

        u_emb = torch.cat((u_int_emb, u_inf_emb), dim=1)
        i_emb = torch.cat((i_int_emb, i_inf_emb), dim=1)

        scores = torch.matmul(u_emb, i_emb.T)

        pred_pop = self.pop_pred(i_inf_emb).expand(scores.shape)
        pred_s = self.psens_pred(u_inf_emb)
        pred_sg = pred_s.view(-1, 1) * pred_pop

        real_p = self.p[user_indices.cpu()].to(self.device)
        scores = scores * pred_sg - self.gamma * real_p

        return scores