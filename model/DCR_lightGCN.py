from dgl.nn.pytorch import GraphConv
from model.base_model.LightGCN import LightGCN


class DCR_LG(LightGCN):
    def __init__(self, train_records, params, sys_params):
        super(DCR_LG, self).__init__(params, sys_params)
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = 64
        self.n_layers = 3

        if sys_params.rAdj:
            self.conv = GraphConv(64, 64, weight=False, bias=False, norm='right', allow_zero_in_degree=True)
            if sys_params.gamma == 100:
                self.conv = GraphConv(64, 64, weight=False, bias=False, norm='left', allow_zero_in_degree=True)
        else:
            self.conv = GraphConv(64, 64, weight=False, bias=False, allow_zero_in_degree=True)
        self.user_int_embedding = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_int_embedding = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.user_int_embedding.weight, std=0.01)
        nn.init.normal_(self.item_int_embedding.weight, std=0.01)

        self.user_inf_embedding = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_inf_embedding = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
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
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim * 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )
        self.psens_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim * 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )

        self.reg_loss_fn = nn.MSELoss()
        self.reg_coe = sys_params.reg_coe
        self.c_coe = sys_params.c_coe

    def computer_int(self, graph):
        users_emb = self.user_int_embedding.weight
        items_emb = self.item_int_embedding.weight
        layer_emb = torch.cat([users_emb, items_emb])
        embs = [layer_emb]
        for layer in range(self.n_layers):
            layer_emb = self.conv(graph, layer_emb)
            embs.append(layer_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer_inf(self, graph):
        users_emb = self.user_inf_embedding.weight
        items_emb = self.item_inf_embedding.weight
        layer_emb = torch.cat([users_emb, items_emb])
        embs = [layer_emb]
        for layer in range(self.n_layers):
            layer_emb = self.conv(graph, layer_emb)
            embs.append(layer_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def contrastive_loss(self, u_emb, pi_emb):
        sim_matrix = torch.matmul(u_emb, pi_emb.T)

        pos_sim = torch.exp(torch.diag(sim_matrix))
        neg_sim = torch.sum(torch.exp(sim_matrix), dim=1)

        total_loss = -torch.mean(torch.log(pos_sim / neg_sim))
        return total_loss

    def bpr_loss(self, graph, users, pos, neg):
        all_int_users, all_int_items = self.computer(graph)
        users_int_emb = all_int_users[users]

        pos_int_emb = all_int_items[pos]
        neg_int_emb = all_int_items[neg]

        all_inf_users, all_inf_items = self.computer(graph)
        users_inf_emb = all_inf_users[users]

        pos_inf_emb = all_inf_items[pos]
        neg_inf_emb = all_inf_items[neg]

        u_emb = torch.cat((users_int_emb, users_inf_emb), dim=1)
        pi_emb = torch.cat((pos_int_emb, pos_inf_emb), dim=1)
        ni_emb = torch.cat((neg_int_emb, neg_inf_emb), dim=1)

        pos_scores = torch.mul(u_emb, pi_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, ni_emb).sum(dim=1)

        pos_pop = self.pop_pred(pos_inf_emb)
        neg_pop = self.pop_pred(neg_inf_emb)
        p_sens = self.psens_pred(users_inf_emb)

        pos_scores = pos_scores * pos_pop * p_sens
        neg_scores = neg_scores * neg_pop * p_sens

        cf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        global_reg_loss = (self.reg_loss_fn(pos_pop, self.pop[pos.cpu()].to(self.device)) +
                           self.reg_loss_fn(neg_pop, self.pop[neg.cpu()].to(self.device))) / 2

        psen_reg_loss = self.reg_loss_fn(p_sens, self.pop_sens[users.cpu()].to(self.device))

        reg_loss = global_reg_loss + psen_reg_loss

        c_loss_t = self.contrastive_loss(users_int_emb, pos_int_emb)
        c_loss_f = self.contrastive_loss(users_inf_emb, neg_inf_emb)

        c_loss = c_loss_t + c_loss_f

        return cf_loss + reg_loss * self.reg_coe + self.c_coe * c_loss, users_int_emb, users_inf_emb, pos_int_emb, pos_inf_emb

    def get_user_ratings(self, users, graph):
        all_int_users, all_int_items = self.computer(graph)
        u_int_emb = all_int_users[users]

        all_inf_users, all_inf_items = self.computer(graph)
        u_inf_emb = all_inf_users[users]

        s = self.pop_sens[users.cpu()].to(self.device)
        s = s.unsqueeze(1).expand(u_inf_emb.shape[0], 64)

        p = self.pop.to(self.device)
        p = p.unsqueeze(1).expand(all_inf_items.shape[0], 64)

        u_inf_emb = u_inf_emb / torch.exp(s)
        all_inf_items = all_inf_items / torch.exp(p)

        u_emb = torch.cat((u_int_emb, u_inf_emb), dim=1)
        i_emb = torch.cat((all_int_items, all_inf_items), dim=1)

        scores = torch.matmul(u_emb, i_emb.T)

        pred_pop = self.pop_pred(all_inf_items).expand(scores.shape)
        pred_s = self.psens_pred(u_inf_emb)
        pred_sg = pred_s.view(-1, 1) * pred_pop

        real_p = self.p[users.cpu()].to(self.device)
        scores = scores * pred_sg - self.gamma * real_p

        return scores