import argparse
import time
import random

from MF.run_MF import read_dataset, demo_sample, shuffle
from model.CLUBSample import CLUBSample
from model.DCR_lightGCN import *
from utils import *

TOP_K = 30

def parse_args(parser):
    parser.add_argument('--dataset', default='', type=str, help='dataset')
    parser.add_argument('--model', default='lightgcn', type=str, help='base_model name')
    parser.add_argument('--gpu', default=4, type=int, help='gpu')
    parser.add_argument('--sim_coe', default=30, type=int, help='#similar users')
    parser.add_argument('--reg_coe', default=1e-1, type=int, help='reg')
    parser.add_argument('--c_coe', default=1e-1, type=int, help='contrastive')
    parser.add_argument('--club_coe', default=1e-2, type=int, help='club_coe')
    parser.add_argument('--club_epoch', default=100, type=int, help='club_epoch')
    parser.add_argument('--gamma', default=400, type=float, help='coefficient of pop')
    parser.add_argument('--seed', default=2024, type=int, help='random seed')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--imp_n_layers', default=5, type=int, help='#layers in IMPGCN')
    parser.add_argument('--imp_n_classes', default=3, type=int, help='#classes in IMPGCN')
    parser.add_argument('--bs', default=8192, type=int, help='batch size')
    parser.add_argument('--rAdj', action='store_true', help='whether use r-AdjNorm')
    parser.add_argument('--debias_data', default=True, type=bool, help='whether use debiased test')
    parser.add_argument('--train', action='store_true', help='whether we train the base_model')
    parser.add_argument('--ablation', default='none', type=str, help='local or global')

    sys_paras = parser.parse_args()

    return sys_paras


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_train_graph(train_records, params):
    src = []
    dst = []
    u_i_pairs = set()
    for uid in train_records:
        iids = train_records[uid]
        for iid in iids:
            if (uid, iid) not in u_i_pairs:
                src.append(int(uid))
                dst.append(int(iid))
                u_i_pairs.add((uid, iid))
    u_num, i_num = params['num_users'], params['num_items']
    src_ids = torch.tensor(src)
    dst_ids = torch.tensor(dst) + u_num
    g = dgl.graph((src_ids, dst_ids), num_nodes=u_num + i_num)
    g = dgl.to_bidirected(g)
    return g

def train_model(dataset, device, i_num, test_records, train_records, model, params, graph, global_pop, club):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    optimizer_c = torch.optim.Adam(club.parameters(), lr=params.lr)
    total_epoch = 500
    best_epoch = 0
    best_res = 0.005
    for epoch in range(total_epoch):
        model.train()
        tim1 = time.time()
        total_loss = 0
        runs = 0
        # if params.cpr:
        #     users, posItems, negItems = demo_cpr_sample(train_records)
        # else:
        users, posItems, negItems = demo_sample(i_num, train_records)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=params.bs):
            optimizer.zero_grad()
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            loss, u_int_emb, u_inf_emb, pi_int_emb, pi_inf_emb = model.bpr_loss(graph, u, i, n)
            loss = loss + (club.forward(u_int_emb, u_inf_emb) + club.forward(pi_int_emb, pi_inf_emb)) * sys_paras.club_coe
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            runs += 1

            u_int_emb = u_int_emb.detach().clone()
            u_inf_emb = u_inf_emb.detach().clone()
            pi_int_emb = pi_int_emb.detach().clone()
            pi_inf_emb = pi_inf_emb.detach().clone()

            for _ in range(sys_paras.club_epoch):
                mi_loss = club.learning_loss(u_int_emb, u_inf_emb) + club.learning_loss(pi_int_emb, pi_inf_emb)
                optimizer_c.zero_grad()
                mi_loss.backward()
                optimizer_c.step()

        model.eval()
        results = test_model(model, train_records, test_records, global_pop, graph)
        ndcg = sum(results['ndcg'][TOP_Ks[1]]) / len(results['ndcg'][TOP_Ks[1]])
        if ndcg > best_res:
            best_res = ndcg
            best_epoch = epoch
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(model.state_dict(), f'/xx/checkpoints/{params.sim_coe}-{params.model}-{dataset}-{params.ablation}.pt')
        if epoch - best_epoch > 10:
            break

        print('Epoch [{}/{}], Loss: {:.4f}, NDCG@{}: {:.4f}, Time: {:.2f}s'.format(epoch + 1, total_epoch,
                                                                                   total_loss / runs,
                                                                                   TOP_Ks[1],
                                                                                   ndcg, time.time() - tim1))
    print(f'Best NDCG@{TOP_Ks[1]}: {best_res:.4f}, Best epoch: {best_epoch}')

def using_new_model(sys_paras):
    torch.cuda.set_device(sys_paras.gpu)
    dataset = sys_paras.dataset
    test_records, train_records, device, i_num, params = read_dataset(dataset, sys_paras.debias_data)
    params['rAdj'] = sys_paras.rAdj
    graph = create_train_graph(train_records, params).to(device)

    emb_size = 64
    hidden_size = 64
    model = DCR_LG(train_records, params, sys_paras).to(device)
    club = CLUBSample(emb_size, emb_size, hidden_size).to(device)

    global_pop = F.normalize(cal_global_nov(train_records, i_num)[1].float(), dim=0).numpy()
    train_model(dataset, device, i_num, test_records, train_records, model, sys_paras, graph, global_pop, club)

    if os.path.exists(f'/xx/checkpoints/{sys_paras.sim_coe}-{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt'):
        model.load_state_dict(
            torch.load(f'/xx/checkpoints/{sys_paras.sim_coe}-{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt', map_location=device),
            strict=False)
    elif os.path.exists(f'/xx/checkpoints/{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt'):
        model.load_state_dict(
            torch.load(f'/xx/checkpoints/{sys_paras.model}-{dataset}-{sys_paras.ablation}.pt', map_location=device),
            strict=False)
    else:
        model.load_state_dict(
            torch.load(f'/xx/checkpoints/{sys_paras.model}-{dataset}.pt', map_location=device),
            strict=False)

    time1 = time.time()
    res = test_model(model, train_records, test_records, global_pop, graph)
    time2 = time.time()
    print(
        f"Inference time: {time2 - time1:.2f}s; #Interactions: {params['num_users'] * params['num_items']};"
        f" Avg time/interaction: {(time2 - time1) / (params['num_users'] * params['num_items']) * 1e9:.2f}ns.")

    for i, topk in enumerate(TOP_Ks):
        print(f'Precision@{topk}: {sum(res["precision"][topk]) / len(res["precision"][topk]):.4f}  '
              f'Recall@{topk}: {sum(res["recall"][topk]) / len(res["recall"][topk]):.4f}  '
              f'NDCG@{topk}: {sum(res["ndcg"][topk]) / len(res["ndcg"][topk]):.4f} '
              f'ARP@{topk}: {sum(res["arp"][topk]) / len(res["arp"][topk]):.4f};  ')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for debias')
    sys_paras = parse_args(parser)
    set_random_seed(sys_paras.seed)
    print(f'Gamma: {sys_paras.gamma}')
    sys_paras.train = True

    using_new_model(sys_paras)
