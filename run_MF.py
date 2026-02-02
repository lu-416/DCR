import argparse
import time
import random

from model.CLUBSample import CLUBSample
from model.DCR_MF import *
from utils import *
TOP_K = 30

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


def parse_args(parser):
    parser.add_argument('--dataset', default='', type=str, help='dataset')
    parser.add_argument('--model', default='mf', type=str, help='base_model name')
    parser.add_argument('--gpu', default=7, type=int, help='gpu')
    parser.add_argument('--reg_coe', default=1e-3, type=int, help='reg')
    parser.add_argument('--c_coe', default=1, type=int, help='contrastive')
    parser.add_argument('--club_coe', default=1e-2, type=int, help='club_coe')
    parser.add_argument('--club_epoch', default=50, type=int, help='club_epoch')
    parser.add_argument('--gamma', default=100, type=float, help='coefficient of pop')
    parser.add_argument('--seed', default=2024, type=int, help='random seed')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--bs', default=8192, type=int, help='batch size')
    parser.add_argument('--debias_data', default=True, type=bool, help='whether use debiased test')
    parser.add_argument('--train', action='store_true', help='whether we train the base_model')

    sys_paras = parser.parse_args()

    return sys_paras

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def demo_sample(i_num, train_records):
    users = [u for u in train_records for _ in train_records[u]]
    pos_items = [pos_i for u in train_records for pos_i in train_records[u]]
    play_num = sum(len(train_records[x]) for x in train_records)
    neg_items = np.random.randint(0, i_num, play_num)

    return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)

def train_model(dataset, device, i_num, test_records, train_records, model, params, global_pop, club):
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
     
        users, posItems, negItems = demo_sample(i_num, train_records)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=params.bs):
            optimizer.zero_grad()
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            loss, u_int_emb, u_inf_emb, pi_int_emb, pi_inf_emb = model(u, i, n)
            loss = loss + (
                        club.forward(u_int_emb, u_inf_emb) + club.forward(pi_int_emb, pi_inf_emb)) * sys_paras.club_coe
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
        results = test_model(model, train_records, test_records, global_pop)
        ndcg = sum(results['ndcg'][TOP_Ks[1]]) / len(results['ndcg'][TOP_Ks[1]])
        if ndcg > best_res:
            best_res = ndcg
            best_epoch = epoch
            if not os.path.exists('/xx/checkpoints'):
                os.mkdir('/xx/checkpoints')
            torch.save(model.state_dict(), f'/xx/checkpoints/{params.model}-{dataset}.pt')
        if epoch - best_epoch > 5:
            break

        print('Epoch [{}/{}], Loss: {:.4f}, NDCG@{}: {:.4f}, Time: {:.2f}s'.format(epoch + 1, total_epoch,
                                                                                   total_loss / runs,
                                                                                   TOP_Ks[1],
                                                                                   ndcg, time.time() - tim1))
    print(f'Best NDCG@{TOP_Ks[1]}: {best_res:.4f}, Best epoch: {best_epoch}')



def read_dataset(dataset, debias=False):
    if 'ml' in dataset:
        return read_ml_dataset(dataset, debias)
    else:
        return read_dataset_else(dataset, debias)


def read_dataset_else(dataset, debias):
    u_num = 0
    i_num = 0
    train_records = collections.defaultdict(list)
    test_records = collections.defaultdict(list)
    file = open(f'/xx/dataset/{dataset}/train.txt', 'r')
    for line in file.readlines():
        ele = line.strip().split(' ')
        user, items = ele[0], ele[1:]
        u_num = max(u_num, int(user))
        for item in items:
            i_num = max(i_num, int(item))
            train_records[int(user)].append(int(item))
    file.close()
    if debias:
        file = open(f'/xx/dataset/{dataset}/balance_test.txt', 'r')
    else:
        file = open(f'/xx/dataset/{dataset}/test.txt', 'r')
    for line in file.readlines():
        ele = line.strip().split(' ')
        user, items = ele[0], ele[1:]
        # make sure all the users are appeared in training dataset.
        if int(user) in train_records:
            for item in items:
                i_num = max(i_num, int(item))
                test_records[int(user)].append(int(item))
    file.close()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {'num_users': u_num + 1,
              'num_items': i_num + 1,
              'device': device,
              'dataset': dataset}
    print(params)
    return test_records, train_records, device, i_num + 1, params


def read_ml_dataset(dataset, debias):
    train_records = collections.defaultdict(list)
    test_records = collections.defaultdict(list)

    file_m = open(f'/xx/dataset/{dataset}/movies.dat', 'r', encoding='utf-8')
    mid = 0
    uid = 0
    item_dict = dict()
    train_item_set = set()
    for line in file_m.readlines():
        ele = line.strip().split('::')
        item_dict[ele[0]] = mid
        mid += 1
    file_m.close()
    file = open(f'/xx/dataset/{dataset}/ratings.train', 'r')
    for line in file.readlines():
        ele = line.strip().split('::')
        train_records[int(ele[0]) - 1].append(item_dict[ele[1]])
        train_item_set.add(item_dict[ele[1]])
        if int(ele[0]) > uid:
            uid = int(ele[0])
    file.close()
    if debias:
        file = open(f'/xx/dataset/{dataset}/balance_ratings.test', 'r')
    else:
        file = open(f'/xx/dataset/{dataset}/ratings.test', 'r')
    for line in file.readlines():
        ele = line.strip().split('::')
        # make sure all the users and items are appeared in training dataset.
        if int(ele[0]) - 1 in train_records and item_dict[ele[1]] in train_item_set:
            test_records[int(ele[0]) - 1].append(item_dict[ele[1]])
    file.close()
    u_num = uid
    i_num = mid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {'num_users': u_num,
              'num_items': i_num,
              'device': device,
              'dataset': dataset}
    print(params)

    return test_records, train_records, device, i_num, params

def using_new_model(sys_paras):
    dataset = sys_paras.dataset
    test_records, train_records, device, i_num, params = read_dataset(dataset, sys_paras.debias_data)
    torch.cuda.set_device(sys_paras.gpu)

    emb_size = 64
    hidden_size = 64
    club = CLUBSample(emb_size, emb_size, hidden_size).to(device)
    model = DCR_MF(train_records, params, sys_paras).to(device)

    global_pop = F.normalize(cal_global_nov(train_records, i_num)[1].float(), dim=0).numpy()

    train_model(dataset, device, i_num, test_records, train_records, model, sys_paras, global_pop, club)

    if os.path.exists(f'/xx/checkpoints/{sys_paras.model}-{dataset}.pt'):
        model.load_state_dict(
            torch.load(f'/xx/checkpoints/{sys_paras.model}-{dataset}.pt', map_location=device),
            strict=False)

    time1 = time.time()
    res = test_model(model, train_records, test_records, global_pop)
    time2 = time.time()
    print(
        f"Inference time: {time2 - time1:.2f}s; #Interactions: {params['num_users'] * params['num_items']};"
        f" Avg time/interaction: {(time2 - time1) / (params['num_users'] * params['num_items']) * 1e9:.2f}ns.")
    # pdb.set_trace()

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

    using_new_model(sys_paras)