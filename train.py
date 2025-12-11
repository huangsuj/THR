
from GCN import *
from GPRGNN import *
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score
from utils import *
from dataset_utils import *

def RunExp(exp_i, args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate, pre_out, candidates):
        model.train()
        optimizer.zero_grad()
        out, pre_out = model(data, pre_out, candidates, 't')
        out = out[data.train_mask]
        if args.dataset.lower() in ['tolokers', 'questions']:
            loss = F.binary_cross_entropy_with_logits(out, data.y[data.train_mask].float())
        else:
            loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()

        optimizer.step()
        if args.rewire == 'THR':
            return pre_out
        else:
            del out


    def test(model, data, pre_out, candidates):
        model.eval()
        logits, pre_out = model(data, pre_out, candidates, 's')
        accs, losses, preds = [], [], []
        for index, (_, mask) in enumerate(data('val_mask', 'test_mask')):
            if args.dataset.lower() in ["tolokers", "questions"]:
                pred = logits[mask].squeeze()
                acc = roc_auc_score(data.y[mask].cpu(), pred.detach().cpu())
                loss = F.binary_cross_entropy_with_logits(logits[mask].squeeze(), data.y[mask].float())
            else:
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                loss = F.nll_loss(model(data, pre_out, candidates, 'r')[0][mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses, pre_out

    appnp_net = Net(dataset, args)
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    if args.dataset.lower() in ['penn94']:
        if args.original_split:
            if exp_i == 0:
                data.train_mask_arr = copy.deepcopy(dataset.data.train_mask)
                data.val_mask_arr = copy.deepcopy(dataset.data.val_mask)
                data.test_mask_arr = copy.deepcopy(dataset.data.test_mask)
            data.train_mask = data.train_mask_arr[:, exp_i % 5]
            data.val_mask = data.val_mask_arr[:, exp_i % 5]
            data.test_mask = data.test_mask_arr[:, exp_i % 5]
        else:
            data.train_mask, data.val_mask, data.test_mask = rand_train_test_idx(
                data, train_prop=args.train_rate, valid_prop=args.val_rate, curr_seed=exp_i + args.run_num)

    elif args.dataset.lower() in ['tolokers', 'questions']:
        if args.original_split:
            if exp_i == 0:
                data.train_mask_arr = copy.deepcopy(dataset.data.train_mask)
                data.val_mask_arr = copy.deepcopy(dataset.data.val_mask)
                data.test_mask_arr = copy.deepcopy(dataset.data.test_mask)
            data.train_mask = data.train_mask_arr[:, exp_i % 10]
            data.val_mask = data.val_mask_arr[:, exp_i % 10]
            data.test_mask = data.test_mask_arr[:, exp_i % 10]
        else:
            permute_masks = random_planetoid_splits
            data = permute_masks(data, dataset.num_classes, exp_i + args.run_num, percls_trn, val_lb)
    else:
        permute_masks = random_planetoid_splits
        data = permute_masks(data, dataset.num_classes, exp_i + args.run_num, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['GPRGNN']:
        if args.net == 'GPRGNN' and args.dataset == 'flickr':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{
                'params': model.lin1.parameters(),
                'weight_decay': args.weight_decay, 'lr': args.lr
            },
                {
                'params': model.lin2.parameters(),
                'weight_decay': args.weight_decay, 'lr': args.lr
            },
                {
                'params': model.prop1.parameters(),
                'weight_decay': 0.0, 'lr': args.lr
            }
            ],
                lr=args.lr)
    elif args.net in ['GCN']:
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=args.weight_decay,
                                     lr=args.lr
                                     )
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    counter = 0

    random_data = torch.randn(data.y.size(0), data.y.unique().size(0)).to(args.device)
    pre_out_train = random_data
    pre_out_test = random_data

    edge_index = data.edge_index.clone()
    num_nodes = data.num_nodes
    edge_index = edge_index.to(torch.long)
    edge_values = torch.ones(edge_index.shape[1], device=edge_index.device)
    adj_ori = torch.sparse_coo_tensor(edge_index, edge_values, (num_nodes, num_nodes))
    candidates = get_topk_candidates(data.x, adj_ori, k=args.add_edge)

    for epoch in range(args.epochs):
        pre_out_train = train(model, optimizer, data, args.dprate, pre_out_train, candidates)

        [val_acc, tmp_test_acc], preds, [
            val_loss, tmp_test_loss], pre_out_test = test(model, data, pre_out_test, candidates)


        if (args.early_stop_loss == 'acc' and (val_acc > best_val_acc)):
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        elif (args.early_stop_loss == 'loss' and (val_loss < best_val_loss)):
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc

            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if (epoch >= 0 and args.early_stop == True):
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, Gamma_0

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hyperparameters for training
    parser.add_argument('--net', type=str,
                        choices=['GCN', 'GPRGNN'],
                        default='GCN', help="Choose the GNN model to train")
    parser.add_argument('--epochs', type=int, default=10000,
                        help="Number of epochs to train (early stopping might change this")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for training 0.01")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay for training 5e-4")
    parser.add_argument('--early_stopping', type=int, default=200, help="Number of epochs to wait for early stopping")
    parser.add_argument('--early_stop', type=str, default=True, help="early stopping")
    parser.add_argument('--early_stop_loss', type=str, default='loss', choices=['acc', 'loss'],
                        help="Early stopping based on loss or accuracy")
    parser.add_argument('--hidden', type=int, default=32, help="Number of hidden units in the GNN")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for training")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the GNN")
    parser.add_argument('--data_split', default='dense', choices=['dense'], help="dense means 0.48/0.32/0.2")
    parser.add_argument("--train_rate", type=int, default=0.48, help="Train data ratio. Default is 0.48.")
    parser.add_argument("--valid_rate", type=int, default=0.32, help="Valid data ratio. Default is 0.32.")
    parser.add_argument('--set_seed', default=True, action=argparse.BooleanOptionalAction,
                        help="Set seed for reproducibility")
    parser.add_argument('--original_split', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Use original split from dataset")

    # GPRGNN specific hyperparameters
    parser.add_argument('--K', type=int, default=10, help="Number of hops (filter order) for GPRGNN")
    parser.add_argument('--alpha', type=float, default=0.1, help="Alpha for GPRGNN Initialization")
    parser.add_argument('--dprate', type=float, default=0.5, help="Dropout rate for GPRGNN")
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR', help="Initialization for GPRGNN")
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'], help="Choose the propagation method for GPRGNN")


    # Hyperparameters of data
    parser.add_argument('--dataset', default='questions', help="Dataset to use")
    parser.add_argument('--cuda', type=int, default=2, help="Which GPU to use")
    parser.add_argument('--RPMAX', type=int, default=10, help="Number of experiments to run (different seeds)")
    parser.add_argument('--run_num', type=int, default=0, help="Starting run number also first seed")
    parser.add_argument('--use_yaml', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Use yaml file for default parameters")
    parser.add_argument('--normalize_data', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Normalize the node features before training")
    parser.add_argument('--random_sort', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Randomly sort the nodes, old, kept for reproducibility")


    parser.add_argument('--rewire', type=str, default='THR', help="a rewiring method")

    parser.add_argument('--add_edge', type=int, default=2, help='The number of add edge of each node')
    parser.add_argument('--sampling_rate', type=float, default=0.2, help='')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    Graph_dataset = ['citeseer']
    args.use_yaml = True if args.use_yaml == "Yes" else False

    for index, item in enumerate(Graph_dataset):
        args.dataset = item

        if args.use_yaml:
            args = update_args_from_yaml(args)
        print('--------------Datasets: {}--------------------'.format(args.dataset))

        # Reproducibility
        if args.set_seed:
            torch.manual_seed(args.run_num)
            np.random.seed(args.run_num)

        # Data splits
        if args.data_split == "dense":
            args.train_rate = 0.48
            args.val_rate = 0.32

        # Convert string to boolean (needed for wandb sweeps)
        args.normalize_data = True if args.normalize_data == "Yes" else False
        args.random_sort = True if args.random_sort == "Yes" else False

        # nets
        gnn_name = args.net
        if gnn_name == 'GCN':
            if args.dataset.lower() in ['penn94']:
                Net = GCN_large
            elif args.dataset.lower() in ['tolokers', 'questions']:
                Net = HeteroGCN
            elif args.dataset.lower() == 'flickr':
                Net = GCNNet
            else:
                Net = GCN_Net
        elif gnn_name == 'GPRGNN':
            Net = GPRGNN

        dname = args.dataset
        dataset, data = DataLoader(dname, args.normalize_data)


        # Random sorting of the nodes
        if args.random_sort and (args.dataset.lower() not in ['penn94']):
            data = random_sort_nodes(data)
            dataset.data = data


        RPMAX = args.RPMAX
        Init = args.Init

        Gamma_0 = None
        alpha = args.alpha
        train_rate = args.train_rate
        val_rate = args.val_rate
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)

        args.C = len(data.y.unique())
        args.Gamma = Gamma_0
        # Use device
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device('cpu')

        Results0 = []

        for RP in tqdm(range(RPMAX), desc='Running Experiments'):
            test_acc, best_val_acc, Gamma_0 = RunExp(RP,
                    args, dataset, data, Net, percls_trn, val_lb)
            Results0.append([test_acc, best_val_acc])
            print("ACC:", test_acc)

        test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
        test_acc_std, val_acc_std = np.sqrt(np.var(Results0, axis=0)) * 100
        Results0_test = np.array(Results0)[:, 0] * 100
        Results0_val = np.array(Results0)[:, 1] * 100

        print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment with {args.rewire} rewiring:')
        print(
            f'val acc mean = {val_acc_mean:.4f} \t val acc std = {val_acc_std:.4f}')
        print(
            f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} ')

