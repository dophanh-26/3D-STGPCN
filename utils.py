import torch 
import dgl
import numpy as np 
import scipy.sparse as sparse


def load_cora_dataset(verbose = True):
    dataset = dgl.data.CoraGraphDataset()
    if verbose:
        print(dataset.num_classes)
    return dataset, dataset[0]

def calculate_normalized_laplacian(A):
    N = A.shape[0]
    I = np.eye(N)
    D = np.sum(A, axis = 1).squeeze()
    D = np.diag(D)
    D_sqrt = np.linalg.inv(np.sqrt(D))
    return I - D_sqrt @ A @ D_sqrt

def laplacian_positional_encoding(L, k):
    eig_values, eig_vectors = np.linalg.eig(L)
    eig_ids = np.argsort(eig_values)
    lap_pos = eig_vectors[:, eig_ids][:, 1 : k + 1]
    return lap_pos

def generate_data(k):
    _, g = load_cora_dataset(verbose = False)
    A = sparse.coo_matrix.todense(g.adj(scipy_fmt = 'coo'))
    A = np.array(A)
    L = calculate_normalized_laplacian(A)
    lap_pos = laplacian_positional_encoding(L, k)
    lap_pos = np.real(lap_pos)
    g.ndata['lap_pos'] = torch.from_numpy(lap_pos).float()
    return g

def calculate_svd_pos(g, r = 100):
    g = g.add_self_loop()
    A = sparse.coo_matrix.todense(g.adj(scipy_fmt = 'coo'))
    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    v = vh.T
    sid = np.argsort(s)[::-1][: r]
    s = s[sid]
    u = u[:, sid]
    v = v[:, sid]
    s = np.sqrt(np.diag(s))
    p1 = np.dot(u, s)
    p2 = np.dot(v, s)
    return np.concatenate([p1, p2], axis = 1)

def train(graph, model, optimizer, loss_fn, num_epoch = 100):
    best_val_acc = 0.0
    best_test_acc = 0.0

    node_feature = graph.ndata['feat']
    lap_pos = graph.ndata['lap_pos']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    labels = graph.ndata['label']
    
    sign_flip = torch.rand(lap_pos.size(1))
    sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
    lap_pos = lap_pos * sign_flip.unsqueeze(0)

    graph.ndata['lap_pos'] = lap_pos

    for e in range(num_epoch):
        logits = model(graph, node_feature)
        pred = logits.argmax(1)

        loss = loss_fn(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc 

        if best_test_acc < test_acc:
            best_test_acc = test_acc 

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if e % 5 == 0:
            print("Epoch: {}, Loss: {:.4f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}, Test Accuracy: {:.4f}"
                .format(e, loss.item(), train_acc, val_acc, test_acc))

if __name__ == '__main__':
    #L = calculate_normalized_laplacian(A)
    #laplacian_positional_encoding(L, 2)
    g = generate_data(4)

    feat = g.ndata['feat']
    feat = torch.tensor(feat.numpy(), dtype = torch.long)
    