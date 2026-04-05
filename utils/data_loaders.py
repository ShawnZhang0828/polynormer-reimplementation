from torch_geometric.datasets import Planetoid


def load_planetoid_dataset(root, name):
    dataset = Planetoid(root=f"{root}/{name}", name=name)
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes
