from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    WikiCS,
    HeterophilousGraphDataset,
)
from ogb.nodeproppred import PygNodePropPredDataset


def load_planetoid_dataset(root, name):
    dataset = Planetoid(root=f"{root}/{name}", name=name)
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_computer_dataset(root):
    dataset = Amazon(root=f"{root}/Amazon", name="Computers")
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_photo_dataset(root):
    dataset = Amazon(root=f"{root}/Amazon", name="Photo")
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_cs_dataset(root):
    dataset = Coauthor(root=f"{root}/Coauthor", name="CS")
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_physics_dataset(root):
    dataset = Coauthor(root=f"{root}/Coauthor", name="Physics")
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_wikics_dataset(root, split_idx):
    dataset = WikiCS(root=f"{root}/WikiCS")
    data = dataset[0]

    if data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, split_idx]
    if data.val_mask.dim() == 2:
        data.val_mask = data.val_mask[:, split_idx]

    return data, dataset.num_node_features, dataset.num_classes


def load_roman_empire_data(root):
    dataset = HeterophilousGraphDataset(
        root=f"{root}/Heterophilous", name="Roman-empire"
    )
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_amazon_rating_data(root):
    dataset = HeterophilousGraphDataset(
        root=f"{root}/Heterophilous", name="Amazon-Photo"
    )
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_minesweeper_data(root):
    dataset = HeterophilousGraphDataset(
        root=f"{root}/Heterophilous", name="Minesweeper"
    )
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_tolokers_data(root):
    dataset = HeterophilousGraphDataset(root=f"{root}/Heterophilous", name="Tolokers")
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_questions_data(root):
    dataset = HeterophilousGraphDataset(root=f"{root}/Heterophilous", name="Questions")
    data = dataset[0]

    return data, dataset.num_node_features, dataset.num_classes


def load_ogbn_arxiv_data(root):
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=f"{root}/OGBN-Arxiv")
    data = dataset[0]
    data.split_idx = dataset.get_idx_split()

    return data, dataset.num_node_features, dataset.num_classes


def load_ogbn_products_data(root):
    dataset = PygNodePropPredDataset(name="ogbn-products", root=f"{root}/OGBN-Products")
    data = dataset[0]
    data.split_idx = dataset.get_idx_split()

    return data, dataset.num_node_features, dataset.num_classes
