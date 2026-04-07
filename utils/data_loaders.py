import torch
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


def load_computer_dataset(root, seed=42):
    dataset = Amazon(root=f"{root}/Amazon", name="Computers")
    data = dataset[0]
    data = create_random_split(data, seed=seed)

    return data, dataset.num_node_features, dataset.num_classes


def load_photo_dataset(root, seed=42):
    dataset = Amazon(root=f"{root}/Amazon", name="Photo")
    data = dataset[0]
    data = create_random_split(data, seed=seed)

    return data, dataset.num_node_features, dataset.num_classes


def load_cs_dataset(root, seed=42):
    dataset = Coauthor(root=f"{root}/Coauthor", name="CS")
    data = dataset[0]
    data = create_random_split(data, seed=seed)

    return data, dataset.num_node_features, dataset.num_classes


def load_physics_dataset(root, seed=42):
    dataset = Coauthor(root=f"{root}/Coauthor", name="Physics")
    data = dataset[0]
    data = create_random_split(data, seed=seed)

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
        root=f"{root}/Heterophilous", name="Amazon-rating"
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


def load_dataset(name, root="data", split_idx=0, seed=42):
    name = name.lower()

    if name == "cora":
        return load_planetoid_dataset(root, "Cora")
    elif name == "computer":
        return load_computer_dataset(root, seed=seed)
    elif name == "photo":
        return load_photo_dataset(root, seed=seed)
    elif name == "cs":
        return load_cs_dataset(root, seed=seed)
    elif name == "physics":
        return load_physics_dataset(root, seed=seed)
    elif name == "wikics":
        return load_wikics_dataset(root, split_idx)
    elif name == "roman-empire":
        return load_roman_empire_data(root)
    elif name == "amazon-rating":
        return load_amazon_rating_data(root)
    elif name == "minesweeper":
        return load_minesweeper_data(root)
    elif name == "tolokers":
        return load_tolokers_data(root)
    elif name == "questions":
        return load_questions_data(root)
    elif name == "ogbn-arxiv":
        return load_ogbn_arxiv_data(root)
    elif name == "ogbn-products":
        return load_ogbn_products_data(root)

    raise ValueError(f"ARGUMENT ERROR: Unsupported dataset name: {name}")


def create_random_split(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
