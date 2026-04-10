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


# Select one of the mask splits from the dataset
def select_mask_split(data, split_idx=0):
    if hasattr(data, "train_mask") and data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, split_idx]
    if hasattr(data, "val_mask") and data.val_mask.dim() == 2:
        data.val_mask = data.val_mask[:, split_idx]
    if hasattr(data, "test_mask") and data.test_mask.dim() == 2:
        data.test_mask = data.test_mask[:, split_idx]

    return data


def load_wikics_dataset(root, split_idx=0):
    dataset = WikiCS(root=f"{root}/WikiCS", is_undirected=False)

    data = select_mask_split(dataset[0], split_idx)

    return data, dataset.num_node_features, dataset.num_classes


def load_roman_empire_data(root, split_idx=0):
    dataset = HeterophilousGraphDataset(
        root=f"{root}/Heterophilous", name="Roman-empire"
    )

    data = select_mask_split(dataset[0], split_idx)

    return data, dataset.num_node_features, dataset.num_classes


def load_amazon_rating_data(root, split_idx=0):
    dataset = HeterophilousGraphDataset(
        root=f"{root}/Heterophilous", name="Amazon-ratings"
    )

    data = select_mask_split(dataset[0], split_idx)

    return data, dataset.num_node_features, dataset.num_classes


def load_minesweeper_data(root, split_idx=0):
    dataset = HeterophilousGraphDataset(
        root=f"{root}/Heterophilous", name="Minesweeper"
    )

    data = select_mask_split(dataset[0], split_idx)

    return data, dataset.num_node_features, dataset.num_classes


def load_tolokers_data(root, split_idx=0):
    dataset = HeterophilousGraphDataset(root=f"{root}/Heterophilous", name="Tolokers")

    data = select_mask_split(dataset[0], split_idx)

    return data, dataset.num_node_features, dataset.num_classes


def load_questions_data(root, split_idx=0):
    dataset = HeterophilousGraphDataset(root=f"{root}/Heterophilous", name="Questions")

    data = select_mask_split(dataset[0], split_idx)

    return data, dataset.num_node_features, dataset.num_classes


def load_ogbn_arxiv_data(root):
    # Fix pytorch "weights only load failed" error.
    torch_load = torch.load

    def modified_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return torch_load(*args, **kwargs)

    # Temporarily turn off the weight only option
    torch.load = modified_torch_load

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=f"{root}/OGBN-Arxiv")

    torch.load = torch_load

    data = dataset[0]
    data.split_idx = dataset.get_idx_split()

    # [N, 1] -> [N]
    if data.y.dim() == 2 and data.y.size(1) == 1:
        data.y = data.y.squeeze(1)

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
        return load_wikics_dataset(root, 2)
    elif name == "roman-empire":
        return load_roman_empire_data(root, split_idx)
    elif name == "amazon-ratings":
        return load_amazon_rating_data(root, split_idx)
    elif name == "minesweeper":
        return load_minesweeper_data(root, split_idx)
    elif name == "tolokers":
        return load_tolokers_data(root, split_idx)
    elif name == "questions":
        return load_questions_data(root, split_idx)
    elif name == "ogbn-arxiv":
        return load_ogbn_arxiv_data(root)

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
