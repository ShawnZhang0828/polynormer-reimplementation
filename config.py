def get_default_config():
    return {
        "seed": 42,
        "hidden_dim": 64,
        "n_local_layers": 2,
        "n_global_layers": 2,
        "n_local_heads": 1,
        "n_global_heads": 8,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "local_to_global_epochs": 20,
        "warm_up_epochs": 5,
        "use_relu": False,
        "use_local_attention_network": True,
        "checkpoint_path": "checkpoints/model.pth",
    }
