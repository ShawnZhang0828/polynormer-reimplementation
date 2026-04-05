def get_default_config():
    return {
        "seed": 42,
        "hidden_dim": 64,
        "n_local_layers": 2,
        "n_global_layers": 2,
        "n_heads": 8,
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "epochs": 20,
        "use_relu": False,
        "use_local_attention_network": True,
        "checkpoint_path": "checkpoints/model.pth",
    }
