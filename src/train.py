import torch
from data import load_dataset
from server import Server
import os
import random
import numpy as np
from logger import CSVLogger

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
torch.cuda.empty_cache()

def dirichlet_non_iid_split(user_seqs, num_clients, alpha=0.1, seed=42):
    """
    Extreme non-IID split using Dirichlet over users.
    alpha ↓ => more extreme heterogeneity
    """
    rng = np.random.default_rng(seed)

    # Collect all interactions
    interactions = []
    for uid, seq in enumerate(user_seqs):
        for it in seq:
            interactions.append((uid, it))

    # Dirichlet distribution over clients
    client_dist = rng.dirichlet([alpha] * num_clients)

    client_seqs = [[] for _ in range(num_clients)]

    for _, item in interactions:
        cid = rng.choice(num_clients, p=client_dist)
        client_seqs[cid].append(item)

    return client_seqs

# --- Configurations ---
config = {
    "dataset_path": "../dataset/CellPhone",
    "clients_per_round": 32,
    "rounds": 100,
    "local_epochs": 3,
    "batch_size": 128,
    "max_seq_len": 50,
    "hidden_units": 50,
    "num_heads": 1,
    "num_blocks": 2,
    "dropout": 0.2,
    "lr": 1e-3,           # Good learning rate for this setup
    "top_k": 20,
    "eval_every": 1,
    "save_dir": "checkpoints"
}

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and partition data
num_items, train_seqs, test_seqs, items_popularity, _ = load_dataset(config["dataset_path"])

# Filter users by min length
# min_len = 5 MATCHES min_samples = 4 in client.py (5 items -> 4 samples)
min_len = 5  
filtered_train, filtered_test = [], []
for u_seq, t_seq in zip(train_seqs, test_seqs):
    if len(u_seq) >= min_len:
        filtered_train.append(u_seq)
        filtered_test.append(t_seq)
train_seqs, test_seqs = filtered_train, filtered_test
print(f"Kept {len(train_seqs)} users after filtering (min_len={min_len})")

# Build client datasets
NON_IID_MODE = "dirichlet"   # ["iid", "dirichlet", "shard"]
DIRICHLET_ALPHA = 0.1       # try: 1.0 (mild), 0.5, 0.1, 0.05 (extreme)

num_clients = len(train_seqs)

if NON_IID_MODE == "iid":
    print(">>> Using IID client split")
    clients_data = [[seq] for seq in train_seqs]

elif NON_IID_MODE == "dirichlet":
    print(f">>> Using Dirichlet non-IID split (alpha={DIRICHLET_ALPHA})")
    non_iid_seqs = dirichlet_non_iid_split(
        train_seqs,
        num_clients=num_clients,
        alpha=DIRICHLET_ALPHA,
        seed=42
    )
    # each client gets ONE local sequence list
    clients_data = [[seq] for seq in non_iid_seqs if len(seq) > 0]

else:
    raise ValueError(f"Unknown NON_IID_MODE: {NON_IID_MODE}")

num_clients = len(clients_data)
clients_per_round = min(config["clients_per_round"], num_clients)

print(f"Total clients after non-IID split: {num_clients}")

# --- CRITICAL: ATTACK TARGET SELECTION ---
# Choose your intent: "promote" (boost item) or "demote" (suppress item)
attack_intent = "demote" 

if items_popularity is not None and len(items_popularity) > 1:
    if attack_intent == "demote":
        # To test demotion, we must target a HOT item to see it fall.
        # Select the most popular item.
        target_item = int(np.argmax(items_popularity[1:]) + 1)
        print(f"--- DEMOTION MODE: Targeting HOT item ID {target_item} (Count: {items_popularity[target_item]}) ---")
    else:
        # To test promotion, we should target a COLD item to see it rise.
        # Select a random item with low interactions (e.g., >1 but <50).
        candidates = np.where((items_popularity > 1) & (items_popularity < 50))[0]
        if len(candidates) > 0:
            target_item = int(np.random.choice(candidates))
        else:
            target_item = int(np.argmax(items_popularity[1:]) + 1)
        print(f"--- PROMOTION MODE: Targeting COLD item ID {target_item} (Count: {items_popularity[target_item]}) ---")
else:
    target_item = 1

# Model architecture
global_model_config = {
    "num_items": num_items,
    "max_seq_len": config["max_seq_len"],
    "hidden_units": config["hidden_units"],
    "num_heads": config["num_heads"],
    "num_blocks": config["num_blocks"],
    "dropout_rate": config["dropout"]
}

# Training hyperparameters
train_config = {
    "num_items": num_items,
    "max_seq_len": config["max_seq_len"],
    "batch_size": config["batch_size"],
    "lr": config["lr"],
    "local_epochs": config["local_epochs"],
    "top_k": config["top_k"],
    "eval_every": config["eval_every"],
    "save_dir": config["save_dir"],

    # contrastive / regularization knobs (Lowered for better convergence)
    "loss_type": "infonce_inbatch",
    "temperature": 0.2,
    "symmetric_infonce": True,
    "combine_bce": False,
    "contrastive_weight": 0.1,
    "lambda_tcr": 0.1,
    "lambda_seq_view": 0.1,
    "lambda_user_view": 0.1,
    "lambda_item_view": 0.1,
    "item_fgsm_eps": 0.1,

    # --- SERVER DEFENSE ---
    # Set to 0.0 initially to verify the attack effectiveness.
    # Once the attack is confirmed (ER@5 changes significantly), increase to 0.01.
    "lambda_sep": 0.01,   
    "lambda_var": 0.01,   

    # required by attacks.py
    "items_popularity": items_popularity.tolist() if items_popularity is not None else [0.0],

    # -------- Attack master config --------
    "attack": {
        "enabled": False, 
        
        # === SELECT ATTACK MODE HERE ===
        
        # [Option 1] Test DEMOTION (Suppression)
        # Ensure attack_intent = "demote" above.
        "modes": ["bandwagon", "psmu"],
        
        # [Option 2] Test PROMOTION (Bandwagon + Hard)
        # Ensure attack_intent = "promote" above.
        # "modes": ["bandwagon", "hard"],
        
        "combine_on_client": True, 
        
        # Number of attackers (approx 20-25% of selected clients)
        "num_malicious": max(1, min(8, clients_per_round)),
        
        # attack params:
        "target_item": target_item,
        "beta": 1.0,        # High beta for Demotion to force the loss down
        "seq_len": 40,
        "num_pseudo": 5,    # Number of fake sequences injected
        "topk_hot": 100,
        "alt_k": 8,
        "K_top": 40,
        
        # Weights for combined modes
        "weights": {"bandwagon": 1.0, "hard": 1.0, "demote": 1.0},
    },
}

# Start training
server = Server(global_model_config, train_config, clients_data, device)
server.federated_train(
    num_rounds=config["rounds"],
    clients_per_round=clients_per_round,
    user_seqs=train_seqs,
    test_seqs=test_seqs,
    itemnum=num_items
)