# 🪰 Fruit Fly Connectome Lichess Bot (`openfly`)

A hybrid Lichess chess bot powered by PyTorch and biologically inspired sparse neural representations. The bot models its neural hidden dynamics on fruit fly (*Drosophila*) connectome sparse network graphs to evaluate board states, process legal chess moves, and learn directly from post-match outcomes.

---

## ✨ Features

* **Sparse Connectome Graph Architecture**: Implements sparse matrix propagation modeling synaptic activations across thousands of biological/synthetic fly neurons.
* **Strict Legal Move Masking**: Guarantees valid move outputs via standard UCI and SAN board transformation fallbacks.
* **Reinforcement Learning Engine**: Updates internal encoder and policy head weights dynamically based on match outcomes (Win: `+1.0`, Draw: `+0.2`, Loss: `-0.8`).
* **Elo Peak Cap Safeguard**: Automatically freezes weight updates if peak rating reaches or exceeds a pre-set rating limit (default: **3000 Elo**) so don’t get smart to fast .
* **Automatic Connectome Synchronization**: Attempts to pull real fruit fly connectome graph data (`fly_connectome.npz`) from external repositories or generates a local sparse graph fallback seamlessly.
* **In-Game Chat Command Interceptor**: Intercepts in-game user chat commands—type **`resign`** in Lichess match chat to gracefully resign and terminate the active game stream.
* **24-Hour Continuous Runner**: Built-in main loop designed to listen for incoming challenges and execute continuously for long-term Replit or cloud deployments.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed along with the required libraries.

```bash
pip install torch numpy scipy python-chess berserk requests
