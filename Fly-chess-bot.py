import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta

import berserk
import chess
import numpy as np
import requests
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =====================================================================
# CONFIGURATION
# =====================================================================
LICHESS_TOKEN = os.environ.get("LICHESS_TOKEN", "lip_izbUSfnPIIZ0LAq3hdf8")
WEIGHTS_FILE = "fly_brain_weights.pth"
CONNECTOME_FILE = "fly_connectome.npz"

MAX_ELO_CAP = 3000       # Freeze learning parameters if peak rating >= 3000
RUN_DURATION_HOURS = 24  # Max continuous execution time in hours

# Global lock to safely update PyTorch model weights across threads
model_lock = threading.Lock()


# =====================================================================
# LOCAL CONNECTOME GENERATOR & PARSER
# =====================================================================

def ensure_fly_connectome(num_neurons=10000, density=0.0005):
    if os.path.exists(CONNECTOME_FILE):
        print(f" Found local connectome graph: '{CONNECTOME_FILE}'")
        return sp.load_npz(CONNECTOME_FILE)

    print(f" Generating biological sparse connectome locally ({num_neurons:,} neurons)...")
    adj_matrix = sp.random(
        num_neurons, 
        num_neurons, 
        density=density, 
        format='csr', 
        dtype=np.float32
    )
    sp.save_npz(CONNECTOME_FILE, adj_matrix)
    print(f" Connectome saved locally to '{CONNECTOME_FILE}'.")
    return adj_matrix


# =====================================================================
# 1. MOVE INDEX MAPPING & SENSORY ENCODER
# =====================================================================

def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    from_square = idx // 64
    to_square = idx % 64
    move = chess.Move(from_square, to_square)
    
    # Handle promotion automatically if legal
    if chess.Move(from_square, to_square, promotion=chess.QUEEN) in board.legal_moves:
        return chess.Move(from_square, to_square, promotion=chess.QUEEN)
    return move


def board_to_fly_sensory(board: chess.Board) -> np.ndarray:
    vector = np.zeros(781, dtype=np.float32)
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,   (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,  (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,   (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }

    for square, piece in board.piece_map().items():
        channel = piece_to_channel[(piece.piece_type, piece.color)]
        vector[square * 12 + channel] = 1.0

    vector[768] = 1.0 if board.turn == chess.WHITE else -1.0
    vector[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    vector[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    vector[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    vector[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        ep_col = chess.square_file(board.ep_square)
        vector[773 + ep_col] = 1.0

    return vector


# =====================================================================
# 2. CONNECTOME GRAPH MODEL & STRICT LEGAL MOVE SELECTION
# =====================================================================

class FlyBrainGraph(nn.Module):
    def __init__(self, adj_matrix_csr, sensory_neurons=500, motor_neurons=300):
        super().__init__()
        
        num_neurons = adj_matrix_csr.shape[0]
        print(f" Initializing FlyBrainGraph Neural Architecture ({num_neurons:,} neurons)...")

        self.num_neurons = num_neurons
        self.sensory_neurons = min(sensory_neurons, num_neurons // 2)
        self.motor_neurons = min(motor_neurons, num_neurons // 2)

        coo = adj_matrix_csr.tocoo()
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)

        self.W = torch.sparse_coo_tensor(indices, values, (num_neurons, num_neurons)).coalesce()
        self.W.requires_grad = False

        self.encoder = nn.Linear(781, self.sensory_neurons)
        self.policy_head = nn.Linear(self.motor_neurons, 4096)

    def forward(self, board_tensor, timesteps=4):
        batch_size = board_tensor.size(0)
        device = board_tensor.device

        h = torch.zeros((batch_size, self.num_neurons), device=device)
        sensory_activation = self.encoder(board_tensor)
        h[:, :self.sensory_neurons] = sensory_activation

        for _ in range(timesteps):
            synaptic_current = torch.sparse.mm(self.W, h.t()).t()
            h = F.relu(synaptic_current)

        motor_signals = h[:, -self.motor_neurons:]
        move_logits = self.policy_head(motor_signals)
        return move_logits


def get_fly_move(board: chess.Board, model: FlyBrainGraph) -> chess.Move:
    model.eval()
    board_vector = board_to_fly_sensory(board)
    board_tensor = torch.tensor(board_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(board_tensor).squeeze(0)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # Strict Legal Masking: Force non-legal moves to -infinity
    masked_logits = torch.full_like(logits, float('-inf'))
    move_map = {}
    
    for m in legal_moves:
        idx = move_to_index(m)
        masked_logits[idx] = logits[idx]
        move_map[idx] = m

    best_idx = torch.argmax(masked_logits).item()
    return move_map.get(best_idx, random.choice(legal_moves))


# =====================================================================
# 3. REPORTING & SELF-LEARNING ENGINE
# =====================================================================

def print_post_game_summary(status, elo_data, is_frozen, saved_to_file=False):
    peak_elo = elo_data['peak']
    
    print("\n" + "=" * 55)
    print("                MATCH COMPLETED SUMMARY                ")
    print("=" * 55)
    print(f" Game Outcome     : {status}")
    print("-" * 55)
    print(" Current Lichess Ratings:")
    print(f"   • Bullet       : {elo_data['bullet']} Elo")
    print(f"   • Blitz        : {elo_data['blitz']} Elo")
    print(f"   • Rapid        : {elo_data['rapid']} Elo")
    print(f"   • Classical    : {elo_data['classical']} Elo")
    print(f"   • Peak Rating  : {peak_elo} Elo")
    print("-" * 55)
    
    if is_frozen:
        print(f" Learning Status : 🔴 FROZEN (Peak Elo {peak_elo} ≥ {MAX_ELO_CAP})")
        print(" Action Taken    : Brain parameters frozen.")
    else:
        print(f" Learning Status : 🟢 ACTIVE (Peak Elo {peak_elo} < {MAX_ELO_CAP})")
        if saved_to_file:
            print(f" Action Taken    : Weights updated & saved to '{WEIGHTS_FILE}'")
        else:
            print(" Action Taken    : No position history recorded.")
    print("=" * 55 + "\n")


def get_all_elos(client):
    try:
        account = client.account.get()
        perfs = account.get('perfs', {})
        ratings = {
            'bullet': perfs.get('bullet', {}).get('rating', 0),
            'blitz': perfs.get('blitz', {}).get('rating', 0),
            'rapid': perfs.get('rapid', {}).get('rating', 0),
            'classical': perfs.get('classical', {}).get('rating', 0),
        }
        ratings['peak'] = max(ratings.values())
        return ratings
    except Exception:
        return {'bullet': 0, 'blitz': 0, 'rapid': 0, 'classical': 0, 'peak': 0}


def learn_from_game_experience(model, history, winner_color, bot_color, client):
    elo_data = get_all_elos(client)
    peak_elo = elo_data['peak']

    if winner_color == bot_color:
        reward = 1.0
        status_label = "WIN (+1.0 Reward)"
    elif winner_color is None:
        reward = 0.2
        status_label = "DRAW (+0.2 Reward)"
    else:
        reward = -0.8
        status_label = "LOSS (-0.8 Penalty)"

    if peak_elo >= MAX_ELO_CAP:
        print_post_game_summary(status_label, elo_data, is_frozen=True)
        return

    if not history:
        print_post_game_summary(status_label, elo_data, is_frozen=False, saved_to_file=False)
        return

    states, action_indices = zip(*history)
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    actions_tensor = torch.tensor(action_indices, dtype=torch.long)

    optimizer = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.policy_head.parameters()}
    ], lr=1e-3)

    with model_lock:
        model.train()
        optimizer.zero_grad()
        
        logits = model(states_tensor)
        cross_entropy = F.cross_entropy(logits, actions_tensor, reduction='none')
        loss = (cross_entropy * reward).mean()
        loss.backward()
        optimizer.step()

        torch.save(model.state_dict(), WEIGHTS_FILE)

    print_post_game_summary(status_label, elo_data, is_frozen=False, saved_to_file=True)


# =====================================================================
# 4. LICHESS STREAM HANDLERS & CHAT INTERCEPTOR
# =====================================================================

def handle_game_stream(client, game_id, model, bot_color):
    board = chess.Board()
    game_history = []
    print(f"\n[Game Started] Playing as {bot_color.upper()} (Game ID: {game_id})")

    try:
        for event in client.bots.stream_game_state(game_id):
            event_type = event.get('type')

            # --- A. CHAT COMMAND RESIGN TRIGGER ---
            if event_type == 'chatLine':
                username = event.get('username', '')
                text = event.get('text', '').strip().lower()

                if text == 'resign':
                    print(f"\n[Chat Command] '{username}' sent 'resign'. Bot resigning and closing game...")
                    try:
                        # FIX: Changed write_in_chat to post_message
                        client.bots.post_message(game_id, "Resign command received. Exiting game...")
                        client.bots.resign_game(game_id)
                    except Exception as e:
                        print(f"Error executing resign API call: {e}")
                    break

            # --- B. BOARD STATE & MOVE DECISION ENGINE ---
            elif event_type in ('gameFull', 'gameState'):
                state = event['state'] if event_type == 'gameFull' else event
                moves = state['moves'].split() if state['moves'] else []
                status = state.get('status')
                winner = state.get('winner')

                board.reset()
                for move_str in moves:
                    try:
                        board.push_uci(move_str)
                    except (ValueError, chess.IllegalMoveError):
                        try:
                            board.push_san(move_str)
                        except Exception as e:
                            print(f"Skipping unparseable move '{move_str}': {e}")

                is_my_turn = (board.turn == chess.WHITE and bot_color == 'white') or \
                             (board.turn == chess.BLACK and bot_color == 'black')

                if is_my_turn and not board.is_game_over():
                    sensory_vec = board_to_fly_sensory(board)
                    move = get_fly_move(board, model)
                    
                    if move is not None:
                        move_idx = move_to_index(move)
                        game_history.append((sensory_vec, move_idx))
                        print(f"Fly Brain move: {move.uci()}")

                        try:
                            client.bots.make_move(game_id, move.uci())
                        except Exception as e:
                            print(f"Failed to submit move: {e}")

                if status in ('mate', 'resign', 'timeout', 'stalemate', 'draw', 'outoftime'):
                    learn_from_game_experience(model, game_history, winner, bot_color, client)
                    break

    except (requests.exceptions.RequestException, berserk.exceptions.BerserkError) as e:
        print(f"[Game Stream Closed] {e}")


def listen_for_challenges(client, model, end_time):
    account_info = client.account.get()
    bot_id = account_info['id']
    print(f"Logged in as account: {account_info['username']}")

    event_stream = client.bots.stream_incoming_events()

    for event in event_stream:
        if datetime.now() >= end_time:
            print("\n[Timer Expired] 24-hour runtime limit reached.")
            break

        if event['type'] == 'challenge':
            challenge_id = event['challenge']['id']
            print(f"\nAccepting challenge: {challenge_id}")
            client.bots.accept_challenge(challenge_id)

        elif event['type'] == 'gameStart':
            game_id = event['game']['gameId']
            game_data = client.games.export(game_id)
            white_user = game_data.get('players', {}).get('white', {}).get('user', {})
            bot_color = 'white' if white_user.get('id') == bot_id else 'black'
            
            game_thread = threading.Thread(
                target=handle_game_stream, 
                args=(client, game_id, model, bot_color),
                daemon=True
            )
            game_thread.start()


def setup_lichess_account(token):
    headers = {"Authorization": f"Bearer {token}"}
    requests.post("https://lichess.org/api/bot/account/upgrade", headers=headers)
    session = berserk.TokenSession(token)
    return berserk.Client(session=session)


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    if not LICHESS_TOKEN or LICHESS_TOKEN == "lip_xxxxxxxxxxxxxx":
        print("Error: LICHESS_TOKEN is missing or invalid.")
        sys.exit(1)

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=RUN_DURATION_HOURS)

    print("=" * 60)
    print("      LICHESS FRUIT FLY BOT (STRICT LEGAL MOVE ENGINE)")
    print("=" * 60)
    print(f" Start Time   : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Scheduled End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Resign Trig  : Type 'resign' in match chat")
    print(" Controls     : Press Ctrl + C at any time to interrupt.")
    print("=" * 60 + "\n")

    bio_matrix = ensure_fly_connectome()
    fly_model = FlyBrainGraph(adj_matrix_csr=bio_matrix)

    if os.path.exists(WEIGHTS_FILE):
        print(f" Loading learned parameters from '{WEIGHTS_FILE}'...")
        fly_model.load_state_dict(torch.load(WEIGHTS_FILE))
    else:
        print(" No saved brain weights found. Initializing clean weights...")

    client = setup_lichess_account(LICHESS_TOKEN)

    try:
        while datetime.now() < end_time:
            try:
                listen_for_challenges(client, fly_model, end_time)
            except (requests.exceptions.RequestException, berserk.exceptions.BerserkError) as e:
                if datetime.now() >= end_time:
                    break
                print(f"\n[Network Notice] Connection lost ({e}). Reconnecting in 5s...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n" + "!" * 60)
        print(" [INTERRUPTED] Ctrl + C detected!")
        print(" Halting processes and exiting safely...")
        print("!" * 60)

    finally:
        print(f"\nBot session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(0)
