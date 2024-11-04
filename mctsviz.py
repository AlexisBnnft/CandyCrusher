import matplotlib.pyplot as plt
import networkx as nx

def visualize_root_moves(mcts):
    """
    Visualizes the mean rewards of each legal move from the root state.
    
    Args:
        mcts (MCTS_CandyCrush): A trained instance of the MCTS_CandyCrush class with populated Q and N dictionaries.
    """
    # Extract root state and legal moves
    root_state = mcts.root_board.state()
    legal_moves = mcts.root_board.get_legal_moves()
    
    # Initialize directed graph
    G = nx.DiGraph()
    G.add_node("Root")
    
    # Dictionary for labels
    labels = {"Root": f"Root\n{hex(root_state)[:6]}"}
    
    # Store move details to determine the best move
    move_details = []
    mean_rewards = []
    
    # Add child nodes for each legal move
    for move in legal_moves:
        # Calculate mean reward for each legal move
        visit_count = mcts.N.get((root_state, move), 0)
        mean_reward = mcts.Q.get((root_state, move), 0)
        mean_rewards.append(mean_reward)

        # Create label for each move
        move_label = (
            f"Move {move}\n"
            f"Mean Reward: {mean_reward:.2f}\n"
            f"Visits: {visit_count}"
        )
        G.add_node(move_label)
        G.add_edge("Root", move_label)
        
        # Store move details for identifying the best move
        move_details.append((move_label, mean_reward))

        labels[move_label] = move_label  # Set label for node
    
    # Identify the best move based on the highest mean reward
    best_move_label = max(move_details, key=lambda x: x[1])[0] if move_details else None
    
    # Normalize mean rewards for node size scaling
    if mean_rewards:
        min_reward = min(mean_rewards)
        max_reward = max(mean_rewards)
        if min_reward != max_reward:
            normalized_sizes = [(reward - min_reward) / (max_reward - min_reward) for reward in mean_rewards]
        else:
            normalized_sizes = [1 for _ in mean_rewards]
    else:
        normalized_sizes = []

    # Scale node sizes
    node_sizes = [1500 * (size + 0.5) for size in normalized_sizes]  # Adding 0.5 to ensure minimum size

    # Create plot layout
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 7))

    # Draw nodes and edges, highlighting the best move in red
    node_colors = ["red" if label == best_move_label else "skyblue" for label in G.nodes]
    nx.draw(G, pos, with_labels=False, node_size=[1500] + node_sizes, node_color=node_colors, font_size=10, font_weight="bold")
    
    # Adjust positions for labels to be below the nodes
    label_pos = {key: (x, y - 0.2) for key, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=9)

      # Adjust axis limits to provide more space for labels
    plt.xlim(min(pos.values(), key=lambda x: x[0])[0] - 0.1, max(pos.values(), key=lambda x: x[0])[0] + 0.1)
    plt.ylim(min(pos.values(), key=lambda x: x[1])[1] - 0.3, max(pos.values(), key=lambda x: x[1])[1] + 0.1)

    # Title and display settings
    plt.title("Root Node with Legal Moves and Mean Rewards")
    plt.show()