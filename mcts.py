import random
import math
from copy import deepcopy
from tqdm import tqdm
import plotly.graph_objects as go
from board import Board, Action

class MCTSNode:
    def __init__(self, board, move=None, parent=None):
        self.board = deepcopy(board)
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0
        self.depth = parent.depth + 1 if parent else 0  # Track the depth of the node

    def is_fully_expanded(self):
        return len(self.children) == len(self.board.get_legal_moves())

    def best_child(self, c_param=1.4):
        choices_weights = []
        for child in self.children:
            if child.visits > 0:
                exploitation = child.score / child.visits
                exploration = c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
                choices_weights.append(exploitation + exploration)
            else:
                choices_weights.append(float('inf'))  # Prioritize unvisited nodes
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        legal_moves = self.board.get_legal_moves()
        for move in legal_moves:
            new_board = deepcopy(self.board)
            Action(new_board).swap(*move[0], *move[1])
            child_node = MCTSNode(new_board, move, self)
            self.children.append(child_node)

    def simulate(self):
        current_board = deepcopy(self.board)
        for _ in range(20):  # Simulate up to 20 moves
            legal_moves = current_board.get_legal_moves()
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            Action(current_board).swap(*move[0], *move[1])
            current_board.update()
        return current_board.scoring_function(len(current_board.get_legal_moves()))

    def backpropagate(self, result):
        self.visits += 1
        self.score += result
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, board):
        self.root = MCTSNode(board)

    def search(self, iterations=1000):
        for _ in tqdm(range(iterations), desc="MCTS Search"):
            node = self.select_node()
            if not node.is_fully_expanded():
                node.expand()
            leaf = node.best_child()
            result = leaf.simulate()
            leaf.backpropagate(result)
        return self.root.best_child(c_param=0).move

    def select_node(self):
        node = self.root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def visualize_tree(self):
        nodes = []
        edges = []
        self._add_node(nodes, edges, self.root)
        
        node_labels = [f"Move: {node['move']}\nVisits: {node['visits']}\nScore: {node['score']}" for node in nodes]
        node_x = [node['x'] for node in nodes]
        node_y = [node['y'] for node in nodes]
        
        # Scale the node sizes by score and depth
        node_sizes = [node['score'] / (40000 / (3**node['depth'])) + 0.5 for node in nodes]
        
        edge_x = []
        edge_y = []
        for edge in edges:
            edge_x.extend([edge['x0'], edge['x1'], None])
            edge_y.extend([edge['y0'], edge['y1'], None])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='gray'),
            hoverinfo='none'
        ))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            text=node_labels,
            textposition='top center',
            marker=dict(size=node_sizes, color='lightblue'),
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='MCTS Tree',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
        
        fig.show()

    def _add_node(self, nodes, edges, node, x=0, y=0, dx=1):
        node_id = len(nodes)
        nodes.append({'id': node_id, 'move': node.move, 'visits': node.visits, 'score': node.score, 'x': x, 'y': y, 'depth': node.depth})
        if node.parent:
            parent_id = next(i for i, n in enumerate(nodes) if n['id'] == node.parent_id)
            edges.append({'x0': nodes[parent_id]['x'], 'y0': nodes[parent_id]['y'], 'x1': x, 'y1': y})
        for i, child in enumerate(node.children):
            child.parent_id = node_id  # Store the parent_id in the child node
            self._add_node(nodes, edges, child, x + i * dx - (len(node.children) - 1) * dx / 2, y - 1, dx / 2)



b = Board(10, 10)
b.fill_random()
b.update()
mcts = MCTS(b)
best_move = mcts.search(iterations=30)
b.display()
print(f"Best move: {best_move}")
print(b.display_move(best_move))  # Display the move on the board

# Visualize the MCTS tree if needed
mcts.visualize_tree()