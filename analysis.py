import os
import pandas as pd
from board import state_to_board, Board, Action
from viz import Viz

def parse_board_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract top-level integers
    int_values = []
    structured_tuples = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('('):  # Detect tuples
            structured_tuples.append(eval(line))
        elif line.isdigit():  # Detect integers
            int_values.append(int(line))
        else:
            continue

    # Sort tuples by Attribute2 (descending order)
    structured_tuples = sorted(structured_tuples, key=lambda x: x[2], reverse=True)

    # Prepare metadata and moves as a flat dictionary for the DataFrame
    data = {f"Metadata_{idx + 1}": val for idx, val in enumerate(int_values)}
    for move_idx, move in enumerate(structured_tuples):
        data[f"Move_{move_idx + 1}_Coordinates"] = move[0]
        data[f"Move_{move_idx + 1}_Attribute1"] = move[1]
        data[f"Move_{move_idx + 1}_Attribute2"] = move[2]

    return data


def get_df_from_board_files(folder_path = "generated/v1"):
    # Process all files
    all_rows = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith("board_n_") and file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            file_data = parse_board_file(file_path)
            file_data['File'] = file_name  # Add column for source file
            all_rows.append(file_data)

#   Create a DataFrame
    combined_df = pd.DataFrame(all_rows)
    combined_df = rename_columns(combined_df)
    return combined_df

def rename_columns(df):
    # Rename columns
    df.columns = df.columns.astype(str)
    
    df.columns = df.columns.str.replace("Metadata_1", "Explo")
    df.columns = df.columns.str.replace("Metadata_2", "n_rollout")
    df.columns = df.columns.str.replace("Metadata_3", "n_simulation")
    df.columns = df.columns.str.replace("Metadata_4", "n_random")
    df.columns = df.columns.str.replace("Metadata_5", "state")
    
    
    # Rename Move columns
    new_column_names = {}
    for col in df.columns:
        if "Move_" in col:
            parts = col.split("_")
            move_number = parts[1]  # e.g., "1" from "Move_1_Coordinates"
            if "Coordinates" in col:
                new_column_names[col] = f"Move_{move_number}"
            elif "Attribute1" in col:
                new_column_names[col] = f"Move_{move_number}_N"
            elif "Attribute2" in col:
                new_column_names[col] = f"Move_{move_number}_Q"

    # Apply renaming
    df = df.rename(columns=new_column_names)
    return df

def visualize_row(df, id):
    """
    Not sure that's the best way to do it but its convienient for now
    """

    for i in range(1, 11):
        print(f"Move {i}: {df[df.index == id][f'Move_{i}'].values[0]}, N: {df[df.index == id][f'Move_{i}_N'].values[0]}, Q: {df[df.index == id][f'Move_{i}_Q'].values[0]}")
    state = df[df.index == id]["state"].values[0]
    board = state_to_board(state,7,7)
    a = Action(board)
    Viz(board, a).Visualize()
    # print all the moves