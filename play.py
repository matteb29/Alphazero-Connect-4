import argparse

import numpy as np
import torch

from game import ConnectFourGame
from mcts import MCTS
from network import AlphaZeroNet


#symbols used to render the board: the first player is X, the second one is O
SYMBOLS = {0: ".", 1: "X", -1: "O"}


def print_board(board, columns):
    """
        Helper method to render an ABSOLUTE board (+1 = first player, -1 = second
        player) on the terminal. Columns are numbered from 1 for convenience.

    """
    print()
    print("  " + " ".join(str(c + 1) for c in range(columns)))

    for row in board:
        print("  " + " ".join(SYMBOLS[int(cell)] for cell in row))

    print()


def load_model(game, path, device):

    model = AlphaZeroNet(game)
    model.load_state_dict(torch.load(path, map_location = device))
    model.to(device)
    model.eval()

    return model


def ask_human_move(game, state):
    """
        Helper method to read a legal column from the keyboard.

    """
    valid_moves = game.get_valid_moves(state)
    legal = [c + 1 for c in np.nonzero(valid_moves)[0]]

    while True:
        answer = input(f"Your move, choose a column among {legal} (q to quit): ").strip()

        if answer.lower() in ("q", "quit", "exit"):
            return None

        if not answer.isdigit():
            print("Please type a number.")
            continue

        column = int(answer) - 1

        if column < 0 or column >= game.action_size or not valid_moves[column]:
            print("That column is not playable.")
            continue

        return column


def play(model_path, num_simulations, mode, human_first, verbose):

    game = ConnectFourGame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(game, model_path, device)

    mcts = MCTS(game, model, num_simulations = num_simulations)

    #state is always seen from the point of view of the player to move,
    #board is the absolute view used only for rendering
    state = game.get_initial_state()
    board = np.zeros_like(state)

    current_player = 1
    human_player = 1 if human_first else -1

    print(f"Loaded {model_path} on {device}, {num_simulations} MCTS simulations per move")
    print_board(board, game.columns)

    while True:

        is_human = mode == "human" and current_player == human_player

        if is_human:
            action = ask_human_move(game, state)

            if action is None:
                print("Game aborted.")
                return

        else:
            probs = mcts.search(state)
            action = int(np.argmax(probs))

            print(f"Player {SYMBOLS[current_player]} plays column {action + 1}")

            if verbose:
                formatted = ", ".join(
                    f"{c + 1}: {p:.2f}" for c, p in enumerate(probs) if p > 0
                )
                print(f"  visit distribution -> {formatted}")

        #record the move on the absolute board before flipping the perspective
        for r in range(game.raws - 1, -1, -1):
            if board[r][action] == 0:
                board[r][action] = current_player
                break

        state = game.get_next_state(state, action)
        current_player = -current_player

        print_board(board, game.columns)

        is_terminal, reward = game.check_game_over(state)

        if is_terminal:
            if reward == 0.0:
                print("Draw!")

            else:
                #reward is -1 from the point of view of the player to move,
                #so the winner is the one who has just played
                winner = -current_player

                if mode == "human":
                    print("You win!" if winner == human_player else "The model wins.")

                else:
                    print(f"Player {SYMBOLS[winner]} wins.")

            return


def main():

    parser = argparse.ArgumentParser(description = "Watch or challenge a trained AlphaZero Connect 4 model")
    parser.add_argument("--model", default = "best_model.pth", help = "path to the checkpoint to load")
    parser.add_argument("--sims", type = int, default = 200, help = "MCTS simulations per move")
    parser.add_argument("--mode", choices = ["human", "self"], default = "human",
                        help = "'human' to play against the model, 'self' to watch it play itself")
    parser.add_argument("--second", action = "store_true", help = "let the model move first")
    parser.add_argument("--verbose", action = "store_true", help = "print the MCTS visit distribution")

    args = parser.parse_args()

    play(args.model, args.sims, args.mode, not args.second, args.verbose)


if __name__ == "__main__":
    main()
