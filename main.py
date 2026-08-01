import torch
import numpy as np
import copy

from game import ConnectFourGame
from mcts import MCTS
from self_play import SelfPlay
from train import Trainer
from arena import Arena
from network import AlphaZeroNet



def main():


    #set the hyperparameters
    NUM_ITERATIONS = 20
    NUM_EPISODES = 50
    MCTS_SIMS = 100
    ARENA_GAMES = 40
    WIN_THRESHOLD = 0.55

    print("Inizializing the Connected4 environment")

    game = ConnectFourGame()

    best_model = AlphaZeroNet(game)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)


    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"Iteration {iteration} out of {NUM_ITERATIONS}")

        print("Self-play phase: the agent is playing againt himself")
        self_play = SelfPlay(
            game, best_model, num_episodes = NUM_EPISODES, mcts_sims = MCTS_SIMS
        )
        training_data = self_play.generate_training_data()

        print("Training phase")
        new_model = AlphaZeroNet(game).to(device)
        new_model.load_state_dict(best_model.state_dict())

        trainer = Trainer(new_model, batch_size = 64, epochs = 10)
        trainer.train(training_data)

        print("Validation phase: the trained model plays against his previous version")

        mcts_new = MCTS(game, new_model, num_simulations=MCTS_SIMS)
        mcts_old = MCTS(game, best_model, num_simulations=MCTS_SIMS)


        def player_new(state):
            probs = mcts_new.search(state)
            return np.argmax(probs)

        def player_old(state):
            probs = mcts_old.search(state)
            return np.argmax(probs)

        arena = Arena(player_new, player_old, game)
        wins_new, wins_old, draws = arena.play_games(
            num_games=ARENA_GAMES, verbose = False
        )

        total_wins = wins_new + wins_old

        if total_wins == 0:
            win_rate = 0

        else:
            win_rate = wins_new / total_wins


        print(f"Win rate of the new model is {win_rate * 100:.1f}% (threshold is {WIN_THRESHOLD * 100}%)")

        #always keep a checkpoint of the latest trained network
        torch.save(new_model.state_dict(), "latest_model.pth")

        if win_rate > WIN_THRESHOLD:
            print("New model won :) ")

            best_model.load_state_dict(new_model.state_dict())
            torch.save(best_model.state_dict(), f"best_model_iteration{iteration}.pth")
            torch.save(best_model.state_dict(), "best_model.pth")

        else:
            print("New model lost :( ")


if __name__ == "__main__":
    main()

        