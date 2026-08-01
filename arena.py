import numpy as np


class Arena:

    def __init__(self, player1, player2, game):
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def play_game(self, verbose = False):

        state = self.game.get_initial_state()

        players = [self.player2, None, self.player1]

        current_player = 1
        step = 0

        while True:
            step += 1

            action = players[current_player + 1](np.copy(state))

            valid_moves = self.game.get_valid_moves(state)
            if not valid_moves[action]:
                raise ValueError(
                    f"Player {1 if current_player == 1 else 2} chose the illegal column {action}"
                )

            if verbose:
                print(f"Play {step}: Player {1 if current_player == 1 else 2} plays in column {action}")

            state = self.game.get_next_state(state, action)

            is_terminal, reward = self.game.check_game_over(state)

            if is_terminal:
                if verbose:
                    print(f"Game over in {step} plays")

                if reward == -1.0:
                    return current_player
                else:
                    return 0 #draw

            current_player = -current_player


    def play_games(self, num_games, verbose = False):

        p1_wins = 0
        p2_wins = 0
        draws = 0

        num_half = num_games // 2
        print("Starting arena, player 1 starts")

        for _ in range(num_half):
            result = self.play_game(verbose)

            if result == 1:
                p1_wins += 1
            if result == -1:
                p2_wins += 1
            if result == 0:
                draws += 1

        #switch the starter player for the other half of the games
        self.player1, self.player2 = self.player2, self.player1

        print("Now player2 starts")

        #after the swap the roles are inverted: a result of +1 means the player
        #sitting in the first seat won, which is now the original player 2
        for _ in range(num_games - num_half):

            result = self.play_game(verbose)

            if result == 1:
                p2_wins += 1
            if result == -1:
                p1_wins += 1
            if result == 0:
                draws += 1


        self.player1, self.player2 = self.player2, self.player1

        print(f"Final results: P1 (NEW NET): {p1_wins} | P2 (OLD NET): {p2_wins} | DRAWS: {draws}")

        return p1_wins, p2_wins, draws


    


        