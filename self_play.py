import numpy as np
from mcts import MCTS


class SelfPlay:

    def __init__(self, game, model, num_episodes = 50, mcts_sims = 400):
        self.game = game
        self.model = model
        self.num_episodes = num_episodes
        self.mcts_sims = mcts_sims

    def execute_episode(self):

        train_example = []
        state = self.game.get_initial_state()


        current_player = 1
        step = 0

        #for each game create a montecarlo tree search
        mcts = MCTS(self.game, self.model, num_simulations = self.mcts_sims)

        while True:

            step += 1

            prob = mcts.search(state, add_noise = True)

            train_example.append([state, prob, current_player])

            if step < 15:

                action = np.random.choice(len(prob), p = prob)

            else:
                action = np.argmax(prob)

            state = self.game.get_next_state(state, action)

            #switch the player
            current_player *= -1 

            is_terminal, reward = self.game.check_game_over(state)


            if is_terminal:

                formatted_example = []

                for hist_state, hist_prob, hist_player in train_example:

                    actual_reward = reward if hist_player == current_player else -reward

                    formatted_example.append((hist_state, hist_prob, actual_reward))

                return formatted_example


    def generate_training_data(self):

        all_examples = []
        for ep in range(self.num_episodes):
            print(f"Generating data: playing game {ep + 1 } of {self.num_episodes}")

            episode_data = self.execute_episode()
            all_examples.extend(episode_data)


        print(f"Completed data generation! {len(all_examples)} have been generated")

        return all_examples

