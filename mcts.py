import math
import numpy as np 



class Node:

    def __init__(self, state, parent = None, action_taken = None, prior_prob = 0.0):

        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior_prob = prior_prob

        self.childre = {}
        #keep track of how many times the node has been visited
        self.visit_count = 0.0
        self.value_sum = 0.0
        self.q_value = 0.0

    def is_expanded(self):

        return len(self.childre) > 0

    def get_puct_score(self, c_puct = 1.0):
        """
            Helper method to compute the Predictor Upper Confidence bound applied to Trees.
            Here the balancement equation is defined
        
        """
        if self.visit_count == 0:
            q_val = 0.0

        else: 
            q_val = self.q_value

        u_val = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)


        return q_val + u_val



class MCTS:
    """
        Class to implement the tree search in order to find the return the optimal decision,
        requires an object GAME and the model
    """

    def __init__(self, game, model, num_simulations = 400, c_puct = 1.0)

        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, initial_state):

        root = Node(state = initial_state)

        for _ in range(self.num_simulations):

            node = root

            while node.is_expanded():
                action, node = self.select_best_child(node)

            is_terminal, reward = self.game_check_game_over(node.state)

            if is_terminal:
                value = reward

            else:

                policy_logits, value = self.model_predict(node.state)
                valid_moves = self.game.get_valid_moves(node.state)

                policy = policy_logits * valid_moves

                if np.sum(policy) > 0:

                    policy /= np.sum(policy)

                else:
                    policy = valid_moves / np.sum(valid_moves)


                for action, is_valid in enumerate(valid_moves):
                    if is_valid:
                        next_state = self.game.get_next_state(node.state, action)
                        node.children[action] = Node(
                            state = next_state,
                            parent = Node,
                            action_taken = action,
                            prior_prob = policy[action]
                        )

            self.backpropagate(node, value)

        #when the simulations are completed
        action_counts = np.zeros(self.game.action_size)

        for action, child in root.children.items():
            action_counts[action] = child.visit_count

        action_probs = action_counts / np.sum(action_counts)

        return action_probs


    def select_best_child(self, node):
        best_action = -1
        best_puct = -float('inf')
        best_child = None

        for action, child in node.children.items():
            puct_score = child.get_puct_score(self.c_puct)

            if puct_score > best_puct:

                best_puct = puct_score
                best_action = action
                best_child = child


        return best_action, best_child


    def backpropagate(self, node, value):
        """
            Helper method to climb up the search tree
        
        """

        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node.q_value = node.value_sum  / node.visit_count

            node = node.parent

            value = -value

    


