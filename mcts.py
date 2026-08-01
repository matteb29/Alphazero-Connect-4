import math
import numpy as np
import torch


class Node:

    def __init__(self, state, parent = None, action_taken = None, prior_prob = 0.0):

        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior_prob = prior_prob

        self.children = {}
        #keep track of how many times the node has been visited
        self.visit_count = 0.0
        self.value_sum = 0.0
        self.q_value = 0.0

    def is_expanded(self):

        return len(self.children) > 0

    def get_puct_score(self, c_puct = 1.0):
        """
            Helper method to compute the Predictor Upper Confidence bound applied to Trees.
            Here the balancement equation is defined.

            NOTE: q_value is stored from the point of view of the player to move in THIS
            node, which is the opponent of the player choosing the move in the parent.
            That is why it is negated here.

        """
        if self.visit_count == 0:
            q_val = 0.0

        else:
            q_val = -self.q_value

        u_val = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)


        return q_val + u_val



class MCTS:
    """
        Class to implement the tree search in order to find the return the optimal decision,
        requires an object GAME and the model
    """

    def __init__(self, game, model, num_simulations = 400, c_puct = 1.0,
                 dirichlet_alpha = 0.8, dirichlet_eps = 0.25):

        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    @torch.no_grad()
    def predict(self, state):
        """
            Helper method to run the network on a single state and return
            a normalized policy (numpy array) plus a scalar value.

        """
        self.model.eval()

        device = next(self.model.parameters()).device
        tensor_state = torch.as_tensor(
            state, dtype = torch.float32, device = device
        ).unsqueeze(0).unsqueeze(0)

        policy_logits, value = self.model(tensor_state)

        policy = torch.softmax(policy_logits, dim = 1).squeeze(0).cpu().numpy()

        return policy, float(value.item())

    def search(self, initial_state, add_noise = False):

        root = Node(state = initial_state)

        for sim in range(self.num_simulations):

            node = root

            while node.is_expanded():
                action, node = self.select_best_child(node)

            is_terminal, reward = self.game.check_game_over(node.state)

            if is_terminal:
                value = reward

            else:

                policy, value = self.predict(node.state)
                valid_moves = self.game.get_valid_moves(node.state)

                policy = policy * valid_moves

                if np.sum(policy) > 0:

                    policy /= np.sum(policy)

                else:
                    policy = valid_moves / np.sum(valid_moves)

                if node is root and add_noise:
                    policy = self.apply_dirichlet_noise(policy, valid_moves)

                for action, is_valid in enumerate(valid_moves):
                    if is_valid:
                        next_state = self.game.get_next_state(node.state, action)
                        node.children[action] = Node(
                            state = next_state,
                            parent = node,
                            action_taken = action,
                            prior_prob = policy[action]
                        )

            self.backpropagate(node, value)

        #when the simulations are completed
        action_counts = np.zeros(self.game.action_size)

        for action, child in root.children.items():
            action_counts[action] = child.visit_count

        total = np.sum(action_counts)

        if total == 0:
            #the root is terminal, no legal move is available
            valid_moves = self.game.get_valid_moves(initial_state)
            return valid_moves / np.sum(valid_moves)

        action_probs = action_counts / total

        return action_probs

    def apply_dirichlet_noise(self, policy, valid_moves):
        """
            Helper method to add exploration noise on the root priors, as in the
            AlphaZero paper. Only the legal moves receive noise.

        """
        valid_idx = np.nonzero(valid_moves)[0]
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_idx))

        policy = np.copy(policy)
        policy[valid_idx] = (
            (1 - self.dirichlet_eps) * policy[valid_idx] + self.dirichlet_eps * noise
        )

        return policy


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
