import numpy as np

class ConnectFourGame:
    """
        Class to implement the Connected4 game, using the following rules:
            - board is a 6x7 matrix
            - 0 is a void space
            - 1 is a piece of the current player
            - -1 is a piece of the opponent player
    
    """

    def __init__(self):

        self.raws = 6
        self.columns = 7
        self.action_size = self.columns

    def get_initial_state(self):

        return np.zeros((self.raws, self.columns), dtype = np.int8)

    def get_next_state(self, state, action):

        board = np.copy(state)

        for r in range(self.raws - 1, -1, -1):
            if board[r][action] == 0:
                board[r][action] = 1 #1 stands for the current player

                break

        #flip the sign of the board to go throught the next player
        return -board

    def get_valid_moves(self, state):

        return (state[0] == 0).astype(np.int8)

    def check_game_over(self, state):

        #case 1, the player who just moved the piece has won the game
        if self._check_win(state, player = -1):
            return True, -1.0

        #case 2, the players draw
        if len(self.get_valid_moves(state).nonzero()[0]) == 0:
            return True, 0.0

        #case 3, game is not finished
        return False, 0.0

    def _check_win(self, board, player):
        """
            Helper method to check if one of the two player has win the game
            by connected 4 pieces in one of raw, column or diagonal.
        
        """

        #orizontal check
        for c in range(self.columns - 3):
            for r in range(self.raws):
                if board[r][c] == player and board[r][c+1] == player and \
                   board[r][c+2] == player and board[r][c+3] == player:
                    return True

        #vertical check
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if board[r][c] == player and board[r+1][c] == player and \
                   board[r+2][c] == player and board[r+3][c] == player:
                    return True

        #positive diagonal
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if board[r][c] == player and board[r+1][c+1] == player and \
                   board[r+2][c+2] == player and board[r+3][c+3] == player:
                    return True

        #negative diagonal
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if board[r][c] == player and board[r-1][c+1] == player and \
                   board[r-2][c+2] == player and board[r-3][c+3] == player:
                    return True


        return False  

    

        