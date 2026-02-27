import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from typing import Optional


class EnvConnect4(gym.Env):

    def __init__(self):
        super().__init__()

        self.num_rows = 6
        self.num_cols = 7

        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.MultiDiscrete([3] * (self.num_rows * self.num_cols)),
                "turn": gym.spaces.Discrete(n=2, start=1),
            }
        )

        self.action_space = gym.spaces.Discrete(self.num_cols)

        self.pos_value_to_name = {0: "-", 1: "X", 2: "O"}
        self.col_id_to_name = {i: f"col-{i}" for i in range(self.num_cols)}

        self.board = None
        self.turn = None
        self.count_moves = None

    # ---------- Core API helpers ----------

    def _get_obs(self):
        # Return a COPY so policies can't mutate env via observation
        return {"board": self.board.copy(), "turn": self.turn}

    def _get_info(self, winner=0, is_draw=False):
        return {
            "board": self.board.copy(),
            "turn": self.turn,
            "legal columns": self._get_legal_actions(),
            "count moves": self.count_moves,
            "winner": winner,
            "is_draw": is_draw,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.board = [0] * (self.num_rows * self.num_cols)
        self.turn = 1
        self.count_moves = 0

        return self._get_obs(), self._get_info()

    # ---------- Pure game logic ----------

    def _idx(self, row, col):
        return row * self.num_cols + col

    def _get_drop_row(self, col):
        for row in range(self.num_rows - 1, -1, -1):
            if self.board[self._idx(row, col)] == 0:
                return row
        return -1  # safer than None

    def _get_legal_actions(self):
        return [
            col for col in range(self.num_cols)
            if self.board[self._idx(0, col)] == 0
        ]

    def is_winner(self, mark):
        # Horizontal
        for row in range(self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row, col + i)] == mark for i in range(4)):
                    return True

        # Vertical
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols):
                if all(self.board[self._idx(row + i, col)] == mark for i in range(4)):
                    return True

        # Diagonal down-right
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row + i, col + i)] == mark for i in range(4)):
                    return True

        # Diagonal up-right
        for row in range(3, self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row - i, col + i)] == mark for i in range(4)):
                    return True

        return False

    # ---------- Gym step() ----------

    def step(self, action):
        legal = self._get_legal_actions()

        # Illegal move => acting player loses (cleanest)
        if action not in legal:
            terminated = True
            truncated = False
            reward = 1.0 if self.turn == 2 else -1.0  # reward from Player 1 perspective
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=0)

        self.count_moves += 1

        row = self._get_drop_row(action)
        if row == -1:
            # Defensive: should not happen due to legality check
            terminated = True
            truncated = False
            reward = 1.0 if self.turn == 2 else -1.0
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=0)

        self.board[self._idx(row, action)] = self.turn

        winner = 0
        is_draw = False

        # Win check
        if self.is_winner(self.turn):
            winner = self.turn
            terminated = True
            reward = 1.0 if self.turn == 1 else -1.0  # P1 perspective

        # Draw check
        elif len(self._get_legal_actions()) == 0:
            terminated = True
            is_draw = True
            reward = 0.0  # draw = 0 (MDP/Q-learning consistent)

        # Continue game
        else:
            terminated = False
            reward = 0.0  # no step penalty (consistent with your MDP/Q-learning write-up)
            self.turn = 2 if self.turn == 1 else 1

        truncated = False
        return self._get_obs(), reward, terminated, truncated, self._get_info(winner, is_draw)

    # ---------- Console helper ----------

    def print_current_board(self):
        print(f"Board after {self.count_moves} moves:")
        readable = [self.pos_value_to_name[v] for v in self.board]
        for r in range(self.num_rows):
            start = r * self.num_cols
            end = start + self.num_cols
            print(readable[start:end])
        print("[0, 1, 2, 3, 4, 5, 6]")

    def check(self):
        check_env(self)