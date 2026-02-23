import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from typing import Optional


class EnvConnect4(gym.Env):

    def __init__(self):
        # Initialize from gymansium environment
        super().__init__()

        # Board configuration (6 rows x 7 columns)
        self.num_rows = 6
        self.num_cols = 7

        # Define what the agent can observe (state space).
        # Every observation/state is a filled board (42 positions in total) and information about the turn.
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.MultiDiscrete([3] * (self.num_rows * self.num_cols)),  # Empty (0), X (1), or O (2)
                "turn": gym.spaces.Discrete(n=2, start=1),  # 1=X, 2=O
            }
        )

        # Dictionaries to make number-based information readable
        self.pos_value_to_name = {0: "-", 1: "X", 2: "O"}
        self.col_id_to_name = {
            0: "col-0",
            1: "col-1",
            2: "col-2",
            3: "col-3",
            4: "col-4",
            5: "col-5",
            6: "col-6",
        }

        # Define what actions are available (action space).
        # Action is the column id (0 to 6) where the player drops a piece.
        self.action_space = gym.spaces.Discrete(self.num_cols)

    def _get_obs(self):
        """Convert internal state to observation format."""

        return {"board": self.board, "turn": self.turn}

    def _get_info(self):
        """Compute auxiliary information for debugging."""

        return {
            "board": self.board,
            "turn": self.turn,
            "legal columns": self._get_legal_actions(),
            "count moves": self.count_moves,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Returns:
            tuple: (observation, info) for the initial period
        """
        # IMPORTANT: Must call this first to seed the random number generator (called via self.np_random)
        super().reset(seed=seed)

        # Initialize board (all empty) and turn (player 1 (X) always starts)
        self.board = [0] * (self.num_rows * self.num_cols)
        self.turn = 1

        # Reset moves counter
        self.count_moves = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _idx(self, row, col):
        return row * self.num_cols + col

    def _get_drop_row(self, col):
        """Return the target row index for a drop in this column, or None if full."""

        for row in range(self.num_rows - 1, -1, -1):
            if self.board[self._idx(row, col)] == 0:
                return row
        return None

    def step(self, action):
        """The step() method contains the core environment logic.
           It takes an action, updates the environment state, and returns the results.

        Args:
            action: The column where the player drops a piece

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if action not in self._get_legal_actions():
            raise ValueError(f"Illegal action {action}. Legal columns are {self._get_legal_actions()}.")

        # Update moves counter
        self.count_moves += 1

        # Update board by dropping piece to lowest available row in selected column
        row = self._get_drop_row(action)
        self.board[self._idx(row, action)] = self.turn

        # Reward: A small penalty to encourage efficiency.
        # Notice that the reward "at the end of the game" is not returned by the environment.
        reward = -0.01

        # Check for end of game (winner or no legal columns = draw)
        terminated = True if self.is_winner(mark=self.turn) or len(self._get_legal_actions()) == 0 else False

        # A step limit is not used in this game
        truncated = False

        # Update turn
        self.turn = 2 if self.turn == 1 else 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def is_winner(self, mark):
        """This method checks if the mark (1=X, 2=O) has won the game."""

        # Horizontal (4 in a row)
        for row in range(self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row, col + i)] == mark for i in range(4)):
                    return True

        # Vertical (4 in a row)
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols):
                if all(self.board[self._idx(row + i, col)] == mark for i in range(4)):
                    return True

        # Diagonal down-right (\)
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row + i, col + i)] == mark for i in range(4)):
                    return True

        # Diagonal up-right (/)
        for row in range(3, self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row - i, col + i)] == mark for i in range(4)):
                    return True

        return False

    def _get_legal_actions(self):
        """This method returns all legal actions of the current observation.

        Returns:
            list: list of non-full columns
        """

        legal_cols = []
        for col in range(self.num_cols):
            if self.board[self._idx(0, col)] == 0:
                legal_cols.append(col)
        return legal_cols

    def print_current_board(self):
        """This method prints the current board."""

        print(f"Board after {self.count_moves} moves:")
        readable_board = [self.pos_value_to_name[pos] for pos in self._get_obs()["board"]]

        for row in range(self.num_rows):
            start = row * self.num_cols
            end = start + self.num_cols
            print(readable_board[start:end])
        print("[0, 1, 2, 3, 4, 5, 6] (column ids)")

    def check(self):
        """This method catches many common issues with the Gymnasium environment."""

        try:
            check_env(self)
            print("Environment passes all checks!")
        except Exception as e:
            print(f"Environment has issues: {e}")


class PolicyRandom:

    def __init__(self):
        pass

    def _get_action(self, env):
        """Returns an action based on a policy in which legal actions are selected with equal probability."""

        legal_actions = env._get_legal_actions()
        action = env.np_random.choice(legal_actions)
        return action


def play(env, opponents_policy=None):
    """
    Enables a game of Connect 4 where the user plays as Player X via console input.
    The opponent's policy can be either human (console input) or an automated policy.

    Args:
        env: The game environment that follows a standard gymnsasium interface with reset(), step(), and print_current_board() methods.
        opponents_policy: Optional; an object with a _get_action() method that determines the opponent's moves automatically.
                          If None, the opponent will be controlled via console input.
    """
    keep_playing = True

    while keep_playing:
        # Reset environment to start a new game episode
        observation, info = env.reset()

        # Flag to determine if the current game has ended
        episode_over = False

        while not episode_over:
            # Display the current state of the game board
            env.print_current_board()

            # Prompt the user (Player X) for a move
            action = int(input(f"Your (valid) move (column number) as player {env.pos_value_to_name[env.turn]}: "))

            # Apply the player's move to the environment
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

            if not episode_over and opponents_policy is not None:
                # Automated opponent's turn based on the supplied policy
                action = opponents_policy._get_action(env)
                print(f"Other player's ({env.pos_value_to_name[env.turn]}) move: {action} ({env.col_id_to_name[action]})")

                # Execute opponent's move
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated

            elif not episode_over and opponents_policy is None:
                # Manual opponent move
                env.print_current_board()
                action = int(input(f"Other player's (valid) move (column number) as player {env.pos_value_to_name[env.turn]}: "))
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated

        # After the game ends, display final board
        env.print_current_board()

        # Determine and display the game result
        if env.is_winner(mark=1):
            result = "X wins!"
        elif env.is_winner(mark=2):
            result = "O wins!"
        else:
            result = "Draw"
        print(f"=> Result: {result}")

        # Close the environment
        env.close()

        # Ask user whether to continue playing
        user_input = input("Continue playing (y=yes, n=no): ").strip().lower()
        keep_playing = True if user_input == "y" else False


if __name__ == "__main__":
    # Check environment
    # env = EnvConnect4()
    # env.check()

    # Play against each other (both via console input)
    play(env=EnvConnect4(), opponents_policy=None)

    # Play against a computer following a random strategy.
    # policy = PolicyRandom()
    # play(env=EnvConnect4(), opponents_policy=policy)
