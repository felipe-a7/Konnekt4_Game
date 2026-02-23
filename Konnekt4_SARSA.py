from typing import Optional


class PolicySARSA:

    def __init__(self):

        # Intialize lookup table for state-action values (Q-table)
        # First key: The board as a tuple (42 entries with 0=empty, 1=X, 2=O)
        # Second key: The decision as a column id between 0 and 6
        self.Q = dict()

    def _get_action(self, env, epsilon=0):
        """Returns an action based on an epsilon-greedy policy.

        Args:
            env: The environment action is taken in
            epsilon: The probability of taking a random action. If epsilon=0, the policy behaves greedy.

        Returns:
            A legal action
        """
        # Initialize Q values for the board if not available
        board = tuple(env._get_obs()["board"])
        if board not in self.Q:
            self.Q[board] = {}

        # Get legal actions for current period of the environment
        legal_actions = env._get_legal_actions()

        if epsilon > 0 and env.np_random.random() < epsilon:
            # Draw action among all legal actions with equal probability
            action = int(env.np_random.choice(legal_actions))

            # Initialize Q value for the board-action if not available
            if action not in self.Q[board]:
                self.Q[board][action] = 0

        else:
            # Initialize Q values for legal board-actions if not available
            for a in legal_actions:
                if a not in self.Q[board]:
                    self.Q[board][a] = 0

            # Select action with the highest value
            action = max(self.Q[board], key=self.Q[board].get)

        return action

    def _learn(self, env, epsilon, learning_rate, num_games, seed: Optional[int] = 0):
        """Learning of the (optimal) state-action values (=Q values).
        The learned values are stored in the dictionary self.Q.

        Args:
            env: a reinforcement learning environment with reset() and step().
            epsilon: exploration rate for the epsilon-greedy policy, must be 0 <= epsilon < 1.
            learning_rate: the learning rate when update step size.
            num_games: number of games to learn from. Must be integer.
            seed: The seed for the learning phase (this will be passed on to the environment).
        """

        # Initialize state-action value function
        self.Q = dict()

        # Initialize counter for number of games
        i = 0

        while i < num_games:
            # Reset environment
            obs, info = env.reset(seed=seed + i)

            # Reset board-action-reward sequence
            sequence = {
                1: {"board": [], "action": [], "reward": []},  # Player 1 (X)
                2: {"board": [], "action": [], "reward": []},  # Player 2 (O)
            }

            # Get information from initial state
            board = tuple(obs["board"])  # Convert to tuple (as lists can't be used as dict keys)
            turn = obs["turn"]

            # Flag to determine if the current game has ended
            terminated = False

            while not terminated:
                # Get action by following the policy
                action = self._get_action(env, epsilon)

                # Take the action in the environment and observe successor state and reward
                obs_tp1, reward, terminated, truncated, info = env.step(action)

                # Add to board-action-reward sequence
                sequence[turn]["board"].append(board)
                sequence[turn]["action"].append(action)
                sequence[turn]["reward"].append(reward)

                # Update board (as tuple) and turn
                board = tuple(obs_tp1["board"])
                turn = obs_tp1["turn"]

            # At the end of the game
            # Identify winner
            x_wins = env.is_winner(mark=1)
            o_wins = env.is_winner(mark=2)
            draw = True if not x_wins and not o_wins else False

            # Add final reward to sequence
            sequence[1]["reward"].append(1 if x_wins else 0.5 if draw else -2)
            sequence[2]["reward"].append(1 if o_wins else 0.5 if draw else -2)

            # Update state-action values (backward pass = based on the game's sequence)
            for turn, board_actions_reward in sequence.items():
                for idx, board in enumerate(board_actions_reward["board"]):
                    # Get action
                    action = board_actions_reward["action"][idx]

                    # Compute return from current and all future rewards
                    return_from_sequence = sum(reward for reward in sequence[turn]["reward"][idx:])

                    # Update state-action value
                    self.Q[board][action] += learning_rate * (return_from_sequence - self.Q[board][action])

            # Update number of games
            i += 1

            if i % 10000 == 0:
                print(f"Learning completed for {i}/{num_games} games.")


if __name__ == "__main__":
    from connect4 import EnvConnect4, PolicyRandom, play

    policy = None  # Against another human
    policy = PolicyRandom()  # Computer with a random strategy

    # Learn a policy via SARSA with backward pass.
    # Connect 4 has a much larger state space than Tic-Tac-Toe, so this may need many games.
    policy = PolicySARSA()
    policy._learn(env=EnvConnect4(), epsilon=0.4, learning_rate=0.1, num_games=50000)

    # Play the game
    play(env=EnvConnect4(), opponents_policy=policy)
