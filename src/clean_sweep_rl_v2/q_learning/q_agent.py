import numpy as np
import pickle


class QLearningAgent:
    def __init__(self,
                 n_actions:     int   = 9,
                 alpha:         float = 0.3,
                 gamma:         float = 1.0,
                 epsilon:       float = 1.0,
                 epsilon_min:   float = 0.05,
                 epsilon_decay: float = 0.001):

        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # sparse Q-table
        self.q_table       = {}   
        self.total_updates = 0

    def _q(self, state) -> np.ndarray:
        # Fetching a state's Q-values, creating a zero row on first visit
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self.q_table[state]

    def select_action(self, state, valid_actions=None, greedy=False) -> int:
        # Epsilon-greedy action choice, restricted to valid actions
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        if not valid_actions:
            return 0
        
        # Explore: random valid action (skipped when greedy)
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploit: highest-Q valid action
        q = self._q(state)
        return max(valid_actions, key=lambda a: q[a])

    def update(self, state, action, reward, next_state, done, next_valid=None):
        # Q-learning update: move Q(s,a) toward reward + gamma * max_a' Q(s',a')
        q      = self._q(state)
        q_next = self._q(next_state)
        if done or not next_valid:
            target = reward   # terminal: no future value
        else:
            best   = max(next_valid, key=lambda a: q_next[a])  # greedy over valid next actions
            target = reward + self.gamma * q_next[best]
        q[action] += self.alpha * (target - q[action])   # TD update
        self.total_updates += 1

    def decay_epsilon(self):
        # Linearly decaying exploration down to epsilon_min
        self.epsilon = max(self.epsilon_min,
                           self.epsilon - self.epsilon_decay)

    @property
    def q_table_size(self): return len(self.q_table)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table,
                         "epsilon": self.epsilon,
                         "total_updates": self.total_updates,
                         "n_actions": self.n_actions}, f)
        print(f"[QLearningAgent] Saved → {path}  "
              f"({self.q_table_size} states, {self.total_updates:,} updates)")

    def load(self, path: str):
        # Restoring a saved Q-table for resuming or deployment
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.q_table       = d["q_table"]
        self.epsilon       = d["epsilon"]
        self.total_updates = d.get("total_updates", 0)
        self.n_actions     = d.get("n_actions", self.n_actions)
        print(f"[QLearningAgent] Loaded ← {path}  ({self.q_table_size} states)")
