# Card RL Environment

A simple 4-player, team-based card game environment for RL experiments.

- 52-card deck is randomly dealt to 4 players (13 cards each) at reset.
- Players 0 and 2 are Team A; Players 1 and 3 are Team B.
- Each trick/round: all four players play one card. The highest rank wins the trick for their team.
- Agent controls Player 0; Players 1-3 play random valid cards.
- A dummy reward function is called at the end of each trick; replace with your shaping later.

## API

- `env = CardGameEnv(seed: Optional[int] = None)`
- `obs = env.reset(seed: Optional[int] = None)` -> observation dict
- `obs, reward, terminated, truncated, info = env.step(action)` where action is an integer representing a card from the agent's current hand.
- `env.render()` prints a compact view of state.

Observation fields:
- `hand`: sorted list of agent-held card ids
- `current_trick`: list of cards played so far in the current trick (tuples `(player_id, card_id)`)
- `trick_leader`: player id who leads the current trick
- `tricks_won`: dict `{0: teamA_tricks, 1: teamB_tricks}`
- `remaining_tricks`: integer

Cards are encoded as integers 0..51. Rank = id % 13 (2..A), Suit = id // 13 (Clubs, Diamonds, Hearts, Spades). Higher rank beats lower; suit is ignored in this base game (no trump, no following suit requirement).

Reward: dummy function returns +1 for team trick win, -1 for opponent, 0 otherwise. Customize as needed.

## Quick start

```python
from card_rl.env import CardGameEnv, decode_card

env = CardGameEnv(seed=42)
obs = env.reset()

done = False
while not done:
    # Pick a random legal action from your hand
    action = obs['hand'][0]  # simple policy: play the lowest id
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

print('Final score:', obs['tricks_won'])
```

## Run a quick demo

Use the included runner to simulate a full game with a random agent policy:

```bash
python run_random.py
```

