from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random

# Card encoding: 0..51
# suit = id // 13  (0=Clubs,1=Diamonds,2=Hearts,3=Spades)
# rank = id % 13   (0=2, 1=3, ..., 8=10, 9=J, 10=Q, 11=K, 12=A)

RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUIT_NAMES = ['C', 'D', 'H', 'S']

def suit_of(card_id: int) -> int:
    return card_id // 13


def rank_of(card_id: int) -> int:
    return card_id % 13


def card_str(card_id: int) -> str:
    return f"{RANK_NAMES[rank_of(card_id)]}{SUIT_NAMES[suit_of(card_id)]}"


def encode_card(rank: int, suit: int) -> int:
    return suit * 13 + rank


def decode_card(card_id: int) -> Tuple[int, int]:
    return rank_of(card_id), suit_of(card_id)


@dataclass
class TrickState:
    leader: int
    plays: List[Tuple[int, int]]  # (player_id, card_id)


class CardGameEnv:
    """
    Simple 4-player trick-taking environment with teams (0,2) vs (1,3) and random opponents.
    No trump; followers must follow suit when possible; highest rank wins the trick.

    Agent is player 0; their action is the specific card id to play from current hand.
    Each step simulates a full trick: agent plays, then players 1-3 play random cards.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.hands: List[List[int]] = [[] for _ in range(4)]
        self.tricks_won: Dict[int, int] = {0: 0, 1: 0}  # team 0 for (0,2), team 1 for (1,3)
        self.tens_won: Dict[int, int] = {0: 0, 1: 0}    # count of 10s captured by each team
        self.current_trick: Optional[TrickState] = None
        self.current_player: int = 0  # informational
        self.done: bool = False
        self._dealt: bool = False
        self._rounds_played: int = 0

    # --- Core API ---
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.hands = [sorted(deck[i * 13:(i + 1) * 13]) for i in range(4)]
        self.tricks_won = {0: 0, 1: 0}
        self.tens_won = {0: 0, 1: 0}
        self.current_player = 0  # agent leads every trick in this simplified env
        self.current_trick = TrickState(leader=0, plays=[])
        self.done = False
        self._dealt = True
        self._rounds_played = 0
        return self._obs()

    def step(self, action: int):
        """
        action: card_id that must be in agent's current hand.
        Returns: obs, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        # In this simplified env, the agent always starts a new trick.
        if action not in self.hands[0]:
            raise ValueError("Invalid action: card not in agent's hand.")

        # Start a fresh trick if needed
        if self.current_trick is None or len(self.current_trick.plays) == 4:
            self.current_trick = TrickState(leader=0, plays=[])
            self.current_player = 0

        # Agent plays
        self._play_card(0, action)

        # Other three players (1,2,3) play a card; if possible, they follow the leading suit
        lead_suit = suit_of(self.current_trick.plays[0][1])
        for pid in [1, 2, 3]:
            if not self.hands[pid]:
                continue
            same_suit_cards = [c for c in self.hands[pid] if suit_of(c) == lead_suit]
            card = self.rng.choice(same_suit_cards) if same_suit_cards else self.rng.choice(self.hands[pid])
            self._play_card(pid, card)

        reward = 0.0
        if self._is_trick_complete():
            reward = self._end_trick_and_score()

        obs = self._obs()
        terminated = self.done
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        print("Hands sizes:", [len(h) for h in self.hands])
        print("Tricks won: TeamA(0,2)=", self.tricks_won[0], " TeamB(1,3)=", self.tricks_won[1])
        print("10s won:    TeamA(0,2)=", self.tens_won[0], " TeamB(1,3)=", self.tens_won[1])
        if self.current_trick:
            plays_str = [f"P{pid}:{card_str(cid)}" for pid, cid in self.current_trick.plays]
            print(f"Current trick (leader P{self.current_trick.leader}):", ", ".join(plays_str))
        print("Agent hand:", " ".join(card_str(c) for c in self.hands[0]))

    # --- Helpers ---
    def _obs(self) -> Dict:
        remaining_tricks = len(self.hands[0])  # each trick consumes one card from agent
        return {
            'hand': sorted(self.hands[0]),
            'current_trick': list(self.current_trick.plays) if self.current_trick else [],
            'trick_leader': self.current_trick.leader if self.current_trick else None,
            'tricks_won': dict(self.tricks_won),
            'tens_won': dict(self.tens_won),
            'remaining_tricks': remaining_tricks,
            'current_player': self.current_player,
        }

    def _play_card(self, pid: int, card: int):
        # Remove from hand
        self.hands[pid].remove(card)
        # Add to current trick
        assert self.current_trick is not None
        self.current_trick.plays.append((pid, card))

        # Print the current trick after each card is played
        self._print_current_trick()

        # Enforce suit-following rules for non-leading plays.
        # If the player has a card in the lead suit but played off-suit, automatically replace
        # the play with a card from the lead suit. If they don't have the lead suit, keep the play.
        if len(self.current_trick.plays) > 1:
            lead_suit = suit_of(self.current_trick.plays[0][1])
            if suit_of(card) != lead_suit:
                led_suit_cards = [c for c in self.hands[pid] if suit_of(c) == lead_suit]
                if led_suit_cards:
                    # Revert off-suit play and replace with a lead-suit card (pick lowest rank deterministically)
                    self.current_trick.plays.pop()
                    self.hands[pid].append(card)
                    follow_card = min(led_suit_cards, key=rank_of)
                    self.hands[pid].remove(follow_card)
                    self.current_trick.plays.append((pid, follow_card))
                    # Reprint updated trick
                    self._print_current_trick()

        # Advance turn
        self.current_player = (self.current_player + 1) % 4

    def _print_current_trick(self):
        """Print the current trick in the order of cards played."""
        if not self.current_trick or not self.current_trick.plays:
            return
        trick_cards = [card_str(cid) for _, cid in self.current_trick.plays]
        print("Current trick cards:", " ".join(trick_cards))

    def _is_trick_complete(self) -> bool:
        return self.current_trick is not None and len(self.current_trick.plays) == 4

    def _end_trick_and_score(self) -> float:
        assert self.current_trick is not None
        # Highest card by rank wins; ignore suit
        winner_pid, _ = max(self.current_trick.plays, key=lambda pc: rank_of(pc[1]))
        winner_team = 0 if winner_pid in (0, 2) else 1
        self.tricks_won[winner_team] += 1

        # If the trick contains any 10s, award them to the winning team (rank_of == 8 corresponds to 10)
        tens_in_trick = sum(1 for _, cid in self.current_trick.plays if rank_of(cid) == 8)
        if tens_in_trick:
            self.tens_won[winner_team] += tens_in_trick

        # Reward via dummy function
        reward = self.compute_reward(winner_team)

        self._rounds_played += 1

        # Next trick: agent leads again in this simplified env
        self.current_player = 0
        self.current_trick = TrickState(leader=0, plays=[])

        # Episode ends when all cards played by agent
        if len(self.hands[0]) == 0:
            self.done = True
        return reward

    # --- Reward ---
    def compute_reward(self, winner_team: int) -> float:
        """Dummy reward function called at end of each trick.
        Default: +1 if agent team (0,2) wins; -1 otherwise.
        Override or monkey-patch for custom shaping.
        """
        return 1.0 if winner_team == 0 else -1.0
