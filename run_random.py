from card_rl import CardGameEnv
import random


def main():
    env = CardGameEnv(seed=123)
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        hand = obs['hand']
        action = random.choice(hand)  # random agent policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.render()
    print("Game finished. Total reward:", total_reward)
    print("Final tricks:", obs['tricks_won'])


if __name__ == "__main__":
    main()
