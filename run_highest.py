from card_rl import CardGameEnv, rank_of

def main():
    env = CardGameEnv()
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        hand = obs['hand']
        # Always play the highest-rank card (suit ignored)
        action = max(hand, key=rank_of)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    # env.render()
    print("Game finished. Total reward:", total_reward)
    print("Final tricks:", obs['tricks_won'])
    if 'tens_won' in obs:
        print("10s won:", obs['tens_won'])


if __name__ == "__main__":
    main()
