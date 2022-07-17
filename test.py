import soccer_twos

env = soccer_twos.make(render=False)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space.shape)

team0_reward = 0
team1_reward = 0
while True:
    obs, reward, done, info = env.step(
        {
            0: env.action_space.sample(),
            1: env.action_space.sample(),
            2: env.action_space.sample(),
            3: env.action_space.sample(),
        }
    )
    for player_index,info_ in info.items():
        print(dict(player_index=player_index, position=info_['player_info']['position'],velocity=info_['player_info']['velocity']))
        break
    team0_reward += reward[0] + reward[1]
    team1_reward += reward[2] + reward[3]
    if done["__all__"]:
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()