from ttwr_steering import TtwrEnv

if __name__ == "__main__":
    # create the environment with render mode
    env = TtwrEnv(render_mode="human")
    env.reset()
    for i in range(1000):
        state, reward, terminate_episode, _, _ = env.step(env.action_space.sample())
        if terminate_episode:
            break
        env.render()
    env.close()