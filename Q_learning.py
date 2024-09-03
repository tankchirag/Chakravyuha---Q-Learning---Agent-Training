import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pygame

def epsilon_greedy_policy(Q, state, epsilon, action_space):
    """
    Select an action based on epsilon-greedy policy.

    Args:
        Q (np.array): Q-table.
        state (np.array): Current state of the agent.
        epsilon (float): Exploration rate.
        action_space (gym.Space): Action space of the environment.

    Returns:
        int: Selected action.
    """
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        return np.argmax(Q[state[0], state[1], :])

def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma):
    """
    Train a Q-learning agent on the given environment.

    Args:
        env (gym.Env): The environment to train on.
        no_episodes (int): Number of episodes to train for.
        epsilon (float): Initial exploration rate.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Exploration decay rate.
        alpha (float): Learning rate.
        gamma (float): Discount factor.

    Returns:
        np.array: Trained Q-table.
    """
    pygame.init()
    screen = pygame.display.set_mode((env.grid_size[1] * 100, env.grid_size[0] * 100))
    pygame.display.set_caption('Chakravyuh Environment')

    Q = np.zeros((env.grid_size[0], env.grid_size[1], env.action_space.n))
    all_episode_rewards = []

    for episode in range(no_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = epsilon_greedy_policy(Q, state, epsilon, env.action_space)
            new_state, reward, done, _ = env.step(action)

            Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (
                reward + gamma * np.max(Q[new_state[0], new_state[1], :]) - Q[state[0], state[1], action]
            )

            state = new_state
            episode_reward += reward

            env.render(screen)
            pygame.display.flip()

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        all_episode_rewards.append(episode_reward)

        #print(f"Episode {episode+1}, Reward: {episode_reward}")

    np.save("q_table.npy", Q)
    np.save("rewards.npy", all_episode_rewards)

    pygame.quit()
    return Q

def visualize_q_table(q_values_path):
    """
    Visualize the trained Q-table.

    Args:
        q_values_path (str): Path to the Q-table file.
    """
    Q = np.load(q_values_path)
    #print(Q)

    # Assuming Q.shape is (7, 7, 4), we need to select the 2x2 portion
    Q_subset = Q[:, :, :]  # Selecting the top-left 2x2 portion of Q-table

    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    actions = ['Up', 'Down', 'Left', 'Right']

    for i in range(4):
        row = i // 2
        col = i % 2
        cax = ax[row, col].matshow(Q_subset[:, :, i], cmap='viridis')

        for x in range(Q_subset.shape[0]):
            for y in range(Q_subset.shape[1]):
                ax[row, col].text(y, x, f'{Q_subset[x, y, i]:.2f}', va='center', ha='center', color='white')



        fig.colorbar(cax, ax=ax[row, col])
        ax[row, col].set_title(f'Q-value for action: {actions[i]}')

    plt.tight_layout()
    plt.show()

    rewards = np.load("rewards.npy")
    plt.plot(rewards)

