from gym_minigrid.wrappers import *
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, InputLayer

from log import Logger

from absl import flags
from absl import app

import os

FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("gamma", 0.95, "Gamma decay factor")
flags.DEFINE_float("exploration_min", 0.01, "Exploration rate")
flags.DEFINE_float("exploration_max", 1.0, "Exploration rate")
flags.DEFINE_float("exploration_decay", 0.995, "Exploration rate")

flags.DEFINE_integer("memory_size", 1000000, "Experience replay buffer size")
flags.DEFINE_integer("batch_size", 20, "Batch size for training")
flags.DEFINE_integer("episodes", 10, "Number of episodes to play")
flags.DEFINE_integer("model_update", 10,
                     "Number of episodes for each update of target model")

flags.DEFINE_string("environment", "MiniGrid-Empty-8x8-v0",
                    "Environment to play in")
flags.DEFINE_string("output_dir", None, "Where to write files produced")


class Agent():

    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.memory = deque(maxlen=FLAGS.memory_size)
        self.exploration_rate = FLAGS.exploration_max
        self.reward_logs = Logger()
        self.reward_update_logs = Logger()

        self.model = tf.keras.models.Sequential([
            InputLayer(input_shape=observation_space),
            Flatten(),
            Dense(128, input_shape=(observation_space,), activation="relu"),
            Dense(128, activation="relu"),
            Dense(self.action_space, activation="linear")
        ])

        self.model.compile(
            loss="mse", optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate))

        self.target_model = tf.keras.models.clone_model(self.model)

        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.reward_logs.update(reward)
        self.memory.append((state, action, reward, next_state, done))

    def q_values(self, state):
        return self.target_model.predict(np.expand_dims(state, axis=0))[0]

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)

        q_values = self.q_values(state)

        return np.argmax(q_values)

    def update_target(self):
        self.target_model = tf.keras.models.clone_model(self.model)

    def experience_replay(self):
        if len(self.memory) < FLAGS.batch_size:
            return

        batch = random.sample(self.memory, FLAGS.batch_size)

        states, actions, rewards, next_states, done = [
            np.array(x) for x in zip(*batch)]

        self.reward_update_logs.update(rewards)

        q_update = np.zeros(rewards.shape)

        q_update[done] = rewards[done]
        q_update[~done] = rewards[~done] + FLAGS.gamma * \
            np.amax(self.target_model.predict(next_states[~done]), axis=1)

        q_values = self.target_model.predict(states)
        q_values[np.arange(len(q_values)), actions] = q_update

        self.model.fit(states, q_values, verbose=0)

        self.exploration_rate *= FLAGS.exploration_decay
        self.exploration_rate = max(
            self.exploration_rate, FLAGS.exploration_min)


def run_episode(env, agent, q_values_s0, q_values_sT):
    state = env.reset()

    step = 0

    while True:
        step += 1

        action = agent.act(state)
        state_next, reward, done, info = env.step(action)

        if step == 1:
            q_values_s0.append(agent.q_values(state)[action])

        if reward > 0:
            q_values_sT.append(agent.q_values(state)[action])

        agent.remember(state, action, reward, state_next, done)

        state = state_next

        if done:
            break

    return step, reward


def results(scores, agent, q_values_s0, q_values_sT):
    if FLAGS.output_dir is None:
        return

    os.mkdir(FLAGS.output_dir)

    FLAGS.append_flags_into_file(os.path.join(FLAGS.output_dir, "flags.txt"))

    plt.plot(q_values_s0)
    plt.savefig(os.path.join(FLAGS.output_dir, "q_values_s0.png"))
    plt.figure()

    plt.plot(q_values_sT)
    plt.savefig(os.path.join(FLAGS.output_dir, "q_values_sT.png"))
    plt.figure()

    plt.plot(np.cumsum(scores) / np.arange(1, len(scores) + 1))
    plt.savefig(os.path.join(FLAGS.output_dir, "score_means.png"))
    plt.figure()

    plt.plot(scores)
    plt.savefig(os.path.join(FLAGS.output_dir, "scores.png"))
    plt.figure()

    plt.plot(agent.reward_logs.temp_means)
    plt.savefig(os.path.join(FLAGS.output_dir, "reward-means.png"))
    plt.figure()

    plt.plot(agent.reward_update_logs.batch_means)
    plt.savefig(os.path.join(FLAGS.output_dir, "update-means.png"))
    plt.figure()

    with open("results.txt", "w") as f:
        print("Distribution of all rewards recorded: {}".format(
            Counter(agent.reward_logs.rewards)), file=f)
        print("Distribution of rewards used for training the q net: {}".format(
            Counter(agent.reward_update_logs.rewards)), file=f)


def main(argv):
    del argv

    print(tf.test.is_gpu_available())

    env = gym.make(FLAGS.environment)
    env = OneHotPartialObsWrapper(env)

    agent = Agent(env.observation_space.shape, env.action_space.n)

    scores = []

    q_values_s0 = []
    q_values_sT = []

    for run in range(1, FLAGS.episodes + 1):
        steps, score = run_episode(env, agent, q_values_s0, q_values_sT)

        scores.append(score)

        agent.experience_replay()

        if run % FLAGS.model_update:
            agent.update_target()

        print("Run {}: exploration {:.2f}, memory size {}, steps {}, reward {:0.2f}, mean {:0.2f}".format(
            run, agent.exploration_rate, len(agent.memory), steps, score, np.mean(scores)))

    q_values_s0 = np.array(q_values_s0)
    q_values_sT = np.array(q_values_sT)

    results(scores, agent, q_values_s0, q_values_sT)


if __name__ == "__main__":
    app.run(main)
