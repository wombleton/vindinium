import tensorflow as tf
import numpy as np
import os
import json
import random
from collections import deque

class VinFlow():
    ACTIONS = ["North", "South", "East", "West", "Stay"]
    ACTIONS_COUNT = 5  # number of valid actions.
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    MEMORY_SIZE = 50000  # number of observations to remember
    MINI_BATCH_SIZE = 100  # size of mini batches
    STATE_FRAMES = 4 # number of frames to store in the state
    SCREEN_X, SCREEN_Y = (80, 80)
    SAVE_EVERY_X_STEPS = 10000
    LEARN_RATE = 1e-6
    STORE_SCORES_LEN = 200.

    def __init__(self, checkpoint_path="vindinium_network"):
        self._checkpoint_path = checkpoint_path
        self._session = tf.Session()
        self._input_layer, self._output_layer = VinFlow._create_network()

        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.mul(self._output_layer, self._action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self._target - readout_action))
        self._train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)

        self._observations = deque()
        self._last_scores = deque()

        self._time = 0

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[4] = 1.

        self._session.run(tf.initialize_all_variables())

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)

        self._saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)

        self._load_observations()

    @staticmethod
    def _create_network():
        # network weights
        convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, VinFlow.STATE_FRAMES, 32], stddev=0.01))
        convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

        convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

        convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
        convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))

        feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

        feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, VinFlow.ACTIONS_COUNT], stddev=0.01))
        feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[VinFlow.ACTIONS_COUNT]))

        input_layer = tf.placeholder("float", [None, VinFlow.SCREEN_Y + 4, VinFlow.SCREEN_X, VinFlow.STATE_FRAMES])

        hidden_convolutional_layer_1 = tf.nn.relu(tf.nn.conv2d(input_layer, convolution_weights_1, strides=[1, 4, 4, 1], padding="SAME") + convolution_bias_1)

        hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_2 = tf.nn.relu( tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2, strides=[1, 2, 2, 1], padding="SAME") + convolution_bias_2)

        hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_3 = tf.nn.relu( tf.nn.conv2d(hidden_max_pooling_layer_2, convolution_weights_3, strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_3)

        hidden_max_pooling_layer_3 = tf.nn.max_pool(hidden_convolutional_layer_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_3, [-1, 256])

        final_hidden_activations = tf.nn.relu( tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

        output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

        return input_layer, output_layer

    def _load_observations(self):
        while (1):
            while len(self._observations) < self.MEMORY_SIZE:
                self._read_game()

            self._train()
            self._time += 1

            print("Time: %s" % (self._time))

    def _train(self):
        # sample a mini_batch to train on
        mini_batch = random.sample(self._observations, self.MINI_BATCH_SIZE)
        # remove this many items from the start of _observations
        [self._observations.popleft() for _ in range(self.MINI_BATCH_SIZE)]

        # get the batch variables
        previous_states = [d[0] for d in mini_batch]
        actions = [d[1] for d in mini_batch]
        rewards = [d[2] for d in mini_batch]
        current_states = [d[3] for d in mini_batch]
        agents_expected_reward = []
        # this gives us the agents expected reward for each action we might

        agents_reward_per_action = self._session.run(self._output_layer, feed_dict={self._input_layer: current_states})
        print agents_reward_per_action
        for i in range(len(mini_batch)):
            if mini_batch[i][4]:
                # this was a terminal frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

        # learn that these actions in these states lead to this reward
        self._session.run(self._train_operation, feed_dict={
            self._input_layer: previous_states,
            self._action: actions,
            self._target: rewards})

        # save checkpoints for later
        if self._time % self.SAVE_EVERY_X_STEPS == 0:
            self._saver.save(self._session, self._checkpoint_path + '/network', global_step=self._time)

    def _parse_move_as(self, move, player):
        count = 20
        tokens = dict({
            '  ': 0. / count,
            '##': 1. / count,
            '$-': 2. / count,
            '[]': 3. / count,
            '@1': 4. / count,
            '@2': 5. / count,
            '@3': 6. / count,
            '@4': 7. / count,
            '$1': 8. / count,
            '$2': 9. / count,
            '$3': 10. / count,
            '$4': 11. / count
        })
        score = 0
        last_action = np.zeros([self.ACTIONS_COUNT])
        state = np.full((VinFlow.SCREEN_Y + 4, VinFlow.SCREEN_X), 1.)
        hero_index = 1
        for hero in move["heroes"]:
            if hero["id"] == player:
                last_dir = hero.get("lastDir", "Stay")
                last_action[self.ACTIONS.index(last_dir)] = 1

                state[0][0] = hero["life"] / 100.
                state[0][1] = hero["gold"] / 10000.
                score = state[0][2] = hero["mineCount"] / 100.
            else:
                state[hero_index][0] = hero["life"] / 100.
                state[hero_index][1] = hero["gold"] / 10000.
                state[hero_index][2] = hero["mineCount"] / 100.
                hero_index += 1

        size = move["board"]["size"]
        s = move[ "board" ][ "tiles" ]
        x = 0
        y = 4
        for (s1, s2) in zip(s[0::2], s[1::2]):
            tile = s1 + s2
            if tile == '@' + str(player):
                state[y][x] = 12. / count
            elif tile == '$' + str(player):
                state[y][x] = 13. / count
            else:
                state[y][x] = tokens[tile]
            x += 1
            if x == size:
                x = 0
                y += 1

        finished = move["finished"]

        return score, last_action, finished, state

    def _read_game(self):
        before = len(self._observations)
        file_name = random.choice(os.listdir("games"))
        with open("games/" + file_name, "r") as f:
            moves = json.load(f)

            for player in range(1, 5): # 1..4
                history = []
                for i, move in enumerate(moves):
                    reward, last_action, finished, frame = self._parse_move_as(move, player)
                    if i == 0:
                        history = np.stack(tuple(frame for _ in range(self.STATE_FRAMES)), axis=2)

                    frame_to_append = np.reshape(frame, (self.SCREEN_Y + 4, self.SCREEN_X, 1))
                    current_state = np.append(history[:, :, 1:], frame_to_append, axis=2)
                    self._observations.append((history, last_action, reward, current_state, finished))
                    history = current_state
            print("Loaded %s observations from games/%s for a total of %s" % (len(self._observations) - before, file_name, len(self._observations)))

VinFlow()
