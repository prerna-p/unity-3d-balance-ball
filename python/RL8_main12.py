import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

from unityagents import UnityEnvironment, BrainInfo

DISCOUNT = 0.9
TAU = 0.01  # soft replacement #0.01
MEM = 10000
BATCH = 32
NUM_EPISODES = 20
EPISODE_STEPS = 2000
ACTOR_RATE = 0.001
CRITIC_RATE = 0.002
RENDER = False
session = tf.Session()

# ===========================
#   Tensorboard
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    #episode_ave_max_q = tf.Variable(0.)
    #tf.summary.scalar("Qmax Value", episode_ave_max_q)

    #summary_vars = [episode_reward, episode_ave_max_q]
    summary_vars = [episode_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.memory = np.zeros((MEM, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = session

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(ACTOR_RATE).minimize(a_loss, var_list=a_params) #stochastic optimization

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + DISCOUNT * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(CRITIC_RATE).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEM, size=BATCH)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEM  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

train_mode = True
env = UnityEnvironment(file_name="3DBall", worker_id=11, seed=1)

default_brain = env.brain_names[0]

brain = env.brains[default_brain]

s_dim = 24
a_dim = 2
a_bound = [-3, 3]  # action

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()


# Set up summary Ops
summary_ops, summary_vars = build_summaries()
session.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./results/tf_ddpg', session.graph)


for i in range(NUM_EPISODES):

    env_info = env.reset(train_mode=train_mode)[default_brain]

    ep_reward = 0
    s = env_info.vector_observations
    actions = [[0] * 2] * 12

    for j in range(EPISODE_STEPS):

        # Add exploration noise
        for iteration in range(12):
            a = ddpg.choose_action(s[iteration])
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            actions[iteration] = a.tolist()
        # print("#")
        #        print(actions)

        new_env_info = env.step(actions)[default_brain]

        r = new_env_info.rewards[0]

        s_ = new_env_info.vector_observations

        ddpg.store_transition(s[0], actions[0], r / 10, s_[0])
        #        for iteration in range(12):

        if ddpg.pointer > MEM:
            var *= .9995  # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r

        summary_str = session.run(summary_ops, feed_dict={
            summary_vars[0]: ep_reward,
            #summary_vars[1]: ep_ave_max_q / float(j)
        })

        writer.add_summary(summary_str, i)
        writer.flush()

        if j == EPISODE_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break

print('Running time: ', time.time() - t1)