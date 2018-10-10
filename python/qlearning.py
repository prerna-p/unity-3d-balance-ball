import numpy as np
import tensorflow as tf

from unityagents import UnityEnvironment, BrainInfo

train_mode = True
env = UnityEnvironment(file_name="3DBall", worker_id=0, seed=1)

#print(str(env))

default_brain = env.brain_names[0]
num_states = env.brains[default_brain].vector_observation_space_size
brain = env.brains[default_brain]
# Reset the environment
env_info = env.reset(train_mode=train_mode)[default_brain]

# Examine the state space for the default brain
print("Agent state looks like:")
print(env_info.vector_observations)
print("State space size", num_states)
print ("reward", env_info.rewards)
print ("agent", env_info.agents)
print ("action", np.random.randn(len(env_info.agents), brain.vector_action_space_size))

# Set learning parameters
discount = .9
alpha = .95
num_episodes = 200
memory = 10000
batch = 32
episode_steps = 2000
actor_rate = 0.001
critic_rate = 0.002
mem_point = 0
session = tf.Session()


########################
#   Tensorboard
########################
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    graph_var = [episode_reward]
    summarize = tf.summary.merge_all()
    return summarize, graph_var

##################################
# params for the neural networks
##################################
state_size = 24  # stacked vector size
action_size = 2  # action vector size
action_bounds = [-3, 3]  #action bounds
memory_arr = np.zeros((memory,state_size + action_size + 1), dtype=np.float32)


#################
# actor network
#################
def create_actor():
    with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
        net = tf.layers.dense(state_size, 30, activation=tf.nn.relu, name='layer1', trainable=True)
        act = tf.layers.dense(net, action_size, activation=tf.tanh, name='act', trainable=True)
        return tf.multiply(act, action_bounds, name='newact')



#################
# critic network
################
def create_critic(state, action):
    with tf.variable_scope('Critic', reuse=None, custom_getter=None):
        net = 30
        new_s_weight = tf.get_variable('s-weight', [state_size, net], trainable=True)
        actor_weight = tf.get_variable('a-weight', [action_size, net], trainable=True)
        val = tf.get_variable('value', [1, net], trainable=True)
        net = tf.nn.relu(tf.matmul(s, new_s_weight) + tf.matmul(action, actor_weight) + val)
        return tf.layers.dense(net, 1, trainable=True)

global_a_net = None

def get_action(state):
    return session.run(global_a_net, {state: s[np.newaxis, :]})[0]

def train():
    s = tf.placeholder(tf.float32, [None, state_size], 's')
    new_s = tf.placeholder(tf.float32, [None, state_size], 'new_s')
    r = tf.placeholder(tf.float32, [None, 1], 'r')

    a_net = create_actor(new_s, reuse=True, custom_getter=None)
    c_net = create_critic(new_s, a_net, reuse=True, custom_getter=None)
    global_a_net = a_net
    session.run(a_net, {s: memory[:,:state_size]})

def sars(state, action, reward, next_state):
    transition = np.hstack((state, a, [reward], next_state))
    memory_arr[mem_point%memory, :] = transition
    mem_point = mem_point + 1



# Initializing summary
summarize, graph_var = build_summaries()
session.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./results/graph', session.graph)

####################
# learning
####################
var = 3
for ep in range(num_episodes):

    env_info = env.reset(train_mode=train_mode)[default_brain]
    ep_reward = 0
    current = env_info.vector_observations
    actions = [[0] * 2] * 12 #12 agents, each having action vector of size 2

    for step in range(episode_steps):
        for iteration in range(12):
            a = get_action(current[iteration])
            a = np.clip(np.random.normal(a, var), 0, 1)
            actions[iteration] = a.tolist()
        # print("#")
        # print(actions)

        new_env_info = env.step(actions)[default_brain]
        r0 = new_env_info.rewards[0]
        new_state = new_env_info.vector_observations
        sars(current[0], actions[0], r0 / 10, new_state[0])

        var *= .9995
        train()

        current = new_state
        ep_reward += r0

        summary_str = session.run(summarize, feed_dict={
            graph_var[0]: ep_reward,
        })
        writer.add_summary(summary_str, ep)
        writer.flush()

        if step < episode_steps:
            print('Episode:', ep, ' Reward: %i' % int(ep_reward))
            break

env.close()


###########################################################
# we tried to implement q learning, it did not work out!!
###########################################################

#𝑄(𝑠,𝑎)=(1−𝛼)𝑄(𝑠,𝑎)+ 𝛼[𝑟+ 𝛾max𝑎′𝑄(𝑠′,𝑎′)]

'''
qtable = {}
def action_lookup(arr):
    if arr[0,0]<0:
        a="-"
    elif arr[0,0]==0:
        a="0"
    else:
        a="+"

    if arr[0,1]>0:
        b="+"
    elif arr[0,1]==0:
        b="0"
    else:
        b="-"
    return action_table[(a,b)]


def max_over_actions(z_rot,x_rot):
    global qtable
    l=[]
    for i in range(1,9):
        if (z_rot,x_rot,i) in qtable:
            l.append(qtable[(z_rot,x_rot,i)])

    if len(l)==0:
        return 0
    else:
        return max(l)


def max_over_vec(z,x):
    global qtable
    max=-9999
    max_action=0
    for i in range(1,9):
        if(z,x,i) in qtable:
            if qtable[(z,x,i)]>max:
                max=qtable[(z,x,i)]
                max_action=i

    return max_action


for episode in range(num_episodes):
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    episode_rewards = 0
    while not done:
        print(env_info.vector_observations)
        action = np.random.randn(len(env_info.agents), brain.vector_action_space_size)
        new_state = env.step(action)[default_brain]
        action_val = action_lookup(action)
        z_rot=new_state.vector_observations[0, 0]
        x_rot=new_state.vector_observations[0,1]
        if (z_rot,x_rot,action_val) in qtable:
            qtable[(z_rot,x_rot,action_val)]= ((1- alpha) * qtable[(z_rot,x_rot,action_val)])+ \
                                              (alpha * (new_state.rewards[0] + discount * max_over_actions(z_rot,x_rot)))
        else:
            qtable[(z_rot, x_rot, action_val)]= new_state.rewards[0]

        done = new_state.local_done[0]
    print(episode)
    #print("Total reward this episode: {}".format(episode_rewards))


for episode in range(num_episodes):
    env_info = env.reset(train_mode=train_mode)[default_brain]

    action = max_over_vec(env_info.vector_observations[0,0], env_info.vector_observations[0,1])

    new_state = env.step(action)[default_brain]

    action_val = action_lookup(action)
    z_rot = new_state.vector_observations[0, 0]
    x_rot = new_state.vector_observations[0, 1]
    if (z_rot, x_rot, action_val) in qtable:
        qtable[(z_rot, x_rot, action_val)] = ((1 - alpha) * qtable[(z_rot, x_rot, action_val)]) + (
        alpha * (new_state.rewards[0] + discount * max_over_actions(z_rot, x_rot)))
    else:
        qtable[(z_rot, x_rot, action_val)] = new_state.rewards[0]

    done = new_state.local_done[0]
    print(episode)
    
    
# action_table={}
#
# action_table[("-","+")]=1
# action_table[("+","-")]=2
# action_table[("-","-")]=3
# action_table[("+","+")]=4
# action_table[("0","+")]=5
# action_table[("+","0")]=6
# action_table[("-","0")]=7
# action_table[("0","-")]=8
#
# reverse_action_table={}
# reverse_action_table[1]=np.array([])
'''