---
layout: post
title: A primer on policy gradients&#58; REINFORCE
---

In this post, we will present a basic form of the policy gradient called the REINFORCE algorithm and show how to derive its policy gradient estimator update with a discrete action space. We will provide an implementation in Tensorflow and evaluate the results on a number of classic control benchmarks on [OpenAI gym](https://gym.openai.com/).

All implementation code can also be accessed on [github](http://github.com/evancasey/demeter).

### Introduction to policy gradients

There are already a number of great resources on policy gradients. In particular, the following resources do a great job of explaining policy gradients and should act as a supplementary reference to this blog post if needed:

1. [Reinforcement Learning: An Introduction](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf) also known as "the holy bible of reinforcement learning"
2. [John Schulman's RL lectures](https://www.youtube.com/watch?v=rO7Dx8pSJQw) at MLSS Cadiz
3. [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) by Andrej Karpathy

### Reinforcement learning with a stochastic policy

As described in my previous post on the cross-entropy method, the reinforcement learning problem setup consists of a series of timesteps, each containing a state that we act on. That is, at time $$t = 0$$ we have some initial state $$s_0$$ and observation $$y_0$$. We pick an action $$a_0$$ according to a stochastic policy $$\pi_\theta(a_0 \mid s_0)$$. More specifically, $$\pi_\theta(a_0 \mid s_0)$$ is a probability distribution conditioned on $$s_t$$ and parameterized by $$\theta$$.

At time $$t>0$$ until the end of the episode, we get a new state $$s_t$$, observation $$y_t$$, and reward from the previous time step $$r_{t-1}$$. After some number of episodes (eg. a rollout), we update our policy $$\pi_\theta(a_t \mid s_t)$$ based on the cumulative reward generated from those episodes.

### Defining the objective function

Our goal is to maximize the expected reward of all possible *trajectories*. A trajectory $$\tau$$ is just a sequence of state-action pairs up to some predetermined time-limit $$t$$ or environment specific termination. The probability of a trajectory $$p_{\pi}(\tau)$$ is parameterized under the stochastic policy $$\pi_\theta$$. We write the expected reward as:

$$
E(R(\tau)) = \sum_{T}p_{\pi}(\tau) R(\tau)
$$

where $$\tau = \{s_0,a_0,s_1,a_1,...,s_t,a_t\}$$, $$R(\tau)$$ is the cumulative reward generated from those state-action pairs, and $$T$$ is the space of all possible trajectories. This is the same as the objective function of cross-entropy, except we're using $$\tau$$ to denote the trajectory instead of $$x$$ and we're looking at the likelihood of $$\tau$$ instead of the log-likelihood.

### Model architecture

Now that we've defined the objective function we'd like to maximize, how do we use this to create an algorithm? Well, it's actually pretty simple! Recall that our policy $$\pi_\theta$$ is what determines the probability of each trajectory $$p_{\pi}(\tau)$$. If we take the gradient of the objective function with respect to $$\theta$$, than we have a way of updating the gradients in the direction that maximizes expected reward.

A general outline of the policy gradient algorithm follows a such:

> Initialize $$\theta_0 \in \mathbb{R}^d$$ <br />
> For iteration $$i \in \{0,\ldots,k\}$$:
>
> > Obtain a policy gradient estimator $$\hat{g}$$ from rollouts <br />
> > Apply the gradient update to $$\theta$$ 

where $$\hat{g}$$ is an estimator of $$\Delta_{\theta} E(R(\tau))$$.

### The Policy Gradient Theorem

In this section we'll derive the policy gradient estimator $$\hat{g}$$ which we'll use to update our policy weights $$\theta$$. Using the objective function we defined above, we first bring the derivative into the expected value:

$$
\begin{aligned}
    \Delta_{\theta}E(R(\tau)) &= \sum_{T}\Delta_{\theta}p_{\pi}(\tau) R(\tau)\\
    &= \sum_{T}p_{\pi}(\tau)\dfrac{\Delta_{\theta}p_{\pi}(\tau)} {p_{\pi}(\tau)}R(\tau)\\
\end{aligned}
$$

Using the fact that $$\Delta \log(f(x)) = \dfrac{\Delta f(x)}{f(x)}$$, we have:

$$
\begin{aligned}
    \Delta_{\theta}E(R(\tau)) &= \sum_{T}p_{\pi}(\tau)\Delta_{\theta}\log(p_{\pi}(\tau)) R(\tau)\\
    &= E\big[R(\tau) \Delta_{\theta} \log(p_\pi(\tau))\big]\\
\end{aligned}
$$

Horray! Now we have a policy gradient estimator written in terms of a trajectory of state-action pairs. You may be thinking: wait a second, this looks way different from the policy gradient estimator formula in the Sutton textbook!?  However, this equation is expressing almost exactly the same thing as Sutton's version. 

To show this, we can simply notice that the expected value of $$R(\tau)$$ is the summation across initial states weighted by the probability of that state, and rewrite the above expression as:


$$
\begin{aligned}
    \Delta_{\theta}E(R(\tau)) &= \sum_s p_{\pi}(s) v_\pi(s) \Delta_{\theta} \log(p_\pi(\tau)) \\ 
    &= \sum_s p_{\pi}(s) \big[v_{\pi}(s)\big] \big[\Delta_{\theta} \log(\prod_a{\pi_{\theta}(a \mid s)}))\big] \\
    &= \sum_s p_{\pi}(s) \big[\sum_a q_{\pi}(s,a) \pi_{\theta}(a \mid s)\big] \big[\sum_a \Delta_{\theta} \log(\pi_{\theta}(a \mid s))) \big]
\end{aligned}
$$

Combining the two right most terms and again using the fact that $$\Delta \log(f(x)) = \dfrac{\Delta f(x)}{f(x)}$$, we get:

$$
\begin{aligned}
    \Delta_{\theta}E(R(\tau)) &= \sum_s p_{\pi}(s) \sum_a q_{\pi}(s,a) \Delta_{\theta} \pi_{\theta}(a \mid s)) \\
\end{aligned}
$$

...and now we have Sutton's version of the policy gradient theorem!

I think the first terminology (which I am borrowing from Schulman's lectures) is easier to understand. The second form of this equation is more often used in literature, so it's useful to understand both versions.


### Likelihood Ratio Methods and REINFORCE

As we stated earlier, the two definitions of the policy gradient estimator shown above are *almost* expressing the same thing...but not quite! The key difference between the two is that in Sutton's version we have to compute the term $$\sum_a q_{\pi}(s,a) \pi_{\theta}(a \mid s)$$ whereas in the first version we don't specify how we are going to compute $$E(R(\tau))$$. Computing $$\sum_a q_{\pi}(s,a) \pi_{\theta}(a \mid s)$$ directly would be intractable or at least very computationally expensive over a large action space.

This brings us to the main idea of REINFORCE: let's express the policy gradient updates in terms of samples whose *expectation approximates the theoretical gradient of the objective function*. Instead of computing $$\sum_a q_{\pi}(s,a) \pi_{\theta}(a \mid s)$$ directly, we rely on the fact that we can sample from the state-action distribution to estimate $$\sum_a q_\pi(s, a)\pi_{\theta}(a \mid s)$$. In REINFORCE, this is done on-policy, meaning the space of state-action pairs that we sample from comes directly from our policy $$\pi_\theta$$.

To derive the gradient update under REINFORCE, let's write the original theoretical gradient update again as an expected value over a trajectory $$\tau$$ with $$T$$ timesteps:

$$
\begin{aligned}
    \hat{g} = \Delta_{\theta}E(R(\tau)) &= E\big[R(\tau) \Delta_{\theta} \log(p_\pi(\tau))\big] \\
    &= E\Big[ \big[\sum_{t=0}^{T-1}r_t \big] \big[\sum_{t=0}^{T-1} \Delta_{\theta}\log(\pi_\theta(a_{t} \mid s_{t}))\big]\Big]
\end{aligned}
$$

Notice that the expected value of one reward term at timestep $$t$$ is:

$$
\begin{aligned}
    \Delta_{\theta}E(r_t) = E\Big[ r_t \sum_{t'=0}^{t-1} \Delta_{\theta}\log(\pi_\theta(a_{t'} \mid s_{t'}))\Big] 
\end{aligned}
$$

It follows that the expected value of the reward over a trajectory of $$T$$ timesteps is:

$$
\begin{aligned}
    \Delta_{\theta}E(R(\tau)) &= E\Big[ \sum_{t=0}^{T-1}r_t \sum_{t'=0}^{t-1} \Delta_{\theta}\log(\pi_\theta(a_{t'} \mid s_{t'}))\Big] \\
    &= E\Big[ \sum_{t'=0}^{t-1} \Delta_{\theta}\log(\pi_\theta(a_{t'} \mid s_{t'})) \sum_{t=t'}^{T-1}r_t \Big]
\end{aligned}
$$

This last equation brings us to the final policy gradient estimator equation, except we also include a discount factor $$\gamma$$ to reward actions more based on how soon they yielded a reward (this avoids conflating multiple actions together with a future reward). For a sampled trajectory consisting of state-action pairs with their corresponding rewards under the current policy $$\pi_\theta$$, we update our policy weights $$\theta$$ for each $$s_{t'}, a_{t'}, r_{t'}$$ in the trajectory as follows:

$$
  \theta \leftarrow \gamma^{t'} \big[\sum_{t=t'}^{T}r_t \big]\big[\Delta_{\theta} \log(\pi_{\theta}(a_{t'} \mid s_{t'}))\big]
$$

The REINFORCE algorithm pseudocode follows below:

> Initialize $$\theta_0 \in \mathbb{R}^d$$ <br />
> For iteration $$i \in \{0,\ldots,k\}$$: <br />
>
> > Generate n trajectories $$b$$ <br />
> > For each timestep $$t'$$ in $$b$$: <br />
> > $$ \theta \leftarrow \gamma^{t'} \big[\sum_{t=t'}^{T}r_t \big]\big[\Delta_{\theta} \log(\pi_{\theta}(a_{t'} \mid s_{t'}))\big] $$
>

### Implementation

The first component of the implementation is the policy $$\pi_\theta$$ which is parameterized with a neural network of arbitrary size and layers. The policy has two methods: `sample_action` which we use for policy evaluations during rollouts and `train` which feeds in the states, actions, and discounted returns into the Tensorflow graph and actuates the gradient update for the batch of examples provided.

```python
import tensorflow as tf
import numpy as np

from networks.nn import FullyConnectedNN

class DiscreteStochasticMLPPolicy(object):

    def __init__(self, network_name,
                       sess,
                       optimizer,
                       hidden_layers,
                       num_inputs,
                       num_actions,
                       annealer = None):

        # tf
        self.sess = sess
        self.optimizer = optimizer

        # env
        self.num_actions = num_actions

        # anneal
        self.annealer = annealer

        # placeholders
        self.actions = tf.placeholder(tf.int32, [None, 1], "actions")
        self.targets = tf.placeholder(tf.float32, [None, 1], "targets")

        # network
        self.network = FullyConnectedNN(
            name = network_name,
            sess = sess, 
            optimizer = optimizer,
            hidden_layers = "",
            num_inputs = num_inputs,
            num_outputs = num_actions)

        # construct pi_theta(a|s)
        self.actions_log_prob = tf.squeeze(tf.nn.log_softmax(self.network.logits))
        self.actions_mask = tf.squeeze(tf.one_hot(indices = self.actions, depth = num_actions))
        self.picked_actions_log_prob = tf.reduce_sum(self.actions_log_prob * self.actions_mask, reduction_indices = 1)

        # construct policy loss and train op
        self.standardized = tf.squeeze(self.targets)
        self.loss = -tf.reduce_sum(self.picked_actions_log_prob * self.standardized)
        self.train_op = self.optimizer.minimize(self.loss)

        # summary
        self.summary_op = tf.scalar_summary("policy_loss", self.loss)

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    def sample_action(self, observation):
        if self.annealer is None or not self.annealer.is_explore_step():
            action_probs = self.network.sess.run(
                self.network.logits, 
                {self.network.observations: [observation]}
            )[0]
            action = np.random.choice(np.arange(len(action_probs)), p = self.softmax(action_probs))

            return action
        else:
            # take random action
            return np.random.randint(0, self.num_actions)

    def train(self, states, actions, targets, summary_op):
        _, summary_str = self.sess.run([
            self.train_op, 
            summary_op
        ], {
            self.network.observations: states,
            self.targets: targets,
            self.actions: actions
        })

        return summary_str
```

Notice that we've purposefully implemented the tensor operations in numpy rather than Tensorflow in `sample_action`. Since this function gets called on every step of each trajectory, we need it to run as fast as possible. Switching from Tensorflow to numpy ops here gives us a huge speed up (around an order of magnitude). This seems to be just a limitation of the current state of Tensorflow CPU support and will likely be improved in the future.

We also define a sampler to handle performing policy rollouts across an arbitrary batch of episodes. This takes an environment `env` (which is Gym in this case), a policy, a reward function `norm_reward`, and a discount factor `discount`. The reward function allows us define custom behavior based on the environment state -- it's a bit analagous to "feature engineering" in supervised learning. We'll discuss in the results section what reward functions we used to get the best results.

```python
from lib.batch_stats import BatchStats
from lib.batch_trajectory import BatchTrajectory

class BatchSampler(object):

    def __init__(self, env,
                       policy,
                       norm_reward,
                       discount):

        self.env = env
        self.policy = policy
        self.norm_reward = norm_reward
        self.discount = discount

    def sample(self, i_batch, batch_size, max_steps, verbose = True):
        traj = BatchTrajectory()
        stats = BatchStats(batch_size, max_steps)

        for i_eps in xrange(batch_size):
            state = self.env.reset()

            for t in xrange(max_steps):
                action = self.policy.sample_action(state)

                next_state, reward, is_terminal, info = self.env.step(action)

                traj.store_step(state.tolist(), action, self.norm_reward(is_terminal))
                stats.store_reward(reward)

                state = next_state

                if is_terminal: break

            # discounts the rewards over a single episode
            eps_rewards = traj.rewards[-t - 1:]
            traj.calc_and_store_discounted_returns(eps_rewards, self.discount)

            stats.summarize_eps(i_batch, i_eps, verbose)

        return stats, traj
```

For the rest of the implementation, take a look at the full repo [here](http://github.com/evancasey/demeter).

### Results

So how does the REINFORCE policy gradient do on some classic control tasks? The answer is: it does alright, but definitely not as well as the cross-entropy method. For simple tasks, policy gradients actually take longer to learn a good policy because they require a large number of samples compared to derivative free methods. However, they're capable of learning much more complex tasks than derivative free methods as we'll see in later posts.

Our REINFORCE agent is trained with a discount factor of .99, using Tensorflow's `AdamOptimizer` with a learning rate of .01. For cartpole, we customize the reward function to set the reward to -10 if the pole falls below 15 degrees from standing and 0.1 otherwise. For acrobot, we set the reward to 5 if the lower link swings above the required height and -0.1 otherwise. The total rewards by episode are plotted below, averaged over 5 runs:

![img]({{site.url}}/blog/img/pg/cartpole_reinforce.png)
![img]({{site.url}}/blog/img/pg/acrobot_reinforce.png)

The graph on the left shows the total episode reward over time for the cartpole environment, and the one on the right shows the same for acrobot. Note that the rewards shown are here the default Gym rewards, not the ones we hand engineered. For cartpole this means reward = 1 for every step the pole is not tipped over and for acrobot reward = -1 for every step the lower link is not above the required height.

### Tips and tricks

In order to get REINFORCE, or any policy gradient algorithm working properly, you'll want to keep a few things in mind:

1. Standardize your rewards (across each episode)!. If you don't, the magnitude of your gradient updates will be all over the place.

2. Policy loss doesn't decrease over time like it does for supervised learning. This is because we're exploring new parts of the sample space as we update our policy. Here's an example of the REINFORCE policy loss over time, visualized via Tensorboard: ![img]({{site.url}}/blog/img/pg/policy_loss.png)

3. Since policy loss doesn't decrease and rewards often have high variance, it's often difficult to figure out if your algorithm is actually improving during a run. I recommend using a metric like average reward over the last 100 episodes.

4. Tensorflow's `AdamOptimizer` seems to work well here, but that's not say other optimizers shouldn't.

### Conclusion

The main idea of policy gradient methods is that they learn a parameterized policy $$\pi_\theta(a \mid s)$$ that can select actions without consulting a value function during rollouts. This differs from methods like Q-learning where we directly use the value function to choose actions at each timestep.

In this post we presented and derived the REINFORCE algorithm and showed results on several simple environments with discrete action spaces. In the next post, we'll discuss a slightly better family of policy gradients called actor-critic methods.
