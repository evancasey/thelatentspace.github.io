---
layout: post
title: A primer on policy gradients 
---



## Algorithm

Similar to cross-entropy, the objective function we'd like to maximize is the expected value of the reward function:

$$
E(R(\tau)) = \sum_{T}p_{\pi}(\tau) R(\tau)
$$

where $$\tau = \{s_0,a_0,s_1,a_1,...,s_t,a_t\}$$ is a trajectory of state-action pairs, $$R(\tau)$$ is the cumulative reward generated from those state-action pairs, and $$T$$ is the space of all possible trajectories. This is the same as the objective function of cross-entropy, except we're using $$\tau$$ to denote the trajectory instead of $$x$$ and we're looking at the likelihood of $$\tau$$ instead of the log-likelihood.

The general policy gradient algorithm is shown below:

> Initialize $$\theta_0 \in \mathbb{R}^d$$ <br />
> For iteration $$i \in \{0,\ldots,k\}$$:
>
> > Obtain a policy gradient estimator $$\hat{g}$$ from rollouts <br />
> > Apply the gradient update to $$\theta$$ 

where $$\hat{g}$$ is an estimator of $$\Delta_{\theta} E(R(\tau))$$.

## The Policy Gradient Theorem

We can derive the policy gradient estimator $$\hat{g}$$ as follows:

$$
\begin{aligned}
    \Delta_{\theta}E(R(\tau)) = \sum_{T}\Delta_{\theta}p_{\pi}(\tau) R(\tau)\\
    = \sum_{T}p_{\pi}(\tau)\dfrac{\Delta_{\theta}p_{\pi}(\tau)} {p_{\pi}(\tau)}R(\tau)\\
    = \sum_{T}p_{\pi}(\tau)\Delta_{\theta}\log(p_{\pi}(\tau)) R(\tau)\\
    = E\big[R(\tau) \Delta_{\theta} \log(p_\pi(\tau))\big]\\
\end{aligned}
$$

Using the fact that the expected value of $$R(\tau)$$ is the summation across initial states weighted by the probability of that state, we can rewrite the above expression as:


$$
\begin{aligned}
    \Delta_{\theta}E(R(\tau)) = \sum_s p_{\pi}(s) v_\pi(s) \Delta_{\theta} \log(p_\pi(\tau)) \\ 
    = \sum_s p_{\pi}(s) \big[v_{\pi}(s)\big] \big[\Delta_{\theta} \log(\prod_a{\pi_{\theta}(a \mid s)}))\big] \\
    = \sum_s p_{\pi}(s) \big[\sum_a q_{\pi}(s,a) \pi_{\theta}(a \mid s)\big] \big[\sum_a \Delta_{\theta} \log(\pi_{\theta}(a \mid s))) \big] \\
    = \sum_s p_{\pi}(s) \sum_a q_{\pi}(s,a) \Delta_{\theta} \pi_{\theta}(a \mid s)) \\
\end{aligned}
$$

You may notice this form of the policy gradient theorem from Sutton or Silver's lectures but it's exactly the same thing! In my opinion, the first form (6) of this equation (which is used in Schulman's DL School lectures) is much more intuitive to work with, but most of the policy gradient literature uses (7), so it's important to understand both.

## Likelihood Ratio Methods and REINFORCE

The main idea of REINFORCE is to express the policy gradient updates in terms of samples whose expectation approximates the theoretical gradient of the objective function. Recall that gradient estimator, $$\hat{g}$$, we got from the policy gradient theorem is expressed as the weighted summation over the entire state space times the summation over the entire action space of $$q_{\pi}(s,a)\Delta_{\theta}(a \mid s)$$. Clearly it's not practical to compute this directly, so we rely on the fact that we can sample from the state-action distribution to estimate $$q_\pi(s \mid a)\Delta_{\theta}(a \mid s)$$. In REINFORCE, this is done on-policy, meaning the space of state-action pairs that we sample from comes directly from our policy.

To derive the gradient update under REINFORCE, let's rewrite the original theoretical gradient update again as an expected value over a trajectory $$\tau$$ with $$t$$ timesteps:

$$
\begin{aligned}
    \hat{g} = \Delta_{\theta}E(R(\tau))  \\
    = E\big[R(\tau) \Delta_{\theta} \log(p_\pi(\tau))\big] \\
    = E\Big[ \sum_{t=0}^{T-1}r_t \sum_{t'=0}^{t-1} \Delta_{\theta}\log(\pi_\theta(a_{t'} \mid s_{t'}))\Big]
\end{aligned}
$$

In practice, we sample a collection of trajectories of state-action pairs with their corresponding rewards under the current policy $$\pi_\theta$$ and use a discount factor $$\gamma$$. We update our policy weights $$\theta$$ for each $$s_t, a_t, r_t$$

$$
  \theta \leftarrow \gamma^{t'} \big[\sum_{t=t'}^{T}r_t \big]\big[\Delta_{\theta} \log(\pi_{\theta}(a_{t'} \mid s_{t'}))\big]
$$

> Initialize $$\theta_0 \in \mathbb{R}^d$$ <br />
> For iteration $$i \in \{0,\ldots,k\}$$: <br />
>
> > Generate n trajectories $$b$$ <br />
> > For each timestep $$t'$$ in $$b$$: <br />
> > $$ \theta \leftarrow \gamma^{t'} \big[\sum_{t=t'}^{T}r_t \big]\big[\Delta_{\theta} \log(\pi_{\theta}(a_{t'} \mid s_{t'}))\big] $$


## REINFORCE with a Baseline

In vanilla REINFORCE, we are reinforecing the prob of all actions equally per grad update. estimator is unbiased but you'll have high variance

$$
  \hat{g} =  E\Big[ \big[\sum_tr_t - b(s_t)\big] \big[\sum_t \Delta_{\theta}\log(\pi_\theta(a_t \mid s_t)\big]\Big]
$$



## Actor Critic Methods

Although the REINFORCE-with-baseline method learns both a policy and a state- value function, we do not consider it to be an actor-critic method because its state- value function is used only as a baseline, not as a critic.

$$
  A = r_t + {\gamma}V(s_{t+1}) - V(s)
$$

We use the baseline to also critique: did the action put me in a more advantagous state at the next time step? Whereas in baseline we just use the value func to to give us a better est. of the action's impact on reward.

## Conclusion

The main idea of policy gradient methods is that they learn a parameterized policy $$\pi_\theta(a \mid s)$$ that can select actions without consulting a value function during rollouts. This differs from methods like Q-learning where we directly use the value function to choose actions at each timestep.

Next steps:
* Hyperbolic discounting
