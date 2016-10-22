---
layout: post
title: Derivative free reinforcement learning with the cross entropy method
---


Modern reinforcement learning can be split up into 3 main approaches:

1. [Policy iteration methods](http://www.control.ece.ntua.gr/UndergraduateCourses/ProxTexnSAE/Bertsekas.pdf) which alternate between estimating the value function under the current policy and improving the policy.
2. [Policy gradient methods](http://is.tuebingen.mpg.de/fileadmin/user_upload/files/publications/Neural-Netw-2008-21-682_4867[0].pdf) which use an estimator of the gradient of the expected return (total reward) obtained from sample trajectories.
3. [Derivative free approaches](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.6579&rep=rep1&type=pdf) which treat the return as a black box function to be optimized in terms of the policy parameters.

In this post we'll walk through an implementation of the cross-entropy optimization method on the [OpenAI Cartpole environment](https://gym.openai.com/envs/CartPole-v0).

### Introduction to RL

The reinforcement learning setting consists of an agent and an environment. At every timestep, the agent chooses an action, and the environment returns a reward and transitions into the next state. We assume the standard, fully-observed setting as described in the excellent OpenAI's excellent [Deep Reinforcement Learning Tutorial](https://gym.openai.com/docs/rl).


## Problem Setup
To briefly summarize, at time $$t = 0$$ we have some initial state $$s_0$$ and observation $$y_0$$. We pick an action $a_0$ according to a policy $$\pi_\theta(a_0 \mid s_0)$$. That is, $$\pi_\theta(a_0 \mid s_0)$$ is a probability distribution conditioned on $$s_t$$ and parameterized by $$\theta$$.

At time $$t>0$$ until the end of the episode, we get a new state $$s_t$$, observation $$y_t$$, and reward from the previous time step $$r_{t-1}$$. After some number of episodes (eg. a rollout), we update our policy $$\pi_\theta(a_t \mid s_t)$$ based on the cumulative reward generated from those episodes.

### Environment

In the cartpole environment, we have a 4 dimensional observation vector $$ o_t \in \mathbb{R^4}$$ that contains the following features:
- cart position
- pole_angle
- cart_velocity
- angle_rate_of_change

## Algorithm
Let $$d$$ be the number of dimensions in the model, $$N$$ be the batch size of each rollout, and $$k$$ be the number of iterations of our algorithm.

> Initialize $$\mu \in \mathbb{R}^d, \sigma \in \mathbb{R}^d$$ <br />
> For iteration $$i \in \{0,\ldots,k\}$$: 
>
> > Collect $$N$$ samples of $$\theta_i \sim \mathcal{N}(\mu_i, \text{diag}(\sigma_i))$$ <br />
> > Perform a noisy evaluation $$f(\theta_i, \zeta_i)$$ on each one <br />
> > Select the top $$p\%$$ of samples, which we'll call the **elite set** <br />
> > Fit a Gaussian distribution, with diagonal covariance, to the **elite set**, obtaining a new $$\mu$$, $$\sigma$$. <br />



## Convergence

To analyze the convergence of CEM, we must first consider the Monte Carlo Expectation Maximization (MCEM) problem, which solves a Maximum Likelihood problem of the form:

$$
\max_\theta \log \int_z p(y,z \mid \theta)dz
$$

where $$\theta$$ is the parameters of the probabilistic model we are trying to find, $$y$$ is the observed variables and $$z$$ is the unobserved variables.

If $$z$$ were known, we could use a simple Maximum A Posterior (MAP) estimation with a prior on $$\theta$$ and be done. However, this is not the case, so we have to use a Monte Carlo estimate of $$z$$ based on the posterior $$p(z \mid y,\theta)$$. The algorithm works by iteratively estimating $$E(\log p(y,z \mid \theta_k) \mid Y)$$ using the sample values of $$z$$, maximizing it with respect to $$\theta_k$$, and then using this new $$\theta_{k+1}$$ in the following timestep. At each timestep $$k$$, for some sample size $$M$$ we have

$$
E(\log p(y,z \mid \theta_k)) = \dfrac{1}{M} \sum_{m=1}^M \log p(y, z_m \mid \theta_k)
$$

In each iteration, we're collecting $$m$$ samples, using these to form a distribution over our latent variables $$z$$, and then reweighting this distribution by maximizing the likelihood of those samples. This is very similar to CEM except that in CEM and other RL problems we're trying to maximize expected reward and we don't know what samples (actions) lead to a good reward. More precisely, MCEM is maximizing the expected value of the likelihood function $$l(\theta;X)$$, and in CEM we're maximizing the expected value of the reward function, which we write as:

$$
E(R) = R \log p(y,z \mid \theta_k)
$$

where $$R = \sum_{m=1}^M r_m$$ is the cumulative reward over $$M$$ samples and $$y$, $$z$$ are both sequences of state-action pairs. Thus, our Monte-Carlo estimation function becomes:

$$
\dfrac{1}{M} \sum_{m=1}^M \log p(y,z_m \mid \theta_k) r_m
$$

Theorem: CEM converges to a local maximum of the objective $$N(\theta) = E(f(\theta))$$
Theorem: CEM does not converge to a local maximum of the objective $$N(\theta) = E_{\zeta}(f(\theta,\zeta))$$


## Conclusion

CEM performs really well on low-dim problems!
Not covered in sutton although used widely as a benchmark in continuous control problems...

MM monotically increases expected reward! Resources: Andrew ng doc, link to proofs...

### Conclusion

Works embarassingly well on problems with a small number of parameters...
CEM and CMA are derivative-free algorithms, hence their sample complexity scales unfavorably with the number of parameters, and they performed poorly on the larger problems.



