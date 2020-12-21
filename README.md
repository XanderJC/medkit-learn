

![](figs/logo.png)

# The Medkit-learn(ing) Environment

Medkit is a synthetic data generation library for medical sequential decision problems.

Example usage:
```python
import medkit as mk

synthetic_dataset = mk.batch_generate(
						domain 		= 'ICU',
						environment = 'HMM',
						policy 		= 'RNN',
						size 		= 10_000,
						test_size   = 1_000, ...)

static_train, observations_train, actions_train = synthetic_dataset['training']
static_test,  observations_test,  actions_test  = synthetic_dataset['testing']
```

While medical machine learning is by necessity almost always entirely offline, we also provide an interface through which you can interact online with the environment should you find that useful.

```python
env = mk.live_simulate(
		domain 		= 'ICU',
		environment = 'HMM',
		policy 		= 'RNN', ...)

observation = env.reset()
observation, reward, info, done = env.step(action)
```