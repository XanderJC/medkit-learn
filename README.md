
# Working title: Medkit-learn(ing) Environment

### High Level Overview

Provide interface for users to generate simulated medical decision making datasets.

### Why?

Current environments inappropriate.

Medical ML development is limited by access to useful data.

Potential use for teaching in medicine.

Give us better abilities to understand decision making.

Potential to collect data from human-in-the-loop experiments with clinicians.

Key points:

- Partially-observed
- Noise
- Missing Data

Bonus points:

- Create-your-own-envrionment function (Take in dataset, produce environment)
- Baseline IL/IRL methods

Far Future:

- Unification of various parts of the lab's work under common interface.

Ultimately I think there should be two settings for the data generation:

1. Highest possible fidelity 
2. Structurally interpretable and customisable

(2) is clearly more interesting to us but (1) will be a very useful selling point. 

**Online/Offline**

Initial focus should be of course offline, I imagine something like:

```python
import medkit as mk

synthetic_dataset = mk.generate_data(setting = 'Cystic Fibrosis',
					model = 'HMM', 
					markov_order = 2,
					size = 10_000, ...)

observations_train, actions_train = synthetic_dataset['training']
observations_test,  actions_test  = synthetic_dataset['testing']
```

But it would also be useful to have access to live simulation as a gym environment, e.g.:

```python
env = mk.make_gym(setting = 'Cystic Fibrosis', 
			model = 'HMM', 
			markov_order = 2, ...)
observation = env.reset()
observation, reward, done, info = env.step(action)
```

### Proof of concept

I've implemented a toy optimal stopping problem as proof of concept. 
With package installed can be can be called simply as:

```python
data = mk.generate_data(setting = 'Optimal Stopping',
						test_split      = True,
                        n_diseases      = 6,
                        test_accuracy   = 0.3,
                        conf_threshold  = 0.8,
                        test_acc_belief = 0.3)
```

We then have a synthetic dataset of observation, action, reward tuples. Easy.