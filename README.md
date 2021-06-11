

<img src="imgs/logo.png" height="120" width=auto>

# The Medkit-learn(ing) Environment

### Alex J. Chan, Ioana Bica, Alihan Huyuk, Daniel Jarrett, and Mihaela van der Schaar

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 
Medkit is a synthetic data generation library for medical sequential decision problems.

The aim is to solve two problems commonly faced when applying machine learning in the area:

1. Provide universal access to synthetic data so that anyone with an interest can develop and test their algorithms.
2. Provide extensive customisation of the generating process in order to better benchmark and test novel algorithms against ground truth properties.

Medkit is pip installable - we recommend cloning it, optionally creating a virtual env, and installing it (this will automatically install dependencies):

```shell
git clone https://github.com/XanderJC/medkit-learn.git

cd medkit-learn

pip install -e .
```


Example usage:
```python
import medkit as mk

synthetic_dataset = batch_generate(
	domain = "Ward",
	environment = "CRN",
	policy = "LSTM",
    size = 1000,
	test_size = 200,
	max_length = 10,
	scale = True
)

static_train, observations_train, actions_train = synthetic_dataset['training']
static_test,  observations_test,  actions_test  = synthetic_dataset['testing']
```

While medical machine learning is by necessity almost always entirely offline, we also provide an interface through which you can interact online with the environment should you find that useful. For example, you could train a custom RL policy on this environment with a specified reward function, then you can test inference algorithms on their ability to represent the policy.

```python
env = mk.live_simulate(
    domain="ICU",
    environment="SVAE"
):

observation = env.reset()
observation, reward, info, done = env.step(action)
```

### Citing 

If you use this software please cite as follows:

```bib
@misc{chan2021medkitlearning,
      title={The Medkit-Learn(ing) Environment: Medical Decision Modelling through Simulation}, 
      author={Alex J. Chan and Ioana Bica and Alihan Huyuk and Daniel Jarrett and Mihaela van der Schaar},
      year={2021},
      eprint={2106.04240},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```