import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="medkit",
    version="0.1.0",
    author="Alex J. Chan",
    author_email="alexjameschan@gmail.com",
    description="Medical sequential decision making simulation tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XanderJC/medkit-learn",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch >= 1.7.0",
        "tqdm >= 4.54.1",
        "numpy >= 1.19.1",
        "pandas >= 1.1.2",
        "opacus >= 0.13.0",
        "gym >= 0.17.2",
        "scikit-learn >= 0.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
