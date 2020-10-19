import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="medkit",
    version="0.0.1",
    author="Alex J. Chan",
    author_email="alexjameschan@gmail.com",
    description="Medical decision making simulation tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XanderJC/medkit-learn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)