import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="toopazo_ulg",    # Replace with your own username
    version="0.0.8",
    author="toopazo",
    author_email="toopazo@protonmail.com",
    description="Python package to process and plot Px4 .ulg files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toopazo/toopazo_ulg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
