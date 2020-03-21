import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="craft-text-detector",
    version="0.1.0",
    author="Clova AI Research, Fatih Cagatay Akyon",
    author_email="youngmin.baek@navercorp.com, fatihcagatayakyon@gmail.com",
    description="Character Region Awareness for Text Detection (CRAFT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fcakyon/craft_text_detector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    python_requires='>=3.6',
)
