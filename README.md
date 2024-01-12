# Deep Learning

## Introduction

This repository contains source code and material produced for the exam of the course Deep Learning, attended in the second semester of the academic year 2022-2023 of master's degree in Physics of Complex Systems at the University of Turin.

The course is teached by prof. Matteo Osella.

As part of the exam I discuss the Transformer model, implementing the code from paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) with the help of the related [YouTube video](https://www.youtube.com/watch?v=ISNdQcPhsts) by Umar Jamil and other sources.

## Environment

Python 3.10 is used for the code base.

### Repostory structure

Source code is stored and described in file `transformer.ipynb`. An effort to follow PEP 8 is done when writing code in notebook cells, but no automatic helper tool is used.

Figures used in the notebook are stored in folder `figures`.
In this folder, files matching the pattern `ModalNet-*.png` are copies of the original figures used in the paper, hence they are creation of the authors of the article. Other files are my own creation.

### Create standalone Python module

A Python module can be extracted from notebook `transformer.ipynb` through utilities like [nbformat](https://nbformat.readthedocs.io/en/latest/) or [nbconvert](https://nbconvert.readthedocs.io/en/latest/). Cells containing source code and comments are identified with tag `convert-module`.

Note that the converted code is likely not PEP 8 compliant. Additional post-processing could be required, as running [autopep8](https://pypi.org/project/autopep8/) or [black](https://black.readthedocs.io/en/stable/).
