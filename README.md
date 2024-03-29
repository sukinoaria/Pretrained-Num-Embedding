# Pretrained Numberic Embeddings by transformer

#### 

This repo is a **third-party implementation** of the **pre-trained numerical representation** in the paper:

[Enhancing Content Planning for Table-to-Text Generation with Data Understanding and Verification,EMNLP Findings 2020](https://www.aclweb.org/anthology/2020.findings-emnlp.262/)

The structure is based on the Transformer structure and the num comparison loss.

#### Details:
1. The values used for model training come from the **row and column values** of each table in the [ROTOWIRE dataset](https://github.com/harvardnlp/boxscore-data).

2. Compared with the original paper's method of encoding values, this repo **divides them into different intervals** according to the size of the value, and use **matrix multiply operations with different parameters**,  thereby achieve the ability to handle floating-point numbers.
