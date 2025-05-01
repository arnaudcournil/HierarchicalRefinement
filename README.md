# 🧠 Optimal Transport - Hierarchical Refinement on COCO 2017


This project implements and evaluates the **Hierarchical Refinement** algorithm introduced in the paper [*Hierarchical Refinement: Optimal Transport to Infinity and Beyond*](https://arxiv.org/pdf/2503.03025) by Halmos et al., on visual representations extracted from the **COCO 2017** dataset.


## 📚 Reference



Peter Halmos, Julian Gold, Xinhao Liu, Benjamin J. Raphael.
*Hierarchical Refinement: Optimal Transport to Infinity and Beyond*, 2025.
[arXiv:2503.03025](https://arxiv.org/abs/2503.03025)



## 🔍 Objective


The objective is to use **Hierarchical Refinement** to approximate an optimal transport between image representations extracted using a pre-trained **ResNet50**, on a subset of the **COCO 2017 (val)** dataset using JAX and OTT-JAX librairies.


## 🧪 Notebook structure


The `coco.ipynb` notebook:


- Loads the **5000 validation images** from COCO 2017.
- Extracts descriptors via a **ResNet50** (file `resnet50-0676ba61.pth` to be placed at the project root).
- Applies the **Hierarchical Refinement** method to the resulting representations.
- Compares transport plans and distances.

## 🗂️ File organization


```bash
├── bench.png
├── coco.ipynb # Main notebook
├── resnet50-0676ba61.pth # Pre-trained ResNet50 model
├── embeddings
├ └── embeddings.pkl # Embeddings
├── images/ # Directory containing COCO images (val2017)
│ ├── 000000000139.jpg
│ ├── ...
│ └── 000000581929.jpg
```


## 📦 Prerequisites

Install dependencies with :


```bash
pip install -r requirements.txt
```


## 📸 Download dataset


Download and unpack the COCO 2017 validation dataset:


```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017 images
```


Make sure the 5000 `.jpg` images are in the `images/` folder at the root of the project.


## 🧠 ResNet50 model


The template used is the **ResNet50 pre-trained on ImageNet**, in `torchvision` format. Manually download the `resnet50-0676ba61.pth` file and place it in the project root.
You can download it from:
[https://download.pytorch.org/models/resnet50-0676ba61.pth]()


## 🚀 Launch notebook

Open `coco.ipynb` in Jupyter Notebook or VSCode and run the cells.