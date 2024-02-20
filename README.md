# Iterative Puzzle Solver
Project page for *Introduction to Computational and Biological Vision* class taken at BGU. 

## Introduction
In the square jigsaw puzzle problem one is required to reconstruct the complete image from a set of non-overlapping, unordered, square puzzle parts.  
Although puzzles are well known popular games, the automated solving is of great interest in different scientific
areas like in archaeology. Reconstruction of archaeological finds from fragments, is a tedious task requiring many hours of work from the
archaeologists and restoration personnel.  
In our problem we assume no clues regarding parts’ location and no prior knowledge about the original image. The pieces of the 2d puzzle are square patches which combined together contain the full image and there is no overlap between pieces.  
The goal of this work is to solve iteraively jigsaw puzzles by recovering images from shuffled versions of it.  

## Approach and Method 
I propose a new unsupervised deep learning model based on previous work of puzzle solving, using learning of the inverse permutation matrix in reliance of time phase.     
Each timestep *t* corresponds to a certain step in the solving procedure.  
The model outputs for each piece(patch) a probability distribution of how likely for piece to originated from a location on the image grid.  
Aspired by the way children solve puzzles, starting with most confident pieces, I use the model's output  
to take the k-best pieces and first permute those to their suggested location. This is done by solving the k-cardinality assignment problem  
with a modification of the Hungarian Algorithm.  
As we progress with the solving process, we decay *k* to ensure convergence.   

**Sources:**
- LEARNING LATENT PERMUTATIONS WITH GUMBELSINKHORN NETWORKS, [Paper.](https://arxiv.org/pdf/1802.08665.pdf)
- DeepPermNet: Visual Permutation Learning, [Paper.](https://basurafernando.github.io/papers/CVPR_2017_DeepPermNet.pdf)
- Diffusion Models Beat GANs on Image Synthesis, [Paper.](https://arxiv.org/abs/2105.05233)
- Solving the k-cardinality assignment problem by transformation, [Paper.](https://www.sciencedirect.com/science/article/pii/S0377221703002054)
- Attention Is All You Need, [Paper.](https://arxiv.org/pdf/1706.03762.pdf)
- Github implementation [Denoising Diffusion Pytorch.](https://github.com/lucidrains/denoising-diffusion-pytorch)
- Niels Rogge, Kashif Rasul, [Huggingface notebook.](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=3a159023)

## Conclusions
- Average accuracy is about 77% with about of 30% perfect assembly.
- Learning permutation matrix via neural networks is a suitable method for the jigsaw puzzle solving.  
- The relaxation layer fits properly to this task and improves the initial guess derived from the network.   
- Opposed to my intuition, solving iteratively did not improve the accuracy. It adds computational complexity and can be neglected.  
- The model performance dropped significantly for 64 pieces puzzle. This might be training issue as no major changes done when switched to 64 pieces training.

## Future Work
Due to the POC nature of this project and the results obtained, some improvements can be considered:  
- [ ] Replace encoding network(VGG11) with SOTA encoder.  
- [ ] Replace compatibility function(sum of square differences) with a more suitable function. Such as Mahalanobis gradient compatibility and its derivatives.  
- [ ] Use coarse-to-fine architecture to compute the similarity between patches(see similar idea [here](https://arxiv.org/pdf/2103.15545.pdf))  
- [ ] Replace the relaxation layer with CNN which approximate the support function. This might add generality and reduce inference time.  
- [ ] Train on larger and more diverse dataset.  
- [ ] Train and test in puzzles with missing pieces.

