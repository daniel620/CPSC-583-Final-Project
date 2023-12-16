# CPSC 583 Final Project
 
How to incorporate the power of Transformer into Graph Neural Network (GNN) has been a heated topic recent days. Though Graph Transformers can naturally capture long-distant information which is difficult for traditional Message-Passing GNN (MPNN), they need to focus more on local-structural and positional information for nodes because they gain global attention at cost of the discard of graph topological information. 

Meanwhile, positional and structural encodings for nodes have also been well studied to increase the expressiveness of MPNNs. However, current methods are mostly domain-specific or task-specific, which generalizability and transferability are limited. An idea of pre-training positional and structural encodings based only on graph topological structures has been recently proposed but also lacks expressiveness due to its non-position-aware MPNN structure.

Therefore, an position-aware and general pre-training method has been proposed to solve the limitations above. The idea adopts the training method from Contrastive Learning and incorporates the power of Graph Transformer, aiming to produce positional and structural encodings for general purposes. 

Although results show the proposed model’s poor generalizability in unseen test dataset, the training framework of encoding non-low-dimension feature of shortest path distance into low-dimension vector has been proved effective. The result may lead future researchers focusing more on generalizability of the proposed model.

# Pipeline
## Pretraining stage
<img width="923" alt="image" src="https://github.com/daniel620/CPSC-583-Final-Project/assets/60392737/f1b720c0-f85d-40e1-911b-94664f7e912f">
## Downstream stage
<img width="780" alt="image" src="https://github.com/daniel620/CPSC-583-Final-Project/assets/60392737/247ebbb8-d06f-4de4-8843-cb244066b450">
