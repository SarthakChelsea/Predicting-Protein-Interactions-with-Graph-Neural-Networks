# Protein-Protein Interaction Prediction using Graph Convolutional Networks

## Overview
This project aims to predict protein-protein interactions using a graph convolutional neural network (GCN) approach. The key steps are:

1. **Dataset**: The dataset used is the yeast protein-protein interaction network. This is a graph-structured dataset, where proteins are represented as nodes and their interactions as edges.

2. **Model**: The researchers utilize a GCN to learn the node (protein) embeddings based on the network topology. GNNs are a class of neural networks that can operate directly on graph-structured data.

3. **Interaction Prediction**: To define the interaction between two proteins, the researchers use the inner product of the learned node embeddings. The higher the inner product, the more confident the model is that the two proteins interact.

4. **Training**: Since some protein interactions are already known (the existing edges in the graph), the researchers train the GCN in a supervised manner. They use the known interactions as positive training examples, and some non-interacting protein pairs as negative examples. The model is trained to minimize the cross-entropy loss between the predicted and true interactions.

5. **Inference**: After training the GCN, the researchers obtain stable node embeddings for each protein. They then apply the inner product operation to all pairs of nodes (proteins) and predict an interaction if the result is higher than a certain threshold.

6. **Implementation**: The researchers implemented this approach using the TensorFlow library, a popular open-source machine learning framework.

## Results
The GCN model achieved the following performance on the test set:

- Test ROC score: 0.87898
- Test AP score: 0.86944

These results demonstrate the effectiveness of the GCN approach in predicting new protein-protein interactions based on the yeast protein interaction network.

## Usage
To run the GCN model, you'll need to have the following dependencies installed:

- TensorFlow
- NumPy
- SciPy
- Scikit-learn
- NetworkX

Once you have the dependencies installed, you can run the Jupyter Notebook provided in the `yeast.edgelist` file.

## Citation
If you use this code or the results in your research, please cite the following paper:

```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```