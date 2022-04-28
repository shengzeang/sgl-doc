# Overview

**SGL** is a Graph Neural Network (GNN) toolkit targeting scalable graph learning, which supports deep graph learning on extremely large datasets. SGL allows users to easily implement scalable graph neural networks and evaluate its performance on various downstream tasks like node classification, node clustering, and link prediction. Further, SGL supports auto neural architecture search functionality based on <a href="https://github.com/PKU-DAIR/open-box" target="_blank" rel="nofollow">OpenBox</a>. SGL is designed and developed by the graph learning team from the <a href="https://cuibinpku.github.io/index.html" target="_blank" rel="nofollow">DAIR Lab</a> at Peking University.



## Main Functionalities

+ A handy platform for **implementing and evaluating scalable GNNs**.
+ Scalable learning on various graph-related tasks, including **node classification**, **node clustering**, and **link prediction**.
+ **Auto neural architecture search** on given tasks, datasets and objectives.



## Training paradigm

The main design goal of SGL is to support scalable graph learning. SGL adopts the scalable training paradigm, **SGAP** (**S**calable **G**raph **A**rchitecture **P**aradigm), in <a href="https://arxiv.org/abs/2203.00638" target="_blank" rel="nofollow">PaSca</a>. **SGAP** split the conventional GNN training process into three independent stages --- **Preprocessing**, **Training**, and **Postprocessing**, which can be represented as follows: 

+ **Preprocessing**: $\textbf{M}=graph\_propagate(\textbf{A}, \textbf{X})$; $\textbf{X}'=message\_aggregate(\textbf{M})$
    + **SGAP** propagates and aggregates information at the graph level.

+ **Training**: $\textbf{Y}=model\_train(\textbf{X}')$
    + **SGAP** feeds the propagated and aggregated information into a machine learning model (e.g., SVM, MLP) for training.

+ **Postprocessing**: $\textbf{M}'=graph\_propagate(\textbf{A},\textbf{Y})$;  $\textbf{Y}'=message\_aggregate(\textbf{M}')$
    + **SGAP** again propagates and aggregates the outputs of the previous stage at the graph level.


To note that, the first $message\_aggregate$ operation in the **Preprocessing** stage will be transferred to the **Training**  stage if it contains learnable parameters; and the second $message\_aggregate$ operation in the **Postprocessing** stage is prohibited to contain learnable parameters.

Compared to conventional GNN training process, **SGAP** has mainly two advantages:

1. The time- and resource-consuming propagation operation is only executed two times during the full training process; while the number of executing propagation in the conventional GNN training process equals to the number of training epochs, which is usually far greater than two.
2. The dependencies between training examples have been fully taken care of in the **Preprocessing** stage. Thus, the training examples can be freely split to small batches to feed into the model in the **Training** stage, which boosts the efficiency and the scalability of the training process.



## Model construction paradigm

Corresponding to its training paradigm, **SGAP**, SGL needs to define the behaviors of two $graph\_propagate$s, two $message\_aggregate$s, and $model\_train$ for each GNN model. To fulfill this goal, SGL designs three important modules:

+ **Graph Operator**: to carry out the functionality of $graph\_propagate$. It receives the adjacency matrix $\textbf{A}$ and the node representation matrix $\textbf{X}$, and outputs a list of propagated information matrices of different propagation depths.
+ **Message Operator**: to carry out the functionality of $message\_aggregate$. It receives a list of propagated information matrices and aggregates the matrices according to pre-defined behaviors. The final output of each **Message Operator** is a single matrix.
+ **Base Model**: to carry out the functionality of $model\_training$. It can be not only deep learning models like MLP, but also traditional machine learning methods like SVM and random forest.

To construct a GNN model in SGL, the users only need to fill in some blanks with pre-/user-defined **Graph Operators**, **Message Operators** and **Base Models**. The code of SGC under SGL is as follows:

```python
class SGC(BaseSGAPModel):
    def __init__(self, prop_steps, feat_dim, num_classes):
        super(SGC, self).__init__(prop_steps, feat_dim, num_classes)

        self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
        self._pre_msg_op = LastMessageOp()
        self._base_model = LogisticRegression(feat_dim, num_classes)
```

$LaplacianGraphOp$, $LastMessageOp$,and $LogisticRegreesion$ are pre-defined **Graph Operator**, **Message Operator**, and **Base Model**, respectively. Please refer to [models part](../../models/models.md) for the detailed API for constructing models. SGL also provides simple interfaces for defining new **Graph Operators** and **Message Operators**, please refer to [operators part](../../operators/operators.md) for more details.
