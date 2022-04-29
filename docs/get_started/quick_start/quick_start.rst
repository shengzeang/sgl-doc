###################
Quick Start
###################

In this short tutorial, we will qucikly go through the basic and advanced usage of SGL. 
The tutorial is composed of following parts:

.. contents::
    :local:


Basic usage
______________________________________________

In this part, we will introduce the basic usage of SGL, including how to excute graph-related tasks and how to use the NAS (Neural Architecture Search) functionality.

___________________________________
Execute graph-related tasks
___________________________________

SGL provides user-friendly interfaces to execute graph-related tasks, including node classification, node clustering, and link prediction.
In this tutorial, we will go through an example of excuting a node classification task with a SGC on the PubMed dataset.

Import datasets
>>>>>>>>>>>>>>>>>>>>>>

Here we import the PubMed dataset via the following code:

.. code:: python

    from sgl.datasets import Planetoid
    dataset = Planetoid(name="pubmed", root="./", split="official")

The :obj:`Planetoid` class contains three popular graph datasets: Cora, Citeseer, and PubMed. 

+ The 1st argument :obj:`name` indicates which dataset among the three to choose; 
+ The 2nd argument :obj:`root` indicates where to put the dataset files;
+ The 3rd argumnet :obj:`split` indicates the train/validation/test split.

SGL has integated many graph datasets other than the Planetoid datasets.
Please refer to `datasets part <../../api/datasets/datasets.html>`__ for the detailed information of each dataset.


Import GNN models
>>>>>>>>>>>>>>>>>>>>>>>

Here we import the GNN model SGC as follows:

.. code:: python

    from sgl.models.homo import SGC
    model = SGC(prop_steps=3, feat_dim=dataset.num_features, num_classes=dataset.num_classes)

SGL supports not only GNNs designed for homogeneous graphs but also GNNs designed for heterogeneous graphs.
The two different categories of models reside in :obj:`sgl.models.homo` and :obj:`sgl.models.hetero`, respectively.
The GNN model SGC is designed for homogeneous graphs, and thus can be imported from :obj:`sgl.models.homo`.
:obj:`SGC` class has three main arguments:

+ The 1st argument :obj:`prop_steps` stands for the propagation depth;
+ The 2nd argument :obj:`feat_dim` stands for the dimension of the input feature;
+ The 3rd argument :obj:`num_classes` stands for the dimension of the output representation.

Please refer to the `models part <../../api/models/models.html>`__ for more details of SGC and other GNN models.

   
Execute tasks
>>>>>>>>>>>>>>>>>>>>>>>>
The node classification task can be executed by the following code:

.. code:: python

    from sgl.tasks import NodeClassification
    device = "cuda:0"
    test_acc = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-5, epochs=200, device=device).test_acc

The users have to input the adopted dataset, the adopted GNN model, and several hyperparameters before executing a task.

The possible output of the above code might be:

.. code:: bash

    Preprocessing done in 0.1280s
    Epoch: 001 loss_train: 1.0985 acc_train: 0.3333 acc_val: 0.2300 acc_test: 0.2110 time: 1.3086s
    Epoch: 002 loss_train: 1.0289 acc_train: 0.3667 acc_val: 0.7100 acc_test: 0.6920 time: 0.0030s
    Epoch: 003 loss_train: 0.9554 acc_train: 0.8667 acc_val: 0.7220 acc_test: 0.7300 time: 0.0040s
    Epoch: 004 loss_train: 0.8918 acc_train: 0.9333 acc_val: 0.7220 acc_test: 0.7300 time: 0.0030s
    Epoch: 005 loss_train: 0.8354 acc_train: 0.9167 acc_val: 0.7400 acc_test: 0.7220 time: 0.0020s
    Epoch: 006 loss_train: 0.7835 acc_train: 0.9333 acc_val: 0.7380 acc_test: 0.7180 time: 0.0030s
    Epoch: 007 loss_train: 0.7358 acc_train: 0.9167 acc_val: 0.7280 acc_test: 0.7240 time: 0.0020s
    Epoch: 008 loss_train: 0.6929 acc_train: 0.9333 acc_val: 0.7320 acc_test: 0.7320 time: 0.0030s
    Epoch: 009 loss_train: 0.6546 acc_train: 0.9333 acc_val: 0.7360 acc_test: 0.7340 time: 0.0030s
    Epoch: 010 loss_train: 0.6198 acc_train: 0.9333 acc_val: 0.7360 acc_test: 0.7360 time: 0.0030s
    ......
    Epoch: 191 loss_train: 0.1886 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 192 loss_train: 0.1886 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7930 time: 0.0030s
    Epoch: 193 loss_train: 0.1885 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 194 loss_train: 0.1884 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 195 loss_train: 0.1884 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 196 loss_train: 0.1883 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0020s
    Epoch: 197 loss_train: 0.1882 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7930 time: 0.0040s
    Epoch: 198 loss_train: 0.1882 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 199 loss_train: 0.1881 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 200 loss_train: 0.1880 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7910 time: 0.0030s
    Optimization Finished!
    Total time elapsed: 1.9751s
    Best val: 0.8020, best test: 0.7920

Please refer to the `tasks part <../../api/tasks/tasks.html>`__ for more details of executing graph-related tasks.

_________________________________________
Auto neural architrcture search (TODO)
_________________________________________




Advanced usage
___________________________________



____________________________________________
Build models under SGAP paradigm
____________________________________________


________________________________________
Implement user-defined graph operators
________________________________________



_________________________________________
Implement user-defined message operators
_________________________________________


_______________________________________
Adopt user-defined datasets
_______________________________________

