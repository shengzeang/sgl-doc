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
In this tutorial, we will go through an example of excuting a node classification task with a SGC on the Cora dataset.

Import datasets
>>>>>>>>>>>>>>>>>>>>>>

SGL has integated many graph datasets. 
Please refer to `datasets part <../../api/datasets/datasets.html>`__ for the detailed information of each dataset.

Here we import the Cora dataset via the following code:

.. code:: python

    from sgl.datasets import Planetoid
    dataset = Planetoid(name="cora", root="./", split="official")

The :obj:`Planetoid` class contains three popular graph datasets: Cora, Citeseer, and PubMed.
The first argument :obj:`name` indicates which dataset among the three to choose; 
the second argument :obj:`root` indicates where to put the dataset files;
and the third argumnet :obj:`split` indicates the train/validation/test split.


Import GNN models
>>>>>>>>>>>>>>>>>>>>>>>
   
Execute tasks
>>>>>>>>>>>>>>>>>>>>>>>>


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

