#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot
import collections


# ### Entropy
#  - Compute the entropy of a vector $Y$ by considering the counts of the unique values $(y_1, ... y_k)$, in $Y$
#  - Returns the entropy of $Y: H(Y) = - P_Y(y_1)log_2(P_Y(y_1)) - ... - P_Y(y_k)log_2(P_Y(y_k))$

# In[6]:


def entropy(y):
    _, ycount = np.unique(y, return_counts=True)
    p = ycount / len(y)
    return -np.sum(p * np.log2(p))


# ### Partition

# In[25]:


def partition(x):
    x_unique = np.unique(x)
    part = {i: np.where(x == i)[0] for i in x_unique}
    return part


# ### Mutual Information

# In[26]:


def mutual_information(x, y):
    
    
    H_y = entropy(y)
    N = len(x)
    uniques = np.unique(x, return_counts=True)
    H_yx = 0
    for unique_x, x_count in zip(*uniques):
        H_yx += x_count / N * entropy(y[x == unique_x])
    
    return H_y - H_yx
    

    


# ### Visualization 

# In[9]:


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


# In[10]:



# ### ID3 Algorithm
# 
#  - Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
#     attribute-value pairs to consider. 
#  - This is a recursive algorithm that depends on three termination conditions:
#        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
#        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
#            value of y (majority label)
#        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
#     Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
#     and partitions the data set based on the values of that attribute before the next recursive call to ID3.
# 
#  - The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1 (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
#     attributes with their corresponding values:
#     
#     \[ ($x_1$, a),
#       ($x_1$, b,)
#       ($x_1$, c),
#       ($x_2$, d),
#       ($x_2$, e) \]
#      
#      If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
#      the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.
# 
#  - The tree is stored as a nested dictionary, where each entry is of the form
#                     (attribute_index, attribute_value, True/False): subtree
#     * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2) indicates that we test if (x4 == 2) at the current node.
#     * The subtree itself can be nested dictionary, or a single label (leaf node).
#     * Leaf nodes are (majority) class labels
# 
#  - Returns a decision tree represented as a nested dictionary, for example
#         {(4, 1, False):
#             {(0, 1, False):
#                 {(1, 1, False): 1,
#                  (1, 1, True): 0},
#              (0, 1, True):
#                 {(1, 1, False): 0,
#                  (1, 1, True): 1}},
#          (4, 1, True): 1}

# In[36]:


def id3(x, y, attributes=None, depth=0, max_depth=10):
    unique_y, ycount = np.unique(y, return_counts=True)
    
    if len(unique_y) == 1:
        return unique_y[0]
    
    if len(ycount)==0:
        return None
    elif len(attributes) == 0 or depth == max_depth:
        return unique_y[np.argmax(ycount)]

    gain = []
    for attribute, value in attributes:
        gain.append(mutual_information(x[:, attribute] == value, y))
    
    best_split_idx = np.argmax(gain)
    
    attr, val = attributes[best_split_idx]
    attributes = np.delete(attributes, best_split_idx, axis=0)
    
    left_idx = np.where(x[:, attr] == val)[0]
    right_idx = np.where(x[:, attr] != val)[0]
    
    x_left = x[left_idx]
    y_left = y[left_idx]
    x_right = x[right_idx]
    y_right = y[right_idx]
    
    right_tree = id3(x_right, y_right, attributes, depth+1, max_depth)
    left_tree = id3(x_left, y_left, attributes, depth+1, max_depth)
    
    if right_tree is not None and left_tree is not None:
        node = {(attr, val, False):right_tree ,
                (attr, val, True):left_tree }
        return node
    
    return unique_y[np.argmax(ycount)]


# ### Predict Example

# In[46]:


def predict_example(x, tree):
    if type(tree) is not dict:
        return tree
    for attribute, val, test in tree:
        if (x[attribute] == val) == test:
            return predict_example(x, tree[(attribute, val, test)])


# ### Compute Error

# In[79]:


def compute_error(y_true, y_pred):
    N=len(y_true)
    return sum(yt != yp for yt, yp in zip(y_true, y_pred)) / N


# ### Main Function

# In[63]:


def main(Xtrain, Ytrain, Xtest, Ytest, final_depth, initial_depth=1):

    attr_val_pair = []
    for attribute in range(len(Xtrain[0])):
        for val in np.unique(Xtrain[:, attribute]):
            attr_val_pair.append((attribute, val))
    
    train_error = []
    test_error = []
    
    for depth in range(initial_depth, final_depth+1):
        decision_tree = id3(Xtrain, Ytrain, attr_val_pair, max_depth=depth)
        
        y_pred_trn = [predict_example(x, decision_tree) for x in Xtrain]
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtest]
        
        train_error.append(compute_error(Ytrain, y_pred_trn))
        test_error.append(compute_error(Ytest, y_pred_tst))
        
    return y_pred_tst,train_error, test_error, decision_tree
        
        
        
    


# ### Plot errors

# In[56]:


def plot_errors(train_error, test_error, plot_title=None, save=False):
    
    plt.plot(list(range(1, len(train_error)+1)), train_error, label="train error")
    plt.plot(list(range(1, len(test_error)+1)), test_error, label="test error")
    plt.title(plot_title)
    plt.legend()
    plt.savefig(plot_title.replace(' ','_') + '.png')
    plt.show()


# ### Creating Confusion Matrix

# In[16]:


def confusion_tree(ytst, y_pred_tst):
    cm = confusion_matrix(ytst, y_pred_tst)
    print(cm)
    labels=['positive','negative']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Classifier Prediction')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual Value')
    plt.show()


# ### 

# ### load data

# In[48]:


def load_data(M,N):
    ytrain = M[:, 0]
    Xtrain = M[:, 1:] 
    
    ytest = N[:, 0]
    Xtest = N[:, 1:]
    
    return Xtrain,ytrain,Xtest,ytest
    


# In[72]:


def my_decision_tree(M,N,depth):
    Xtrain,ytrain,Xtest,ytest=load_data(M,N)
    y_pred_tst,train_error, test_error,decision_tree = main(Xtrain, ytrain, Xtest, ytest, depth)
    plot_errors(train_error, test_error, "Dataset error plots", True)
    
    return y_pred_tst, ytest, decision_tree
   
    


# ### Question 1

# ### Monks-1

# In[80]:


M = np.genfromtxt('data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
N = np.genfromtxt('data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
depth=10

y_pred_tst,ytst,decision_tree = my_decision_tree(M,N,depth)


# ### Monks-2

# In[86]:


M = np.genfromtxt('data/monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
N = np.genfromtxt('data/monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)


depth=10

y_pred_tst,ytest,decision_tree = my_decision_tree(M,N,depth)


# ### Monks-3

# In[85]:


M = np.genfromtxt('data/monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
N = np.genfromtxt('data/monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)


depth=10

y_pred_tst,ytest,decision_tree = my_decision_tree(M,N,depth)


# ### Question 2

# In[84]:


M = np.genfromtxt('data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
N = np.genfromtxt('data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)

print("For Depth=1")
depth=1
y_pred_tst,ytest,decision_tree = my_decision_tree(M,N,depth)
visualize(decision_tree)
confusion_tree(ytest, y_pred_tst)

print("For Depth=2")
depth=2
y_pred_tst,ytest,decision_tree = my_decision_tree(M,N,depth)
visualize(decision_tree)
confusion_tree(ytest, y_pred_tst)


# ### Question 3

# In[81]:


M = np.genfromtxt('data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
N = np.genfromtxt('data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)

Xtrain,ytrain,Xtest,ytest=load_data(M,N)
model = tree.DecisionTreeClassifier() # for classification, here default is gini
model.fit(Xtrain, ytrain)

#creates a dot file
dot_data = StringIO() 
tree.export_graphviz(model, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_pdf("dtree(Q3).pdf")


#Predict Output
predicted = model.predict(Xtest)
print("accuracy: "+str(model.score(Xtest,ytest)))
print("confusion matrix is below:-")
confusion_tree(ytest,predicted)


# ### Question 4

# In[88]:


M = np.genfromtxt('data/breast-cancer.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
N = np.genfromtxt('data/breast-cancer.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)

print("For Depth=1")
depth=1
y_pred_tst,ytst,decision_tree=my_decision_tree(M,N,depth)
visualize(decision_tree)
confusion_tree(ytst, y_pred_tst)

print("For Depth=2")
depth=2
y_pred_tst,ytst,decision_tree=my_decision_tree(M,N,depth)
visualize(decision_tree)
confusion_tree(ytst, y_pred_tst)

print("Using Scikit-Learn")
Xtrain,ytrain,Xtest,ytest=load_data(M,N)
model = tree.DecisionTreeClassifier() # for classification, here default is gini
model.fit(Xtrain, ytrain)

#creates a dot file
dot_data = StringIO() 
tree.export_graphviz(model, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_pdf("dtree(Q4).pdf")


#Predict Output
predicted = model.predict(Xtest)
print("accuracy: "+str(model.score(Xtest,ytest)))
print("confusion matrix is below:-")
confusion_tree(ytest,predicted)
