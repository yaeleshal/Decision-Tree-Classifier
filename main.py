from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import anytree
from anytree.exporter import DotExporter


#A

X = load_breast_cancer().data
y = load_breast_cancer().target
z = load_breast_cancer().feature_names
names = load_breast_cancer().target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)

#B

tree_c = tree.DecisionTreeClassifier(random_state=2, criterion='entropy')
tree_c = tree_c.fit(X_train,y_train)

train_pred = tree_c.predict(X_train)
test_pred = tree_c.predict(X_test)

acc1 = accuracy_score(y_train, train_pred)
acc2 = accuracy_score(y_test, test_pred)

print("Sklearn accuracy score for train data: ", acc1)
print("Sklearn accuracy score for test data: ", acc2)

dot_data = export_graphviz(tree_c, out_file=None, feature_names=z, class_names=names, filled=True, rounded=True,
                special_characters=True)
graph = graphviz.Source(dot_data)
graph.save(filename='my decision tree.dot')

plt.figure(figsize=(25,8))
tree.plot_tree(tree_c, feature_names=z, class_names=names, rounded=True, filled=True)
plt.show()


#C

class DecisionTreeNode(anytree.Node):
    def __init__(self, num_samples, values, entropy, parent=None, children=None):
        if values[0] > values[1]:
            class_ = 1
        else:
            class_ = 0
        self.pred_class = class_
        if parent:
            self.parent = parent
        if children:
            self.children = [children]
        self.entropy = entropy
        self.num_samples = num_samples
        self.values = values
        self.threshold = 0
        self.feature = 0
        self.name = None


def calculate(value1, value2):
    calc_value = value1 / (value2+value1)
    return calc_value


def count_values(data):
    counter_0 = 0
    counter_1 = 0
    for row in data:
        if row[(len(row)-1)] == 0:
            counter_0 += 1
        else:
            counter_1 += 1
    return [counter_1, counter_0]


def entropy_measurement(data):
    counter_of_1, counter_of_0 = count_values(data)
    if counter_of_1 == 0 or counter_of_0 == 0:
        return 0
    p1 = calculate(counter_of_1,counter_of_0)
    p2 = calculate(counter_of_0,counter_of_1)
    s = -(p1 * np.log2(p1)) - (p2 * np.log2(p2))
    return s


def splitting(idx, value, set):
    right = []
    left = []
    for row in set:
        if row[idx] <= value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def split(set):
    c1 = 0
    c2 = 0
    for row in set:
        if row[(len(row) - 1)] == 0:
            c1 = 1
        else:
            c2 = 1
    if c1 == 0 or c2 == 0:
        return None, None

    best_entropy = np.inf
    best_res1 = None
    best_res2 = None

    for idx in range((len(set[0]) - 1)):
        for row in set:
            temp = 0.0
            right, left = splitting(idx, row[idx], set)
            right_size = float(len(right))
            left_size = float(len(left))
            if right:
                temp += entropy_measurement(right) * calculate(right_size, left_size)
            if left:
                temp += entropy_measurement(left) * calculate(left_size, right_size)
            if temp < best_entropy:
                best_entropy = temp
                best_res1 = idx
                best_res2 = row[idx]
    return best_res1, best_res2


def prediction(data, root):
    pred = []
    for i in data:
        node = root
        ent = node.entropy
        while ent != 0:
            if i[node.feature] <= node.threshold:
                node = node.children[0]
            else:
                node = node.children[1]
            ent = node.entropy
        class_pred = node.pred_class
        pred.append(class_pred)
    return pred


def CART(node, set, features_names):
    idx, threshold = split(set)
    e = node.entropy
    if e != 0:
        left, right = splitting(idx, threshold, set)
        left_values = count_values(left)
        right_values = count_values(right)
        left_entropy = entropy_measurement(left)
        right_entropy = entropy_measurement(right)
        left_node = DecisionTreeNode(num_samples=len(left), values=left_values, parent=node, entropy=left_entropy)
        right_node = DecisionTreeNode(num_samples=len(right), values=right_values, parent=node, entropy=right_entropy)
        node.name = f'{features_names[idx]} <= {threshold}'
        node.feature = idx
        node.threshold = threshold
        node.children = [left_node, right_node]
        if left_entropy != 0:
            CART(left_node, left, features_names)
        if right_entropy != 0:
            CART(right_node, right, features_names)
    return


def save_dot(root, classes):
    def node_name(node):
        return '%s\nentropy=%s\nsamples=%s\nvalues=%s\nclass=%s' % \
               (node.name, node.entropy, node.num_samples, node.values, classes[node.pred_class])

    def edge_attr(node, child):
        return f'label={True if child == node.children[0] else False}'

    def edge_type(node, child):
        return '--'

    with open('graph.dot', 'w') as file:
        for line in DotExporter(root, graph="graph", nodenamefunc=node_name,
                                nodeattrfunc=lambda node: "shape=box ,fontcolor=blue ,style=filled",
                                edgeattrfunc=edge_attr, edgetypefunc=edge_type):
            file.write(line)

#D

new_train = np.c_[X_train, y_train]
new_test = np.c_[X_test, y_test]
node_v = count_values(new_train)
e = entropy_measurement(new_train)
node = DecisionTreeNode(num_samples=len(new_train), values=node_v, entropy=e)
CART(node, new_train, z)
train_pred1 = prediction(new_train, node)
test_pred1 = prediction(new_test, node)
acc3 = accuracy_score(y_train, train_pred1)
acc4 = accuracy_score(y_test, test_pred1)

print("Our new accuracy score for train data: ", acc3)
print("Our new accuracy score for test data: ", acc4)

save_dot(node, [0, 1])
DotExporter(node).to_dotfile("tree.dot")

#E

my_confusion_matrix = confusion_matrix(y_train, train_pred1)
sn.heatmap(my_confusion_matrix, annot=True, cmap='Pastel1', fmt='g')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Our New CART Tree : Train')
plt.show()

my_confusion_matrix = confusion_matrix(y_test, test_pred1)
sn.heatmap(my_confusion_matrix, annot=True, cmap='Pastel1', fmt='g')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Our New CART Tree : Test')
plt.show()

my_confusion_matrix = confusion_matrix(y_train, train_pred)
sn.heatmap(my_confusion_matrix, annot=True, cmap='Pastel1', fmt='g')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Sklearn CART Tree : Train')
plt.show()

my_confusion_matrix = confusion_matrix(y_test, test_pred)
sn.heatmap(my_confusion_matrix, annot=True, cmap='Pastel1', fmt='g')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Sklearn CART Tree : Test')
plt.show()

