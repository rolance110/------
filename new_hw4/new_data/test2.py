import numpy as np

class DecisionTree:
    """Decision Tree class."""

    class Node:
        """Node class for the decision tree."""
        def __init__(self, attribute):
            self.attribute = attribute
            self.branches = {}

        def add_branch(self, value, subtree):
            self.branches[value] = subtree

    def __init__(self):
        self.root = None

    def fit(self, examples, attributes, parent_examples=None):
        """Build the decision tree."""
        self.root = self._decision_tree_learning(examples, attributes, parent_examples)

    def predict(self, example):
        """Predict the class of a given example."""
        return self._predict_recursive(example, self.root)

    def _predict_recursive(self, example, node):
        """Recursive function for prediction."""
        if isinstance(node, DecisionTree.Node):  # Check if the node is a Node instance
            value = example[node.attribute]
            if value in node.branches:
                return self._predict_recursive(example, node.branches[value])
            else:
                # Handle unknown values or missing branches
                return None
        else:
            return node  # Return the leaf value

    def _entropy(self, examples):
        """Calculate entropy."""
        counts = np.unique(examples[:, -1], return_counts=True)
        p = counts[1] / len(examples)
        return -np.sum(p * np.log2(p))

    def _importance(self, attribute, examples):
        """Calculate attribute importance."""
        left_examples = examples[examples[:, attribute] == 0]
        right_examples = examples[examples[:, attribute] == 1]
        entropy_total = self._entropy(examples) - \
            (self._entropy(left_examples) + self._entropy(right_examples)) / 2
        return entropy_total

    def _decision_tree_learning(self, examples, attributes, parent_examples):
        """Recursive decision tree learning."""
        if len(examples) == 0:
            return parent_examples.mode(axis=0)[0]
        elif all(examples[:, -1] == examples[0, -1]):
            # return examples[:, -1].item()
            return examples[0, -1]
        elif len(attributes) == 0:
            # return parent_examples.mode(axis=0)[0]
            return np.argmax(np.bincount(examples[:, -1].astype(int)))
        else:
            index = np.argmax([self._importance(i, examples) for i in attributes])
            attribute = attributes[index]
            tree = DecisionTree.Node(attribute)
            for value in np.unique(examples[:, attribute]):
                exs = examples[examples[:, attribute] == value]
                subtree = self._decision_tree_learning(
                    exs, [attr for attr in attributes if attr != attribute], examples)
                tree.add_branch(value, subtree)
            return tree

# 測試資料
test_data = np.array([
    [1, 0, 1],  # 類別 0
    [0, 0, 1],  # 類別 1
    [1, 1, 0],  # 類別 0
])

# 訓練資料
train_data = np.array([
    [1, 0, 0, 1],  # 類別 1
    [1, 1, 1, 0],  # 類別 0
    [0, 0, 1, 1],  # 類別 1
    [0, 1, 0, 0],  # 類別 1
    [1, 0, 1, 1],  # 類別 0
])

# 屬性名稱
# 列出0~len(train_data[0])-1
attributes = list(range(len(train_data[0]) - 1)) 

# 訓練模型
dt = DecisionTree()
dt.fit(train_data, attributes)

# 進行預測
dt = DecisionTree()
dt.fit(train_data, attributes)
predictions = [dt.predict(example) for example in test_data]

# 顯示結果
print("預測結果:", predictions)