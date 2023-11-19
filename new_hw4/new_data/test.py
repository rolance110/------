import numpy as np


def entropy(examples):
    """計算熵。"""
    counts = np.unique(examples[:, -1], return_counts=True)
    p = counts[1] / len(examples)
    print("p ", p)
    return -p * np.log2(p)


def importance(attribute, examples):
    """計算屬性的重要性。"""
    print("attribute ", attribute)
    left_examples = examples[examples[:, attribute] == 0]
    right_examples = examples[examples[:, attribute] == 1]
    entropy_total = entropy(examples) - \
        (entropy(left_examples) + entropy(right_examples)) / 2
    print("entropy_total ", entropy_total)
    return entropy_total


def decision_tree_learning(examples, attributes, parent_examples):
    """決策樹學習。"""
    if len(examples) == 0:
        return parent_examples.mode(axis=0)[0]
    elif all(examples[:, -1] == examples[0, -1]):
        # return examples[:, -1].item()
        return examples[0, -1]
    elif len(attributes) == 0:
        # return parent_examples.mode(axis=0)[0] # 1
        return np.argmax(np.bincount(examples[:, -1].astype(int)))
    else:
        print("attributes ", attributes)
        index = np.argmax([importance(i, examples) for i in attributes])
        print("index ", index) 
        attribute = attributes[index-1] # 選出entropy最大的屬性  
        print("attribute_value ", attributes[index-1]) # !
        tree = DecisionTree(attribute)
        for value in np.unique(examples[:, attribute]):
            exs = examples[examples[:, attribute] == value]
            # subtree = decision_tree_learning(exs, attributes - [attribute], examples)
            subtree = decision_tree_learning(
                exs, [attr for attr in attributes if attr != attribute], examples)
            tree.add_branch(value, subtree)
        return tree


class DecisionTree(object):
    """決策樹類別。"""

    def __init__(self, attribute):
        self.attribute = attribute
        self.left = None
        self.right = None

    def add_branch(self, value, subtree):
        if value < self.attribute:
            self.left = subtree
            self.right = DecisionTree(value)
        else:
            self.left = DecisionTree(value)
            self.right = subtree

    def classify(self, example):
        print()
        print("predicting...")
        print("example ", example)
        print("attribute ", self.attribute)
        
        if example[self.attribute] < self.attribute:
            if self.left is not None:
                print("left ", self.left.classify(example))
                return self.left.classify(example)
            else:
                return None
        else:
            print("right ", self.right.classify(example))
            return self.right.classify(example)



# 生成數據集
examples = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 1, 1],
    [1, 1, 1]
])
attributes = [1, 0]
parent_examples = None
# 訓練決策樹
tree = decision_tree_learning(examples, attributes, parent_examples)

# 預測結果
print(tree.classify([1, 0]))
# 輸出：0
