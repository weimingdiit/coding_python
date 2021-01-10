import treelib
from treelib import Tree, Node


class Nodex(object):
    def __init__(self, num):
        self.num = num


def createTree():
    tree1 = Tree()
    tree1.create_node('Root', 'root', data=Nodex('3'))
    tree1.create_node('Child1', 'child1', parent='root', data=Nodex('4'))
    tree1.create_node('Child2', 'child2', parent='root', data=Nodex('5'))
    tree1.create_node('Child3', 'child3', parent='root', data=Nodex('6'))
    tree1.show()
    tree1.show(data_property = 'num')
    print(tree1.nodes['child1'].data.num)
    print(tree1.nodes.values())


if __name__ == "__main__":
    createTree()
