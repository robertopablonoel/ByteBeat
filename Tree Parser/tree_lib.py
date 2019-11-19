from queue import Empty
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statistics import mean 
import librosa             # The librosa library
import librosa.display     # librosa's display module (for plotting features)
import math

class Error(Exception):
    def __init__(self, msg):
        self.msg = msg
        
class ArrayStack:
    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def push(self, e):
        self._data.append(e)

    def top(self):
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._data[-1]

    def pop(self):
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._data.pop(-1)

    def __repr__(self):
        return str(self._data)

class Node:
    def __init__(self, element, parent = None, left = None, right = None):
        self._parent = parent
        self._element = element
        self._left = left
        self._right = right
        self._height = 1
            
class Tree:
    class TreeNode:
        def __init__(self, element, parent = None, left = None, right = None):
            self._parent = parent
            self._element = element
            self._left = left
            self._right = right

    #-------------------------- binary tree constructor --------------------------
    def __init__(self):
        """Create an initially empty binary tree."""
        self._root = None
        self._size = 0

    #-------------------------- public accessors ---------------------------------
    def __len__(self):
        """Return the total number of elements in the tree."""
        return self._size

    def is_root(self, node):
        """Return True if a given node represents the root of the tree."""
        return self._root == node

    def is_leaf(self, node):
        """Return True if a given node does not have any children."""
        return self.num_children(node) == 0

    def is_empty(self):
        """Return True if the tree is empty."""
        return len(self) == 0

    def __iter__(self):
        """Generate an iteration of the tree's elements."""
        for node in self.nodes():                        # use same order as nodes()
            yield node._element                               # but yield each element

    def depth(self, node):
        """Return the number of levels separating a given node from the root."""
        if self.is_root(node):
            return 0
        else:
            return 1 + self.depth(self.parent(node))

    def _height1(self):                 # works, but O(n^2) worst-case time
        """Return the height of the tree."""
        return max(self.depth(node) for node in self.nodes() if self.is_leaf(node))

    def _height2(self, node):                  # time is linear in size of subtree
        """Return the height of the subtree rooted at the given node."""
        if self.is_leaf(node):
            return 0
        else:
            return 1 + max(self._height2(c) for c in self.children(node))

    def height(self, node=None):
        """Return the height of the subtree rooted at a given node.

        If node is None, return the height of the entire tree.
        """
        if node is None:
            node = self._root
        return self._height2(node)        # start _height2 recursion

    def nodes(self):
        """Generate an iteration of the tree's nodes."""
        return self.preorder()      
                              # return entire preorder iteration
    def operators(self):
        """Generate a preorder iteration of nodes in the tree."""
        if not self.is_empty():
            for node in self._subtree_preorder(self._root):  # start recursion
                if not self.is_leaf(node):
                    yield node, self.height(node)

    def operands(self):
        """Generate a preorder iteration of nodes in the tree."""
        if not self.is_empty():
            for node in self._subtree_preorder(self._root):  # start recursion
                if self.is_leaf(node):
                    yield node, self.height(node)
    
    def numbers(self):
        """Generate a preorder iteration of nodes in the tree."""
        if not self.is_empty():
            for node in self._subtree_preorder(self._root):  # start recursion
                if self.is_leaf(node) and node._element != "t":
                    yield node._element

    def preorder(self):
        """Generate a preorder iteration of nodes in the tree."""
        if not self.is_empty():
            for node in self._subtree_preorder(self._root):  # start recursion
                yield node

    def _subtree_preorder(self, node):
        """Generate a preorder iteration of nodes in subtree rooted at node."""
        yield node                                           # visit node before its subtrees
        for c in self.children(node):                        # for each child c
            for other in self._subtree_preorder(c):         # do preorder of c's subtree
                yield other                                   # yielding each to our caller

    def postorder(self):
        """Generate a postorder iteration of nodes in the tree."""
        if not self.is_empty():
            for node in self._subtree_postorder(self._root):  # start recursion
                yield node

    def _subtree_postorder(self, node):
        """Generate a postorder iteration of nodes in subtree rooted at node."""
        for c in self.children(node):                        # for each child c
            for other in self._subtree_postorder(c):        # do postorder of c's subtree
                yield other                                   # yielding each to our caller
        yield node                                           # visit node after its subtrees
    def inorder(self):
        """Generate an inorder iteration of positions in the tree."""
        if not self.is_empty():
          for node in self._subtree_inorder(self._root):
            yield node

    def _subtree_inorder(self, node):
        """Generate an inorder iteration of positions in subtree rooted at p."""
        if node._left is not None:          # if left child exists, traverse its subtree
          for other in self._subtree_inorder(node._left):
            yield other
        yield node                               # visit p between its subtrees
        if node._right is not None:         # if right child exists, traverse its subtree
          for other in self._subtree_inorder(node._right):
            yield other

    def root(self):
        """Return the root of the tree (or None if tree is empty)."""
        return self._root

    def parent(self, node):
        """Return node's parent (or None if node is the root)."""
        return node._parent

    def left(self, node):
        """Return node's left child (or None if no left child)."""
        return node._left

    def right(self, node):
        """Return node's right child (or None if no right child)."""
        return node._right

    def children(self, node):
        """Generate an iteration of nodes representing node's children."""
        if node._left is not None:
            yield node._left
        if node._right is not None:
            yield node._right

    def num_children(self, node):
        """Return the number of children of a given node."""
        count = 0
        if node._left is not None:     # left child exists
            count += 1
        if node._right is not None:    # right child exists
            count += 1
        return count

    def sibling(self, node):
        """Return a node representing given node's sibling (or None if no sibling)."""
        parent = node._parent
        if parent is None:                    # p must be the root
            return None                         # root has no sibling
        else:
            if node == parent._left:
                return parent._right         # possibly None
            else:
                return parent._left         # possibly None

    #-------------------------- nonpublic mutators --------------------------
    def add_root(self, e = None):
        """Place element e at the root of an empty tree and return the root node.

        Raise ValueError if tree nonempty.
        """
        if self._root is not None:
            raise ValueError('Root exists')
        self._size = e._height - 2
        self._root = e
        return self._root

    def add_left(self, node, e = None):
        """Create a new left child for a given node, storing element e in the new node.

        Return the new node.
        Raise ValueError if node already has a left child.
        """
        if node._left is not None:
            raise ValueError('Left child exists')
        self._size += 1
        node._left = self.TreeNode(e, node)             # node is its parent
        return node._left

    def add_right(self, node, e = None):
        """Create a new right child for a given node, storing element e in the new node.

        Return the new node.
        Raise ValueError if node already has a right child.
        """
        if node._right is not None:
            raise ValueError('Right child exists')
        self._size += 1
        node._right = self.TreeNode(e, node)            # node is its parent
        return node._right

    def _replace(self, node, e):
        """Replace the element at given node with e, and return the old element."""
        old = node._element
        node._element = e
        return old

    def _delete(self, node):
        """Delete the given node, and replace it with its child, if any.

        Return the element that had been stored at the given node.
        Raise ValueError if node has two children.
        """
        if self.num_children(node) == 2:
            raise ValueError('Position has two children')
        child = node._left if node._left else node._right  # might be None
        if child is not None:
            child._parent = node._parent     # child's grandparent becomes parent
        if node is self._root:
            self._root = child             # child becomes root
        else:
            parent = node._parent
            if node is parent._left:
                parent._left = child
            else:
                parent._right = child
        self._size -= 1
        return node._element



    def _attach(self, node, t1, t2):
        """Attach trees t1 and t2, respectively, as the left and right subtrees of the external node.

        As a side effect, set t1 and t2 to empty.
        Raise TypeError if trees t1 and t2 do not match type of this tree.
        Raise ValueError if node already has a child. (This operation requires a leaf node!)
        """
        if not self.is_leaf(node):
            raise ValueError('position must be leaf')
        if not type(self) is type(t1) is type(t2):    # all 3 trees must be same type
            print(type(self), type(t1), type(t2))
            raise TypeError('Tree types must match')
        self._size += len(t1) + len(t2)
        if not t1.is_empty():         # attached t1 as left subtree of node
            t1._root._parent = node
            node._left = t1._root
            t1._root = None             # set t1 instance to empty
            t1._size = 0
        if not t2.is_empty():         # attached t2 as right subtree of node
            t2._root._parent = node
            node._right = t2._root
            t2._root = None             # set t2 instance to empty
            t2._size = 0

    def Bheight(self, node=None):
        """Return the height of the subtree rooted at a given node.

        If node is None, return the height of the entire tree.
        """
        if node is None:
            return -1
        return self._height2(node)        # start _height2 recursion 
    
    def is_height_balanced(self):
        '''
        @return: True if self BinaryTree is height balanced. False otherwise.
        '''
        return self.node_balance(self._root)
    
    def node_balance(self, node):
        if(node is None): 
            return True
        #print(node._element," : |", self.Bheight(node._left), " - ", self.Bheight(node._right), "| : Diff = ", abs(self.Bheight(node._left) - self.Bheight(node._right)))
        return self.node_balance(node._right) and self.node_balance(node._left) and abs(self.Bheight(node._left) - self.Bheight(node._right)) <= 1
        
    def sum_of_leaves(self):
        '''
        @return: Sum value for all the leaf nodes. You can assume test Tree only contains integers
        '''
        # Task 2
        agg = 0
        for c in self.nodes():
            if self.is_leaf(c):
                agg += c._element
        
        return agg

    def flip_tree(self, node = None):
        '''
        @node: a TreeNode object

        flips the left and right children all nodes in the subtree of given node, 
        and if node parameter is omitted it flips the entire tree.

        @return: Nothing. Modify self
        '''
        #Task 3
        if(node == None):
            node = self._root
        self.flop(node)
        
    def flop(self, node):
        if node == None:
            return
        hodl = node._left
        node._left = node._right
        node._right = hodl
        self.flop(node._left)
        self.flop(node._right)

    def evaluate(self):
        '''
        Evaluates self Expression Tree. You can assume this function is called only on Expression Binary Trees.

        @return: Float result value for evaluating self Tree.
        '''
        # Task 5
        
        return self.helper(self._root)
    def helper(self, node):
        if node == None:
            return
        if self.is_leaf(node):
            return float(node._element)
        else:
            if node._element == "+":
                return self.helper(node._left) + self.helper(node._right)
            if node._element == "*":
                return self.helper(node._left) * self.helper(node._right)
            if node._element == "/":
                return self.helper(node._left) / self.helper(node._right)
            if node._element == "-":
                return self.helper(node._left) - self.helper(node._right)
    
def build_expression_tree(postfix):
    '''
    @postfix: a python string. contains spaces between each operand/operator.

    @return: a class Tree object. This tree should be the Expression Tree for the given postfix string.
    '''
    # Task 4, modify the code below
    opStack = ArrayStack()
    postfix = postfix.split(" ")
    for i in postfix:
        if i not in "+-/*":
            opStack.push(i)
        else:
            babytree = Tree()
            babytree.add_root(i)
            node1 = opStack.pop()
            node2 = opStack.pop()
            if type(node1) == str:
                babybabytree1 = Tree()
                babybabytree1.add_root(node1)
            elif type(node1) != str:
                babybabytree1 = node1
            if type(node2) == str:
                babybabytree2 = Tree()
                babybabytree2.add_root(node2)
            elif type(node2) != str:
                babybabytree2 = node2
            babytree._attach(babytree._root, babybabytree2, babybabytree1)
            opStack.push(babytree)
    return babytree


def pretty_print(tree):
    # ----------------------- Need to enter height to work -----------------
    
    levels = tree.height() + 1  
    print_internal([tree._root], 1, levels)

def print_internal(this_level_nodes, current_level, max_level):
    if (len(this_level_nodes) == 0 or all_elements_are_None(this_level_nodes)):
        return  # Base case of recursion: out of nodes, or only None left

    floor = max_level - current_level;
    endgeLines = 2 ** max(floor - 1, 0);
    firstSpaces = 2 ** floor - 1;
    betweenSpaces = 2 ** (floor + 1) - 1;
    print_spaces(firstSpaces)
    next_level_nodes = []
    for node in this_level_nodes:
        if (node is not None):
            print(node._element, end = "")
            next_level_nodes.append(node._left)
            next_level_nodes.append(node._right)
        else:
            next_level_nodes.append(None)
            next_level_nodes.append(None)
            print_spaces(1)

        print_spaces(betweenSpaces)
    print()
    for i in range(1, endgeLines + 1):
        for j in range(0, len(this_level_nodes)):
            print_spaces(firstSpaces - i)
            if (this_level_nodes[j] == None):
                    print_spaces(endgeLines + endgeLines + i + 1);
                    continue
            if (this_level_nodes[j]._left != None):
                    print("/", end = "")
            else:
                    print_spaces(1)
            print_spaces(i + i - 1)
            if (this_level_nodes[j]._right != None):
                    print("\\", end = "")
            else:
                    print_spaces(1)
            print_spaces(endgeLines + endgeLines - i)
        print()

    print_internal(next_level_nodes, current_level + 1, max_level)

def all_elements_are_None(list_of_nodes):
    for each in list_of_nodes:
        if each is not None:
            return False
    return True

def print_spaces(number):
    for i in range(number):
        print(" ", end = "")

def spaceOut(exp):
    new_exp = []
    skip_next = False
    i = 0
    while i < len(exp):
        if exp[i] == '>':
            new_exp.append(">>")
            i += 2
        elif exp[i] == '<':
            new_exp.append("<<")
            i += 2
        elif exp[i] in "0123456789":
            if new_exp[-1][-1] in "0123456789":
                new_exp[-1] += exp[i]
            else:
                new_exp.append(exp[i])
            i += 1
        else:
            new_exp.append(exp[i])
            i += 1
            
    new_str = new_exp[0]
    for i in range(1,len(new_exp)):
        new_str += " "
        new_str += new_exp[i]
    return new_exp 

def getSubIDX(exp):
    parenth = 1
    i = 0
    while parenth != 0:
        if exp[i] == "(":
            parenth += 1
        elif exp[i] == ")":
            parenth -= 1
        i += 1
    return i

def createTree(expList):
    root = Node('?')
    opStack = []
    varStack = []
    ops = {'*':1,'/':1,'%':1,'+':2,'-':2,'>>':3,'<<':3,'&':4,'^':5,'|':6}
    i = 0
    
    while i < len(expList):
        elem = expList[i]
        if elem in ops:
            opStack.append(elem)
            i += 1
        elif elem == "(":
            subIDX = getSubIDX(expList[i+1:])
            sub = expList[i+1: i+subIDX]
            varStack.append(sub)
            i += 1 + subIDX
        elif elem == ")":
            i += 1
        else:
            varStack.append(elem)
            i+= 1
            
    root._element = opStack.pop()
    
    if type(varStack[-1]) == type([]):
        root._right = createTree(varStack.pop()) 
        root._right._parent = root   
    else:
        root._right = Node(varStack.pop())
        root._right._parent = root
    if type(varStack[-1]) == type([]):
        root._left = createTree(varStack.pop())
        root._left._parent = root
    else:
        root._left = Node(varStack.pop())
        root._left._parent = root
    
    
    if not(root._left is None or root._right is None):
        root._height += max(root._left._height, root._right._height)    
    return root

def makeTree(expression):
    tree = Tree()
    tree.add_root(createTree(spaceOut(expression)))
    return tree

def prettyParse(tree):
    node = tree._root
    return parseOut(tree, node)[1:-1]

def parseOut(tree, node):
    if tree.is_leaf(node):
        return node._element
    else:
        return "(" + parseOut(tree,node._left) + node._element + parseOut(tree,node._right) + ")"
        
def mutate(tree, TRM_flag = 0):
    # Change of Operator, Change of Variable, Constrained Extension, Unconstrained Extension, Trim
    mTypes = ['COO', 'COV','CEXT', 'TRM']
    nodes = list(tree.nodes())
    if len(nodes) < 7:
        TRM_flag = 1
    
    if TRM_flag == 0:
        mProbs = [.3,.4,.2,.1]
    else:
        mProbs = [.3,.3,.4,0]
    #mProbs = [.2,.6,.05,.1,.025,.025]
    mChoice = random.choice(mTypes, p=mProbs)

    new_tree = makeTree(prettyParse(tree))
    
    eval(mChoice)(new_tree, new_tree._root)

    return new_tree
    
def COV(tree, node, parent = None):
    if tree.is_leaf(node):
        if node._element != "t":
            if parent != ">>" and parent != "<<":
                modifier = int(np.round(np.random.normal(loc = 0, scale=2)))
                new_var = int(node._element) + modifier
                node._element = str(new_var)
            else:
                modifier = int(np.round(np.random.normal(loc = 0, scale=2)))
                new_var = abs(int(node._element) + modifier)
                node._element = str(new_var)
        return
    else:
        COV(tree, node._left, node._element)
        COV(tree, node._right, node._element)
        return

def COO(tree, node = None):
    tree_ops = dict(tree.operators())
    height = max(tree_ops.values())
    inverted = [height+1-i for i in tree_ops.values()]
    total = sum(inverted)

    #oVals = ['>>','*','&','^','|','-','+','/']
    #oProbs = [.2,.2,.2,.1 + 2/30,.1,0,2/30,2/30]
    
    oVals = ['>>','*','&','^','%','|','-','+','/']
    oProbs = [.2,.2,.2,.1,2/30,.1,1/30,1/30,2/30]

    nodeProbs = [i / total for i in inverted]
    nodeVals = list(tree_ops.keys())
    
    nodeChoice = random.choice(nodeVals, p=nodeProbs)
    oChoice = random.choice(oVals, p=oProbs)
    
    nodeChoice._element = oChoice
    return

def COP(tree):
    print("COP")

def CEXT(tree, node = None):
    tree_ops = dict(tree.operands())
    height = max(tree_ops.values())
    tree_nums = [int(i) for i in list(tree.numbers())]
    avg_num = mean(tree_nums)
    inverted = [i + 1 for i in tree_ops.values()]
    total = sum(inverted)
    
    #oVals = ['>>','*','&','^','|','-','+','/']
    #oProbs = [.2,.2,.2,.1 + 2/30,.1,0,2/30,2/30]

    oVals = ['>>','*','&','^','%','|','-','+','/']
    oProbs = [.2,.2,.2,.1,2/30,.1,1/30,1/30,2/30]
    

    nodeProbs = [i / total for i in inverted]
    nodeVals = list(tree_ops.keys())
    
    nodeChoice = random.choice(nodeVals, p=nodeProbs)
    oChoice = random.choice(oVals, p=oProbs)
    
    nodeChoice._element = oChoice
    
    nodeChoice._left = Node("t")
    nodeChoice._right = Node(str(int(np.random.normal(loc = avg_num, scale=2))))
    
    return

def TRM(tree, node = None):
    tree_ops = dict(tree.operators())
    tree_nums = [int(i) for i in list(tree.numbers())]
    avg_num = mean(tree_nums)

    nodeVals = [i for i in list(tree_ops.keys()) if tree_ops[i] == 1]
    
    nodeChoice = random.choice(nodeVals)
    
    nodeChoice._element = random.choice([str(int(np.random.normal(loc = avg_num, scale=2))),"t"], p = [.9,.1])
    nodeChoice._left = None
    nodeChoice._right = None
    
    return

def selectOp(tree, max_height = 2, fixed_height = None, node = None):
    tree_ops = dict(tree.operators())
    if fixed_height == None:
        low_ops = [i for i in tree_ops if tree_ops[i] <= max_height]
    else:
        low_ops = [i for i in tree_ops if tree_ops[i] == fixed_height]
        
    height = max([tree_ops[i] for i in low_ops])
    inverted = [height+1-tree_ops[i] for i in low_ops]
    total = sum(inverted)

    nodeProbs = [i / total for i in inverted]
    nodeVals = low_ops
    
    nodeChoice = random.choice(nodeVals, p=nodeProbs)
    return (nodeChoice, tree_ops[nodeChoice])

def cross(tree1, tree2, max_height = None, node1 = None, node2 = None):
    tree1_copy = makeTree(prettyParse(tree1))
    tree2_copy = makeTree(prettyParse(tree2))
    
    if max_height == None:
        donor = selectOp(tree2_copy)
    else:
        donor = selectOp(tree2_copy, max_height = min(max_height, tree2_copy._height1() - 1))
        
    receiver = selectOp(tree1_copy, fixed_height = min(donor[1], tree1_copy._height1() - 1))

    if receiver[0]._parent._left == receiver[0]:
        receiver[0]._parent._left = donor[0]
    if receiver[0]._parent._right == receiver[0]:
        receiver[0]._parent._right = donor[0]
    
    return tree1_copy

def stuborn_mutate(tree, TRM_flag = 0):
    new_tree = mutate(tree,TRM_flag)
    childParsed = prettyParse(new_tree)
    try:
        test_wave = np.array([eval(childParsed) % 256 for t in range(1000,1257)])
        if test_wave.sum() != 0:
            return(childParsed)
        else:
            return stuborn_mutate(tree, 1)
    except:
        return stuborn_mutate(tree, 1)

def mutate_or_kill(tree, kill_duds = True):
    new_tree = mutate(tree)
    childParsed = prettyParse(new_tree)
    try:
        test_wave = np.array([eval(childParsed) % 256 for t in range(1000,1257)])
        if test_wave.sum() != 0:
            return(childParsed)
        elif kill_duds == False:
            return(childParsed)
        return
    except:
        return

def render(children, fs, seconds):
    #must %256 so it doesn't go out of char range
    renderings = []
    dead_children = []
    for child in children:
        try:
            renderings.append([eval(child) % 256 for t in range(fs * seconds)])
        except:
            dead_children.append(child)

    return renderings, dead_children

def extract_features(tree, node = None):
    
    features = {'height': tree._height1(), 
                'leaves': len([i for i in list(tree.operands()) if i[0]._element != 't']), 
                't-count': len([i for i in list(tree.operands()) if i[0]._element == 't']), 
                'operators': len(list(tree.operators())),
                '2_powers': 0,
                'r(>>)':0, 'r(*)':0, 'r(&)':0, 'r(^)':0, 'r(%)':0, 'r(|)':0, 'r(-)':0, 'r(+)':0, 'r(/)':0,
                '>>':0, '*':0, '&':0, '^':0, '%':0, '|':0, '-':0, '+':0, '/':0}
    
    tree_ops = dict(tree.operators())
    nums = [int(i[0]._element) for i in list(tree.operands()) if i[0]._element != 't']
    for op in tree_ops:
        features[op._element] += 1
    
    for num in nums:
        if math.log(num, 2).is_integer():
            features['2_powers'] += 1
    root_str = "r(" + tree._root._element + ")"
    features[root_str] = 1
    return features

def randomTree():
    oVals = ['>>','*','&','^','%','|','-','+','/']
    oProbs = [.2,.2,.2,.1,2/30,.1,1/30,1/30,2/30]
    tree = makeTree("t" + str(random.choice(oVals, p=oProbs)) + "(t" + str(random.choice(oVals, p=oProbs)) + str(random.randint(3,12)) + ")")
    ext_num = int(np.random.normal(loc = 8, scale=2))
    for i in range(ext_num):
        CEXT(tree, tree._root)

    return tree

def create():
    new_tree = randomTree()
    childParsed = prettyParse(new_tree)
    try:
        test_wave = np.array([eval(childParsed) % 256 for t in range(1000,1257)])
        if (test_wave.sum() != 0):
            return(childParsed)
        
        else:
            return stuborn_create()
    except:
        return stuborn_create()
 
def stuborn_create():
    tree = create()
    try:
        test_wave = np.array([eval(tree) % 256 for t in range(25*8000, 26*8000)], dtype = 'f')
        cent = librosa.feature.spectral_centroid(y=test_wave, sr=8000)
        cent_std = np.std(cent)
        tempo, beat_frames = librosa.beat.beat_track(y=test_wave, sr=8000)
        if (cent_std >= 70 and cent_std < 200 and cent.mean() > 400 and tempo > 0):
                return(tree)
        else:
            return stuborn_create()
    except:
        return stuborn_create()

def graph(wave, fs = 16000):
    arrOG = np.array(wave)
    f, t, Sxx = signal.spectrogram(arrOG, fs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.figure(figsize=(10, 3))
    librosa.display.waveplot(np.array(wave, dtype= 'f'), sr = fs)


    
