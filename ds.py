import sys
from string import ascii_lowercase, ascii_uppercase

class ListNode:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.previous = None


class TreeNode:
    def __init__(self, value):
        self.value = Value
        self.parent = None
        self.children = []


class BSTNode:
    def __init__(self, value=None, parent=None):
        self.value = value
        self.parent = parent
        self.left = None
        self.right = None


class Queue(object):
    """FIFO. Useful for BFS and caching."""
    def __init__(self, maxsize=0):
        self.length = 0
        self.head = None
        self.tail = None
        self.maxsize = maxsize

    def __repr__(self):
        values = []
        node = self.head

        while node.next is not None:
            values.append(node.value)
            node = node.next
        values.append(node.value)
        return 'values: {}'.format(values)

    def insert(self, value):
        """Insertion always happens on the tail."""
        node = ListNode(value)

        # First insert.
        if self.head == None:
            self.head = self.tail = node

        # Append node to the tail.
        else:
            self.tail.next = node
            node.previous = self.tail
            self.tail = node

        self.length += 1

        # Remove the final element if we exceed the stack size.
        if self.length >= self.maxsize and self.maxsize > 0:
            _ = self.dequeue()

    def pop(self):
        value = self.head.value
        self.head = self.head.next
        self.length -= 1

        if self.length == 0:
            self.last = None

        return value

    def peek(self):
        return self.head.value


class Stack(Queue):
    """LIFO. Useful for backtracking during recursion. DFS."""
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)

    def pop(self):
        item = self.tail.value
        self.tail = self.tail.previous
        self.length -= 1

        if self.length == 0:
            self.last = None

        return item

    def peek(self):
        return self.tail.value


class BinarySearchTree:
    def __init__(self):
        self.root = BSTNode()

    def _insert(self, value, node):

        if not node.value:
            node.value = value

        elif value == node.value:
            raise ValueError('Duplicates in BST are not allowed.')

        elif value < node.value:
            if not node.left:
                node.left = BSTNode(parent=node)
            self._insert(value, node.left)

        elif value > node.value:
            if not node.right:
                node.right = BSTNode(parent=node)
            self._insert(value, node.right)


    def _contains(self, value, node):
        # We are in an empty leaf node.
        if isinstance(node, type(None)):
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._contains(value, node.left)
        elif value > node.value:
            return self._contains(value, node.right)

    def _inorder(self, node, values):
        if not isinstance(node, type(None)):
            values = self._inorder(node.left, values)
            values.append(node.value)
            values = self._inorder(node.right, values)

        return values

    def _preorder(self, node, values):
        if not isinstance(node, type(None)):
            values.append(node.value)
            values = self._inorder(node.left, values)
            values = self._inorder(node.right, values)

        return values

    def _postorder(self, node, values):
        if not isinstance(node, type(None)):
            values = self._inorder(node.left, values)
            values = self._inorder(node.right, values)
            values.append(node.value)

        return values

    def inorder(self):
        return self._inorder(self.root, [])

    def preorder(self):
        return self._preorder(self.root, [])

    def postorder(self):
        return self._postorder(self.root, [])

    def insert(self, value):
        return self._insert(value, self.root)

    def contains(self, value):
        return self._contains(value, self.root)


class Heap:
    def __init__(self, mode='min', n=10):
        assert mode in ['min', 'max']
        self.mode = mode
        self.heap = [0] * (n+1)
        self.first_insert_flag = True
        self.end = 0  # end index for the heap.
        self.heap[0] = sys.maxsize*-1  # Initalize.
        
    def parent(self, idx):
        return 0 if idx == 0 else (idx-1) // 2
        
    def left(self, idx):
        return (2*idx) + 1
    
    def right(self, idx):
        return (2*idx) + 2
    
    def is_leaf(self, idx):
        return True if idx >= len(self.heap) // 2 else False  

    def swap(self, idx_a, idx_b):
        self.heap[idx_a], self.heap[idx_b] = self.heap[idx_b], self.heap[idx_a]
    
    def percolate(self, idx): 
        """Percolates nodes down."""
        # If the node is a non-leaf node and smaller than all children.
        if not self.is_leaf(idx):
            
            # If this node is smaller than either children.
            if (self.heap[idx] < self.heap[self.left(idx)] or
                self.heap[idx] < self.heap[self.right(idx)]):
  
                # Swap with the left child and percolate down.
                if self.heap[self.left(idx)] > self.heap[self.right(idx)]: 
                    self.swap(idx, self.left(idx)) 
                    self.percolate(self.left(idx)) 
  
                # Swap with the right child and percolate down.
                else: 
                    self.swap(idx, self.right(idx)) 
                    self.percolate(self.right(idx)) 

    def insert(self, value):
        if self.mode == 'min':
            value *= -1
        self.end += 1
        self.heap[self.end] = value
        curr_idx = self.end

        while self.heap[curr_idx] > self.heap[self.parent(curr_idx)]:
            self.swap(curr_idx, self.parent(curr_idx))
            curr_idx = self.parent(curr_idx)
        
        # Removes the dummy head node.
        if self.first_insert_flag:
            self.first_insert_flag = False
            self.heap[self.end] = 0
            self.end -= 1

    def pop(self):
        value = self.root()
        self.heap[0] = self.heap[self.end]
        self.end -= 1
        self.percolate(0)
        
        return value
    
    def root(self):
        value = self.heap[0]
        if self.mode == 'min':
            value *= -1

        return value
        
    def display(self):
        for i in range(0, (len(self.heap)//2)):
            print('idx  {} < {}, {} : heap {} < {}, {} '.format(
                i, self.left(i), self.right(i),
                self.heap[i], 
                self.heap[self.left(i)], 
                self.heap[self.right(i)]
            ))


class HashTable:
    """Hash Table with Seperate Chaining."""
    def __init__(self, n=91):
        self.n = n
        self.table = [None] * n
        self.p = 31
        self.mapper = {}
        for i, letter in enumerate(list(ascii_lowercase + ascii_uppercase)):
            self.mapper[letter] = i+1

    def _hash_string(self, string):
        assert isinstance(string, str)
        hash_val = 0
        for i, char in enumerate(list(string)):
            hash_val += self.mapper[char] * self.p**i

        return hash_val % self.n

    def insert(self, key, value):
        idx = self._hash_string(key)
        if not self.table[idx]: 
            self.table[idx] = [[key, value]]
        elif isinstance(self.table[idx], list):
            self.table[idx].append([key, value])

    def get(self, key):
        idx = self._hash_string(key)
        for elem in self.table[idx]:
            if key == elem[0]:
                return elem[1]


if __name__ == "__main__":

    def test_queue():
        queue = Queue()
        for value in [10, 7, 6]:
            queue.insert(value)
        assert queue.peek() == 10
        assert queue.pop() == 10  # FIFO

    def test_stack():
        stack = Stack()
        for value in [10, 7, 6]:
            stack.insert(value)
        assert stack.peek() == 6
        assert stack.pop() == 6  # FILO

    def test_bst():
        bst = BinarySearchTree()
        for value in [10, 4, 7, 20, 100]:
            bst.insert(value)
        assert bst.contains(20)
        assert not bst.contains(21)
        assert bst.inorder() == [4, 7, 10, 20, 100]
        assert bst.preorder() == [10, 4, 7, 20, 100]
        assert bst.postorder() == [4, 7, 20, 100, 10]

    def test_heap():
        heap = Heap(mode='min')
        for value in [10, 50, 30, 22, 587]:
            heap.insert(value)
        assert heap.root() == 10
        heap.pop()
        assert heap.root() == 22

        heap = Heap(mode='max')
        for value in [10, 50, 30, 22, 587]:
            heap.insert(value)
        assert heap.root() == 587
        heap.pop()
        assert heap.root() == 50

    def test_hash():
        table = HashTable(n=7)
        for k, v in zip(['bib', 'billy', 'franklin'], [490, 123, 15590]):
            table.insert(k, v)
        assert table.get('bib') == 490
        print(table.table)


    test_queue()
    test_stack()
    test_bst()
    test_heap()
    test_hash()