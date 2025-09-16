# Python Interview Questions with Solutions

## BASIC LEVEL QUESTIONS & SOLUTIONS

### 1. String Manipulation

#### Reverse a String
```python
# Method 1: Slicing
def reverse_string(s):
    return s[::-1]

# Method 2: Using loop
def reverse_string_loop(s):
    result = ""
    for char in s:
        result = char + result
    return result

# Method 3: Using join
def reverse_string_join(s):
    return ''.join(reversed(s))
```

#### Check if String is Palindrome
```python
def is_palindrome(s):
    # Remove spaces and convert to lowercase
    s = ''.join(s.split()).lower()
    return s == s[::-1]

# More efficient - two pointers
def is_palindrome_optimized(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

#### Count Vowels and Consonants
```python
def count_vowels_consonants(s):
    vowels = "aeiouAEIOU"
    v_count = c_count = 0
    
    for char in s:
        if char.isalpha():
            if char in vowels:
                v_count += 1
            else:
                c_count += 1
    
    return v_count, c_count
```

#### Check if Two Strings are Anagrams
```python
def are_anagrams(s1, s2):
    return sorted(s1.lower()) == sorted(s2.lower())

# Using Counter
from collections import Counter
def are_anagrams_counter(s1, s2):
    return Counter(s1.lower()) == Counter(s2.lower())
```

### 2. Lists and Arrays

#### Find Maximum and Minimum
```python
def find_max_min(arr):
    if not arr:
        return None, None
    return max(arr), min(arr)

# Without built-in functions
def find_max_min_manual(arr):
    if not arr:
        return None, None
    
    max_val = min_val = arr[0]
    for num in arr[1:]:
        if num > max_val:
            max_val = num
        if num < min_val:
            min_val = num
    
    return max_val, min_val
```

#### Remove Duplicates from List
```python
def remove_duplicates(arr):
    return list(set(arr))  # Order not preserved

def remove_duplicates_preserve_order(arr):
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
```

#### Find Second Largest Number
```python
def second_largest(arr):
    if len(arr) < 2:
        return None
    
    first = second = float('-inf')
    for num in arr:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num
    
    return second if second != float('-inf') else None
```

#### Two Sum Problem
```python
def two_sum(nums, target):
    """Find two numbers that add up to target"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### 3. Basic Math Problems

#### Check if Number is Prime
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

#### Fibonacci Sequence
```python
# Recursive
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# Iterative
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Generate sequence
def fibonacci_sequence(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib
```

#### Factorial
```python
def factorial(n):
    if n < 0:
        return None
    if n <= 1:
        return 1
    return n * factorial(n-1)

# Iterative
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

## MEDIUM LEVEL QUESTIONS & SOLUTIONS

### 1. Array Problems

#### Maximum Subarray Sum (Kadane's Algorithm)
```python
def max_subarray_sum(nums):
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

#### Three Sum Problem
```python
def three_sum(nums):
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return result
```

#### Rotate Array
```python
def rotate_array(nums, k):
    """Rotate array to the right by k steps"""
    n = len(nums)
    k = k % n
    
    # Method 1: Extra space
    return nums[-k:] + nums[:-k]

def rotate_array_inplace(nums, k):
    """Rotate in-place"""
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    
    n = len(nums)
    k = k % n
    
    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
```

### 2. Linked List Problems

#### Linked List Node Definition
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

#### Reverse Linked List
```python
def reverse_linked_list(head):
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev
```

#### Detect Cycle in Linked List
```python
def has_cycle(head):
    if not head or not head.next:
        return False
    
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False
```

#### Merge Two Sorted Lists
```python
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 or l2
    return dummy.next
```

### 3. Tree Problems

#### Binary Tree Node Definition
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

#### Binary Tree Traversals
```python
def inorder_traversal(root):
    """Left -> Root -> Right"""
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result

def preorder_traversal(root):
    """Root -> Left -> Right"""
    result = []
    
    def preorder(node):
        if node:
            result.append(node.val)
            preorder(node.left)
            preorder(node.right)
    
    preorder(root)
    return result

def postorder_traversal(root):
    """Left -> Right -> Root"""
    result = []
    
    def postorder(node):
        if node:
            postorder(node.left)
            postorder(node.right)
            result.append(node.val)
    
    postorder(root)
    return result
```

#### Level Order Traversal
```python
from collections import deque

def level_order_traversal(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

#### Maximum Depth of Binary Tree
```python
def max_depth(root):
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)
```

### 4. Dynamic Programming

#### Climbing Stairs
```python
def climb_stairs(n):
    """Ways to climb n stairs (1 or 2 steps at a time)"""
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Space optimized
def climb_stairs_optimized(n):
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1
```

#### Coin Change Problem
```python
def coin_change(coins, amount):
    """Minimum coins needed to make amount"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

## ADVANCED LEVEL QUESTIONS & SOLUTIONS

### 1. Advanced String Algorithms

#### Longest Substring Without Repeating Characters
```python
def longest_substring_without_repeating(s):
    char_map = {}
    left = max_length = 0
    
    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1
        
        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

#### Longest Palindromic Substring
```python
def longest_palindromic_substring(s):
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    start = end = 0
    
    for i in range(len(s)):
        len1 = expand_around_center(i, i)  # Odd length
        len2 = expand_around_center(i, i + 1)  # Even length
        max_len = max(len1, len2)
        
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end + 1]
```

### 2. Graph Algorithms

#### DFS and BFS
```python
from collections import deque, defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # For undirected graph
    
    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        
        visited.add(start)
        print(start, end=' ')
        
        for neighbor in self.graph[start]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)
    
    def bfs(self, start):
        visited = set([start])
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            print(node, end=' ')
            
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
```

#### Number of Islands
```python
def num_islands(grid):
    if not grid:
        return 0
    
    def dfs(i, j):
        if (i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or 
            grid[i][j] != '1'):
            return
        
        grid[i][j] = '0'  # Mark as visited
        
        # Check all 4 directions
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)
    
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    
    return count
```

### 3. Advanced Dynamic Programming

#### Longest Common Subsequence
```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

#### Edit Distance
```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    
    return dp[m][n]
```

### 4. System Design Problems

#### LRU Cache
```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)

# More efficient implementation using doubly linked list
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCacheOptimized:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        
        # Create dummy head and tail
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove an existing node"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node):
        """Move node to head"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Pop the last node"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key):
        node = self.cache.get(key)
        if node:
            self._move_to_head(node)
            return node.val
        return -1
    
    def put(self, key, value):
        node = self.cache.get(key)
        
        if node:
            node.val = value
            self._move_to_head(node)
        else:
            new_node = Node(key, value)
            
            if len(self.cache) >= self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            self.cache[key] = new_node
            self._add_node(new_node)
```

## COMPLEXITY ANALYSIS GUIDE

### Time Complexity Common Cases:
- **O(1)**: Hash table operations, array access
- **O(log n)**: Binary search, heap operations
- **O(n)**: Linear search, single loop
- **O(n log n)**: Efficient sorting algorithms
- **O(n²)**: Nested loops, bubble sort
- **O(2^n)**: Recursive algorithms with branching

### Space Complexity:
- **O(1)**: Constant extra space
- **O(n)**: Extra array/list of size n
- **O(h)**: Recursion depth (h = height of tree)

## INTERVIEW TIPS

### Before Coding:
1. **Understand the problem** - Ask clarifying questions
2. **Think of edge cases** - Empty input, single element, etc.
3. **Discuss approach** - Start with brute force, then optimize
4. **Estimate complexity** - Time and space

### While Coding:
1. **Write clean code** - Good variable names, proper indentation
2. **Think out loud** - Explain your thought process
3. **Test with examples** - Walk through your code
4. **Handle edge cases** - Check for null/empty inputs

### After Coding:
1. **Trace through examples** - Make sure it works
2. **Discuss optimizations** - Can you do better?
3. **Analyze complexity** - Time and space complexity
4. **Consider variations** - What if requirements change?

## PYTHON-SPECIFIC INTERVIEW CONCEPTS

### Important Built-ins and Libraries:
```python
# Collections
from collections import deque, Counter, defaultdict, OrderedDict

# Heapq for priority queues
import heapq

# Bisect for binary search
import bisect

# Itertools for combinations/permutations
import itertools

# Math functions
import math
```

### Python Idioms:
```python
# List comprehensions
squares = [x**2 for x in range(10) if x % 2 == 0]

# Dictionary comprehensions
word_count = {word: len(word) for word in words}

# Enumerate for index and value
for i, value in enumerate(arr):
    print(f"Index {i}: {value}")

# Zip for parallel iteration
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Using * for unpacking
def func(a, b, c):
    return a + b + c

args = [1, 2, 3]
result = func(*args)
```

This comprehensive guide covers the most important Python interview questions with detailed solutions. Practice these regularly and you'll be well-prepared for your interviews!
