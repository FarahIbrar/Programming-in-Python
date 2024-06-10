Python Commands Reference
=========================

.. list-table:: 
   :header-rows: 1

   * - Command
     - Definition
     - Example
   * - `print`
     - Outputs a message to the console.
     - ``print("Hello, World!")``
   * - `input`
     - Reads a string from user input.
     - ``name = input("Enter your name: ")``
   * - `int`
     - Converts a value to an integer.
     - ``z = int(input("Enter a number: "))``
   * - `if`
     - Executes a block of code if a condition is true.
     - ``if x > 0: print("Positive")``
   * - `else`
     - Executes a block of code if the preceding `if` condition is false.
     - ``if x > 0: print("Positive") else: print("Non-positive")``
   * - `elif`
     - Checks another condition if the preceding `if` condition is false.
     - ``if x > 0: print("Positive") elif x == 0: print("Zero")``
   * - `and`
     - Returns True if both conditions are true.
     - ``if x > 0 and y > 0: print("Both positive")``
   * - `or`
     - Returns True if at least one condition is true.
     - ``if x > 0 or y > 0: print("At least one positive")``
   * - `not`
     - Inverts the truth value of the condition.
     - ``if not x: print("x is False")``
   * - `type`
     - Returns the type of an object.
     - ``print(type(5))  # Outputs: <class 'int'>``
   * - `for`
     - Iterates over a sequence.
     - ``for i in range(5): print(i)``
   * - `while`
     - Repeats a block of code while a condition is true.
     - ``while x > 0: x -= 1``
   * - `def`
     - Defines a function.
     - ``def greet(): print("Hello")``
   * - `return`
     - Exits a function and returns a value.
     - ``def add(a, b): return a + b``
   * - `import`
     - Imports a module into the script.
     - ``import math``
   * - `from`
     - Imports specific attributes or functions from a module.
     - ``from math import pi``
   * - `as`
     - Provides an alias for a module.
     - ``import numpy as np``
   * - `class`
     - Defines a class.
     - ``class MyClass: pass``
   * - `try`
     - Attempts to execute a block of code.
     - ``try: x = 1/0 except ZeroDivisionError: print("Error")``
   * - `except`
     - Catches and handles exceptions raised by `try`.
     - ``try: x = 1/0 except ZeroDivisionError: print("Error")``
   * - `finally`
     - Executes a block of code regardless of whether an exception occurred.
     - ``try: pass finally: print("Always execute this")``
   * - `with`
     - Simplifies exception handling by encapsulating common preparation and cleanup tasks.
     - ``with open("file.txt") as f: content = f.read()``
   * - `lambda`
     - Creates an anonymous function.
     - ``add = lambda x, y: x + y``
   * - `list`
     - Creates a list.
     - ``numbers = [1, 2, 3, 4, 5]``
   * - `dict`
     - Creates a dictionary.
     - ``ages = {"Alice": 30, "Bob": 25}``
   * - `set`
     - Creates a set.
     - ``unique_numbers = {1, 2, 3, 4, 5}``
   * - `tuple`
     - Creates a tuple.
     - ``coordinates = (10.0, 20.0)``
   * - `str`
     - Converts a value to a string.
     - ``s = str(123)``
   * - `float`
     - Converts a value to a float.
     - ``f = float("3.14")``
   * - `bool`
     - Converts a value to a boolean.
     - ``b = bool(1)``
   * - `range`
     - Generates a sequence of numbers.
     - ``for i in range(5): print(i)``
   * - `len`
     - Returns the length of a sequence.
     - ``length = len([1, 2, 3])``
   * - `open`
     - Opens a file and returns a file object.
     - ``with open("file.txt", "r") as f: content = f.read()``
   * - `append`
     - Adds an item to the end of a list.
     - ``numbers.append(6)``
   * - `pop`
     - Removes and returns an item at a given index.
     - ``numbers.pop(0)``
   * - `split`
     - Splits a string into a list.
     - ``words = "Hello World".split()``
   * - `join`
     - Joins elements of a list into a string.
     - ``sentence = " ".join(words)``
   * - `strip`
     - Removes leading and trailing whitespace from a string.
     - ``clean_str = "   Hello   ".strip()``
   * - `find`
     - Returns the lowest index of a substring in a string.
     - ``index = "Hello".find("e")``
   * - `replace`
     - Replaces occurrences of a substring in a string with another substring.
     - ``new_str = "Hello".replace("e", "a")``
   * - `enumerate`
     - Returns an enumerate object.
     - ``for i, value in enumerate(['a', 'b', 'c']): print(i, value)``
   * - `zip`
     - Combines two or more sequences element-wise.
     - ``zipped = list(zip([1, 2], ['a', 'b']))``
   * - `map`
     - Applies a function to all items in an input list.
     - ``squared = list(map(lambda x: x**2, [1, 2, 3]))``
   * - `filter`
     - Constructs an iterator from elements of an iterable for which a function returns true.
     - ``evens = list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))``
   * - `reduce`
     - Applies a rolling computation to sequential pairs of values in a list.
     - ``from functools import reduce; sum = reduce(lambda x, y: x + y, [1, 2, 3])``
   * - `all`
     - Returns True if all elements of the iterable are true.
     - ``all_true = all([True, True, True])``
   * - `any`
     - Returns True if any element of the iterable is true.
     - ``any_true = any([False, True, False])``
   * - `sort`
     - Returns a new sorted list from the elements of any iterable.
     - ``sorted_list = sorted([3, 1, 2])``
   * - `reversed`
     - Returns a reversed iterator.
     - ``reversed_list = list(reversed([1, 2, 3]))``
   * - `sum`
     - Sums start and the items of an iterable from left to right and returns the total.
     - ``total = sum([1, 2, 3])``
   * - `max`
     - Returns the largest item in an iterable or the largest of two or more arguments.
     - ``maximum = max([1, 2, 3])``
   * - `min`
     - Returns the smallest item in an iterable or the smallest of two or more arguments.
     - ``minimum = min([1, 2, 3])``
   * - `abs`
     - Returns the absolute value of a number.
     - ``absolute = abs(-5)``
   * - `round`
     - Rounds a number to a given precision in decimal digits.
     - ``rounded = round(3.14159, 2)``
   * - `divmod`
     - Takes two numbers and returns a pair of numbers (a tuple) consisting of their quotient and remainder.
     - ``quotient, remainder = divmod(9, 2)``
   * - `isinstance`
     - Returns True if the specified object is of the specified type.
     - ``is_num = isinstance(5, int)``
   * - `issubclass`
     - Returns True if a class is a subclass of another class.
     - ``class A: pass; class B(A): pass; issubclass(B, A)``
   * - `callable`
     - Returns True if the object appears callable.
     - ``callable(print)``
   * - `eval`
     - Parses the expression passed to this method and runs python expression (code) within the program.
     - ``result = eval("1 + 1")``
   * - `exec`
     - Executes the dynamically created program, which is either a string or a code object.
     - ``exec('x = 5')``
   * - `compile`
     - Compiles source into a code or AST object.
     - ``code = compile('a = 5', '<string>', 'exec')``
   * - `globals`
     - Returns the dictionary representing the current global symbol table.
     - ``global_vars = globals()``
   * - `locals`
     - Updates and returns a dictionary representing the current local symbol table.
     - ``local_vars = locals()``
   * - `dir`
     - Attempts to return a list of valid attributes for the object.
     - ``attributes = dir([])``
   * - `help`
     - Invokes the built-in help system.
     - ``help(print)``
   * - `id`
     - Returns the identity of an object.
     - ``obj_id = id([])``
   * - `+`
     - Addition operator
     - ``2 + 3``
   * - `-`
     - Subtraction operator
     - ``5 - 2``
   * - `*`
     - Multiplication operator
     - ``3 * 4``
   * - `/`
     - Division operator
     - ``10 / 2``
   * - `==`
     - Equality comparison operator
     - ``x == y``
   * - `=`
     - Assignment operator
     - ``x = 5``
   * - `equation`
     - Mathematical equation
     - ``x = 2 * (y + 3)``
   * - `.capitalize()`
     - Returns a capitalized version of the string
     - ``"hello".capitalize()``
   * - `.upper()`
     - Converts a string to uppercase
     - ``"hello".upper()``
   * - `.title()`
     - Converts the first character of each word to uppercase
     - ``"hello world".title()``
   * - `.lower()`
     - Converts a string to lowercase
     - ``"HELLO".lower()``
   * - `True`
     - Boolean value representing true
     - ``a = True``
   * - `False`
     - Boolean value representing false
     - ``b = False``
   * - `>=`
     - Greater than or equal to comparison operator
     - ``if x >= y:``
   * - `if/else`
     - Conditional statement
     - ``if condition:``
   * - `int(input("Enter a value:"))`
     - Reads and converts input to an integer
     - ``num = int(input("Enter a number: "))``
   * - `and`
     - Logical operator - and
     - ``if x and y >= z:``
   * - `type(int(input("Enter a value:")))`
     - Reads input, converts to int, and checks type
     - ``type(int(input("Enter a value: ")))``
   * - `try/except`
     - Exception handling
     - ``try:``
   * - `if num is not None:`
     - Checks if variable `num` is not None
     - ``if num is not None:``
