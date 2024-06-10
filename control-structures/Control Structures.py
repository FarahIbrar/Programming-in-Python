#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:beige;color:beige">
# <header>
# <br><br><br><br>
# <h1 style="padding:1em;text-align:center;color:#00008B">Basics of Python programming <br><br> &nbsp;&nbsp; Control Structure </h1> 
# </header>
# <br><br><br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar</td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# <header>
# <h1 style="padding:1em;text-align:left;color:#00008B">Control Structure</h1>
# </header>
# 
#     
# <ul><li><span style="color:#00008B;font-size:24px">Control flow of execution of a series of expressions. </span> <br><br>
#         <li><span style="color:#00008B; font-size:24px">"Logic” into your code. </span> <br><br>
#         <li><span style="color:#00008B; font-size:24px">Respond to inputs/features of the data and execute different expressions accordingly. </span><br><br>
#         </li>
#        </ul>
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# <header>
# <h1 style="padding:1em;text-align:left;color:#00008B">Control Structure</h1>
# </header>
# 
#     
# <ul><li><span style="color:#00008B;font-size:24px">A control structure in programming refers to the way in which the execution flow of a program is managed or controlled. <br><br>It determines the order in which individual statements, instructions, or function calls are executed within a program. <br><br>Control structures are fundamental to organizing the logic and flow of a program to achieve specific tasks. </span> <br><br>
# <br><br>
#         </li>
#        </ul>
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# <header>
# <h1 style="padding:1em;text-align:left;color:#00008B">Control Flow</h1>
# </header>
# 
#     
# <ul><li><span style="color:#00008B;font-size:24px">Control flow refers to the order in which the individual statements, instructions, or function calls of a program are executed. <br><br>Control flow structures dictate how the program moves through its code, determining which statement is executed next based on certain conditions. <br><br>It determines the order in which individual statements, instructions, or function calls are executed within a program.
#     
# </span>
#         </li>
#        </ul>
# <h1 style="padding:1em;text-align:left;color:#00008B">Logic</h1>    
# <ul><li><span style="color:#00008B;font-size:24px">Control flow refers to the order in which the individual statements, instructions, or function calls of a program are executed. <br><br>Control flow structures dictate how the program moves through its code, determining which statement is executed next based on certain conditions. <br><br>It determines the order in which individual statements, instructions, or function calls are executed within a program.
#     
# </span> <br>
#         </li>
#        </ul>
#        
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# <header>
# <h1 style="padding:1em;text-align:left;color:#00008B">Responding to Inputs/Features</h1>
# </header>
# 
#     
# <ul><li><span style="color:#00008B;font-size:24px"> In programming, your code often needs to react differently based on the input it receives or the features of the data it processes. <br><br>This involves using conditional statements to check certain conditions and executing different blocks of code based on the evaluation of those conditions.
# </span>
#         </li>
#        </ul>
# <h1 style="padding:1em;text-align:left;color:#00008B">Executing Different Expressions</h1>    
# <ul><li><span style="color:#00008B;font-size:24px">Expressions in programming are snippets of code that produce a value. Different expressions can be executed based on the control flow of the program. <br><br>For example, you might have different calculations, operations, or function calls that need to be executed based on the input data or specific conditions within the program. <br><br>
# </span> <br>
#         </li>
#        </ul>
#        
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# In[1]:


weather_forecast = "rainy"

if weather_forecast == "sunny":
    print("Take sunglasses.")
elif weather_forecast == "rainy":
    print("Take an umbrella.")
else:
    print("Check the weather again later.")


# <div style="background-color:beige;color:beige">
# <header>
# <h2 style="padding:1em;text-align:left;color:#00008B">Broadly, two types:</h2>
# </header>
# 
#     
# <ul><li><span style="color:#00008B;font-size:24px">Selection - used for decision. [example - if, if-else, if-elif, else] </span> <br><br>
#         <li><span style="color:#00008B; font-size:24px">Repetition – used for looping. [example - for] </span> <br><br>
#         </li>
#        </ul>
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar</td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# 
#  <h2 style="padding:1em;text-align:left;color:#00008B">Control Structure : <b>for </b> </h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Commonly used looping structure. <br><br>
# Used for iterating over the elements of an object (list, string, etc.)<br><br>
# Define an iterator (e.g., i) and assign an iterable i.e., an object used for iteration. <br><br>
# <b>Syntax:</b> for iterator in iterable:<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     
#     Statement 1<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
#     Statement n 
# <br>
# </span></li></ul> <br><br>
#  
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar</td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# **Control Structure: `for`**
# 
# - **Commonly used looping structure**: The `for` loop is used when you want to repeat a block of code a certain number of times or over a sequence of items. It's one of the most common ways to perform iteration in programming.
# 
# - **Iterating over elements**: It's used to go through each item in a list, string, or other collections of data (called iterables). For example, you might want to process each character in a string or each element in a list one by one.
# 
# - **Syntax**: The syntax of a `for` loop is straightforward:
# 
# ```python
# for iterator in iterable:
#     # Statement 1
#     # Statement 2
#     # ...
#     # Statement n
# ```
# 
# Here’s what each part means:
# 
# - `iterator`: This is a variable name that you choose to represent each element of the `iterable` (the collection of items you're iterating over). It will take on each value of the iterable in turn.
#   
# - `in`: This keyword is used to link the iterator variable to the iterable object.
# 
# - `iterable`: This is the object (like a list, string, or range) that contains the elements you want to loop through.
# 
# - `Statement 1` to `Statement n`: These are the statements (lines of code) that you want to repeat for each element in the iterable.
# 
# **Example:**
# 
# Let's look at an example to iterate over a list of names and print each name:
# 
# ```python
# names = ["Alice", "Bob", "Charlie", "David"]
# 
# for name in names:
#     print("Hello, " + name)
# ```
# 
# - Here, `name` is the iterator variable that takes each name from the `names` list in turn.
# - The `print("Hello, " + name)` statement is executed for each name in the list, resulting in:
#   
#   ```
#   Hello, Alice
#   Hello, Bob
#   Hello, Charlie
#   Hello, David
#   ```
# 
# **Summary:**
# 
# The `for` loop is used to execute a block of code multiple times, once for each item in a sequence or collection. It simplifies iterating over elements like lists or strings and allows you to perform operations on each item in the sequence.

# In[2]:


# Creating a list
list1 = [1, 2, 3, 4]  

# Using for loop to print all elements in the list
for i in list1:
    print(i)


# In[3]:


# Creating a list having a mix of numeric and string variables
list2 = [1, 2.5, 'Nadeer']

# Using for loop to print all elements in the list
for i in list2:
    print(i)


# In[4]:


# Creating a tuple
tuple1 = (1, 2, 3)

# Using for loop to print all elements in the tuple
for x in tuple1:
    print(x)


# In[5]:


# Creating a dictionary
dict1 = {1:"Farah", 2:"Python", 3:"Anaconda"}

# Using for loop to print key/values/items in the dictionary
for d in dict1:
    print(d)
    
for d in dict1.values():
    print(d)
    
for d in dict1.items():
    print(d)


# ### The Dictionary
# First, we have a dictionary called `dict1`:
# ```python
# dict1 = {1: "Farah", 2: "Python", 3: "Anaconda"}
# ```
# This dictionary has keys (`1`, `2`, `3`) and corresponding values (`"Farah"`, `"Python"`, `"Anaconda"`).
# 
# ### First `for` Loop: Iterating Over Keys
# ```python
# for d in dict1:
#     print(d)
# ```
# - `for d in dict1:`: This loop iterates over the keys of the dictionary.
# - `d` will take each key from the dictionary one by one.
# - `print(d)`: This prints each key.
# 
# Output:
# ```
# 1
# 2
# 3
# ```
# 
# ### Second `for` Loop: Iterating Over Values
# ```python
# for d in dict1.values():
#     print(d)
# ```
# - `for d in dict1.values():`: This loop iterates over the values of the dictionary.
# - `d` will take each value from the dictionary one by one.
# - `print(d)`: This prints each value.
# 
# Output:
# ```
# Farah
# Python
# Anaconda
# ```
# 
# ### Third `for` Loop: Iterating Over Items
# ```python
# for d in dict1.items():
#     print(d)
# ```
# - `for d in dict1.items():`: This loop iterates over the key-value pairs (items) of the dictionary.
# - `d` will take each item (which is a tuple containing a key and its corresponding value) from the dictionary one by one.
# - `print(d)`: This prints each item (tuple).
# 
# Output:
# ```
# (1, 'Farah')
# (2, 'Python')
# (3, 'Anaconda')
# ```
# 
# ### Summary
# - The first loop prints the keys of the dictionary.
# - The second loop prints the values of the dictionary.
# - The third loop prints the key-value pairs (items) of the dictionary.

# In[6]:


# Creating a set
set1 = set([1, 2, 3, 4])

# Using for loop multiply all the elements of the set by 2 and print them
for s in set1:
    z = s*2
    print(z)


# <div style="background-color:beige;color:beige">
# 
#  <h2 style="padding:1em;text-align:left;color:#00008B">Control Structure : <b>for </b> </h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Commonly used looping structure. <br><br>
# Used for iterating over the elements of an object (list, string, etc.)<br><br>
# Define an iterator value (e.g., i) and assign an iterable i.e., an object used for iteration. <br><br>
# <b>Syntax:</b> for iterator in iterable:<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     
#     Statement 1<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
#     Statement n 
# <br><br>
# Iterate over a sequence of numbers –  <b>range() </b>.
# </span></li></ul> <br><br>
#  
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar</td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# In[7]:


#prints 0 to 9
for i in range(10):
    print(i)


# In[8]:


# define the limits in range
# for iterator range (intial_value, end_condition)
# Remeber while iterating, it will process the value end_condition-1, like in this example it will print values 1, 2, 3 and 4

for i in range(1, 5):
    print(i)
    
    
# for iterator range (intial_value, end_condition, incremental_value)
# incremental_value - performs an action after each iteration
for i in range(1, 5, 2):
    print(i)


# ^ Explain the `for` loops using `range`:
# 
# ### First `for` Loop: Basic Range
# ```python
# for i in range(1, 5):
#     print(i)
# ```
# - `range(1, 5)`: Generates numbers from 1 to 4 (stops before 5).
# - `for i in range(1, 5)`: `i` takes each value from 1 to 4 in sequence.
# - `print(i)`: Prints each value of `i`.
# 
# Output:
# ```
# 1
# 2
# 3
# 4
# ```
# 
# ### Second `for` Loop: Range with Step
# ```python
# for i in range(1, 5, 2):
#     print(i)
# ```
# - `range(1, 5, 2)`: Generates numbers from 1 to 4, stepping by 2.
# - `for i in range(1, 5, 2)`: `i` takes values 1, then 3 (increments by 2 each time).
# - `print(i)`: Prints each value of `i`.
# 
# Output:
# ```
# 1
# 3
# ```
# 
# ### Summary
# - `range(start, end)`: Generates numbers from `start` to `end-1`.
# - `range(start, end, step)`: Generates numbers from `start` to `end-1`, incrementing by `step`.
# - The loop prints each generated number.

# In[12]:


# Ask user to enter a value, and calculate its factorial value 
z = int(input("Enter a value:"))
factorial = 1

for i in range(1, z+1):
    factorial = i*factorial
    print(factorial)


# <div style="background-color:beige;color:beige">
# 
#  <h2 style="padding:1em;text-align:left;color:#00008B">Control Structure : <b>for </b> </h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Commonly used looping structure. <br><br>
# Used for iterating over the elements of an object (list, string, etc.)<br><br>
# Define an iterator value (e.g., i) and assign an iterable i.e., an object used for iteration. <br><br>
# <b>Syntax:</b> for iterator in iterable:<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     
#     Statement 1<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
#     Statement n 
# <br><br>
# Iterate over a sequence of numbers – range(). <br><br>
# Iterate over the indices of a sequence - range() and <b>len()</b>.
# </span></li></ul> <br><br>
#  
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# In[13]:


# Creating a list having a mix of numeric and string variables
list2 =  [1, 2.5, 'Umar']

# Using for loop with len function to print all elements with their index position
for i in range(len(list2)):
    print(i, list2[i])


# <div style="background-color:beige;color:beige">
# 
#  <h2 style="padding:1em;text-align:left;color:#00008B">Control Structure : <b>for </b> </h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Commonly used looping structure. <br><br>
# Used for iterating over the elements of an object (list, string, etc.)<br><br>
# Define an iterator value (e.g., i) and assign an iterable i.e., an object used for iteration. <br><br>
# <b>Syntax:</b> for iterator in iterable:<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     
#     Statement 1<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
#     Statement n 
# <br><br>
# Iterate over a sequence of numbers –  range().<br><br>
# Iterate over the indices of a sequence - range() and len(). <br><br>
# <b>Break</b> (stop and exit the loop) and <b>continue</b> (stops current iteration) statement.
# 
# </span></li></ul> <br><br>
#  
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# In[14]:


# Example using break in a for loop
for i in range(1, 15, 2):
    if i == 7:
        break
    print(i)


# ^
# ### `for` Loop with `break`
# ```python
# for i in range(1, 15, 2):
#     if i == 7:
#         break
#     print(i)
# ```
# 
# ### Explanation:
# 
# - `range(1, 15, 2)`: Generates numbers starting from 1 up to but not including 15, with a step of 2. This gives the sequence: 1, 3, 5, 7, 9, 11, 13.
# - `for i in range(1, 15, 2)`: The loop iterates over each number in the generated sequence.
# - `if i == 7: break`: When `i` equals 7, the `break` statement is executed, which exits the loop immediately.
# - `print(i)`: Prints the value of `i` in each iteration, unless the loop is broken.
# 
# ### Output:
# ```
# 1
# 3
# 5
# ```
# 
# ### Summary:
# The loop prints numbers 1, 3, and 5. When `i` becomes 7, the `break` statement stops the loop, so 7 and any numbers after it are not printed.

# In[15]:


# Example using continue in a for loop
for i in range(1, 15, 2):
    if i == 7:
        continue
    print(i)


# ^
# 
# ### `for` Loop with `continue`
# ```python
# for i in range(1, 15, 2):
#     if i == 7:
#         continue
#     print(i)
# ```
# 
# ### Explanation:
# 
# - `range(1, 15, 2)`: Generates numbers starting from 1 up to but not including 15, with a step of 2. This gives the sequence: 1, 3, 5, 7, 9, 11, 13.
# - `for i in range(1, 15, 2)`: The loop iterates over each number in the generated sequence.
# - `if i == 7: continue`: When `i` equals 7, the `continue` statement is executed. This skips the rest of the loop body for the current iteration and moves to the next iteration.
# - `print(i)`: Prints the value of `i` in each iteration, unless the loop is skipped by `continue`.
# 
# ### Output:
# ```
# 1
# 3
# 5
# 9
# 11
# 13
# ```
# 
# ### Summary:
# The loop prints numbers 1, 3, 5, 9, 11, and 13. When `i` becomes 7, the `continue` statement skips the `print(i)` statement for that iteration, so 7 is not printed. The loop then continues with the next numbers in the sequence.

# <div style="background-color:beige;color:beige">
# 
#  <h2 style="padding:1em;text-align:left;color:#00008B">Control Structure : <b>for </b> </h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Commonly used looping structure. <br><br>
# Used for iterating over the elements of an object (list, string, etc.)<br><br>
# Define an iterator value (e.g., i) and assign an iterable i.e., an object used for iteration. <br><br>
# <b>Syntax:</b> for iterator in iterable:<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     
#     Statement 1<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
#     Statement n 
# <br><br>
# Iterate over a sequence of numbers –  <b>range() </b>.<br><br>
# Iterate over the indices of a sequence - range() and <b>len()</b>. <br><br>
# Break (stop and exit the loop) and continue (stops current iteration) statement. <br><br>
# <b> Else </b> in a for loop.
# 
# </span></li></ul> <br><br>
#  
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar</td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# In[16]:


# Example using else with a for loop
for i in range(1, 15, 2):
    print(i)
else:
    print("All the value have been printed.")


# In[17]:


# Else clause will not be executed if there is break clause, but it will with continue
z = 7
for i in range(1, 15, 2):
    if i == z:
        break
    print(i)
else:
    print("Exiting as z value in the loop.")


# <div style="background-color:beige;color:beige">
# 
#  <h2 style="padding:1em;text-align:left;color:#00008B">Control Structure : <b>for </b> </h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Commonly used looping structure. <br><br>
# Used for iterating over the elements of an object (list, string, etc.)<br><br>
# Define an iterator value (e.g., i) and assign an iterable i.e., an object used for iteration. <br><br>
# <b>Syntax:</b> for iterator in iterable:<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     
#     Statement 1<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
#     Statement n 
# <br><br>
# Iterate over a sequence of numbers –  range().<br><br>
# Iterate over the indices of a sequence - range() and len(). <br><br>
# Break (stop and exit the loop) and continue (stops current iteration) statement. <br><br>
# Else in a for loop. <br><br>
# <b>Nested</b> loop.
# 
# </span></li></ul> <br><br>
#  
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# In[18]:


list1 = [1, 2, 3, 4] 
list3 = ['Dipankar', 'Python', 'Notebook']

for i in list1:
    for j in list3:
        print(i, j)


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# 1. Write a program in python, which asks user to enter two numerical values. Return all the even numbers in between these two user-specified values. <br><br>
#     
# 2. WAP in Python to classify the given set of proteins into two groups. First group should have proteins whose M.W. is greater than or equal to Protein X (M.W. 450 Kda) and for second group lesser than Protein X.
# 
# ![image.png](attachment:image.png) <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">Control Structure</td></tr></table>
# </footer>
# 
# </div>

# In[19]:


# Ask the user to enter two numerical values
start = int(input("Enter the first number: "))
end = int(input("Enter the second number: "))

# Ensure start is less than end
if start > end:
    start, end = end, start

# Initialize an empty list to store the even numbers
even_numbers = []

# Iterate over the range and collect even numbers
for i in range(start + 1, end):
    if i % 2 == 0:
        even_numbers.append(i)

# Print the list of even numbers
print("Even numbers between", start, "and", end, "are:", even_numbers)


# In[21]:


# Define the molecular weight of Protein X
protein_x_mw = 450

# Function to classify proteins
def classify_proteins(proteins):
    greater_equal_group = []
    lesser_group = []
    
    for protein, mw in proteins.items():
        if mw >= protein_x_mw:
            greater_equal_group.append(protein)
        else:
            lesser_group.append(protein)
    
    return greater_equal_group, lesser_group

# Example set of proteins with their molecular weights
proteins = {
    'Protein 1': 180,
    'Protein 2': 345,
    'Protein 3': 568,
    'Protein 4': 765,
    'Protein 5': 564,
    'Protein 6':120,
    'Protein 7':1350,
    'Protein 8':245,
    'Protein 9':865,
    'Protein 10':514
}

# Classify the proteins
greater_equal_group, lesser_group = classify_proteins(proteins)

# Print the results
print("Proteins with M.W. >= 450 KDa:")
print(greater_equal_group)

print("\nProteins with M.W. < 450 KDa:")
print(lesser_group)

