#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:beige;color:beige">
# <header>
# <h1 style="padding:1em;text-align:center;color:#00008B">Basics of Python programming <br><br> &nbsp;&nbsp; Data Structure </h1> 
# </header>
# <br><br><br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# <header>
# <h1 style="padding:1em;text-align:left;color:#00008B">Outline</h1>
# </header>
# 
#     
# <ul><li><span style="color:#00008B;font-size:24px">List </span> <br><br>
#         <li><span style="color:#00008B; font-size:24px">Tuple </span> <br><br>
#         <li><span style="color:#00008B; font-size:24px">Dictionary </span><br><br>
#         <li><span style="color:#00008B; font-size:24px">Sets </span> 
#         </li>
#        </ul>
# 
# <br><br><span style="color:#00008B;font-size:20px">
#  <b>*Remember</b> - In Python, anything which can be modified after creation is mutable, while which cannot is immutable.</span><br>
# <span style="color:#00008B;font-size:16px">
# Example: Lists, Sets, Dictionaries are mutable.  Tuples and basic data types (numeric, boolean, strings) are non-mutable.
#     </span>
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# #### List, Tuple, Dictionary, Sets are all mutable. 

# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">List</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Stores heterogenous data items (mixed of data types) - numeric (numbers), string (within quotes), or other data structures like lists themselves. <br><br>
# They are mutable, i.e., elements can be changed after creation. This means you can modify, add, or remove items from a list. <br><br>
# Allows duplicate elements: Elements can repeat in a list. Each occurrence of a value maintains its own place in the list.<br><br>
# To create a list, elements are specified within [ ] by placing comma-separated . <br><br>
# <b>Syntax:</b> Object_Name = [item1, item2, item3] <br>
# </span></li></ul> <br><br>
#  
# 
# <br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[15]:


# Creating a list - Example 1
list1 = [1, 2, 3, 2]  
print(list1)


# In[21]:


# Creating a list - Example 2
my_list = [1, 'hello', 3.14, True]
my_list

# You can print without writing print itself by just calling the name of whatever you want to print


# In[22]:


# Creating a list - Example 3
nested_list = [[1, 2, 3], ['a', 'b', 'c'], [True, False]]
nested_list


# In[24]:


# The second element ('banana') in the fruits list is replaced with 'orange'
fruits = ['apple', 'banana', 'cherry']
fruits[1] = 'orange'
print(fruits)


# #### Lists are incredibly flexible and useful for many programming tasks, including storing collections of data, iterating over elements, and performing operations like sorting and searching.

# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# Write python code(s) to perform the following tasks:<br>
#   <ul><li><span style="color:#00008B;font-size:16px">
#    a. Ask the user to input six numeric values (value1, value2, value3, value4, value5, value6).  Store these values in a list. <br>
#   </span></li></ul>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[26]:


# Prompting the user to input six numeric values
value1 = float(input("Enter value 1: "))
value2 = float(input("Enter value 2: "))
value3 = float(input("Enter value 3: "))
value4 = float(input("Enter value 4: "))
value5 = float(input("Enter value 5: "))
value6 = float(input("Enter value 6: "))

# Storing the values in a list
numeric_values = [value1, value2, value3, value4, value5, value6]

# Printing the list of numeric values
print("List of numeric values:", numeric_values)


# <div style="background-color:beige;color:beige">
# 
# <h3 style="padding:1em;text-align:left;color:#00008B">Accessing Elements in a List</h3>
# 
# <ul><li><span style="color:#00008B;font-size:16px">
# In Python, a list is a versatile data structure that allows you to store and manipulate collections of items. Each element in a list is indexed based on its position, starting from 0.<br><br>
# Each element in a list is indexed depending on its position. <br><br>
# <b>Example 1:</b> In the list list1 = [1, 2, 3, 2], element 1[1], is indexed at position 0, element 2[2] at 1, element 3[3] at 2, and element 4[2] at 3. <br><br>
# 
# <b>Example 2:</b>
# Consider the list list1 = [1, 2, 3, 2].
# Element 1 is at index 0.
# Element 2 is at index 1.
# Element 3 is at index 2.
# Element 2 (second occurrence) is at index 3.
# To access a particular element in a list, you use its position within square brackets ([ ]):
# Syntax: list_name[position]<br><br>
#     
#     
# <b>index()</b> -> Use this method to know position of any element.
# For example:list1.index(2) would return 1 (the index of the first occurrence of 2).
# list1.index(3) would return 2.
# Syntax: to access a particular element in a list: list_name[position]<br><br>
#     
# <b>Slicing</b> -  range of elements between two positions. Slicing allows you to extract a subset of elements from a list, creating a new list:
# Syntax: list_name[start:end]
#     
# start is the index where the slice begins (inclusive).
# end is the index where the slice ends (exclusive).
# 
# Using negative indices:
# list1[:-1] would return [1, 2, 3]. This excludes the last element.
# list1[-2:] would return [3, 2]. This extracts the last two elements.
#     
# Syntax: list_name[position1:position2]<br><br>
# </span></li></ul> <br><br>
#  
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[28]:


# Index searches for an element in a list and returns its postion
list3 = [1, 2, 2.5, 4, 6, 'Python', 6, 8, 'Farah']
print(list3.index('Python'))


# In[32]:


# Accessing the third element of this list3 [this is at index position 2]
print(list3[2])


# In[30]:


# Define a list
list_extra = [1, 2, 3, 2, 4]

# Use index() method to find the position of an element
position = list_extra.index(2)

# Print the position
print(f"The first occurrence of the element '2' is at index: {position}")


# In[33]:


# Slicing to access the first four element of this list
list_new = list3[0:4]
list_new


# #### In slicing:
# - The start index is inclusive.
# - The end index is exclusive.
# 
# To include the element at index 4 in the slice, one need to set the end index to 5 because the end index is not included in the result of the slice. Thus, list3 [0:5] will include elements from index 0 to 4.

# In[34]:


# Use the reverse() function to reverese the elements of this list
list3.reverse()
print(list3)


# In[37]:


# Define the list
list_extra = [1, 2, 3, 2, 4]

# Reverse the list in place
list_extra.reverse()

# Print the reversed list
print(list_extra)


# #### To temporarily reverse the list (e.g., for a specific operation) without modifying the original list: 
# - use slicing: list_extra[::-1] 
# 
# or 
# 
# - the reversed() function: list(reversed(list_extra)).
# 
# If you have permanently reversed a list using the **reverse( )** method and want to change it back to its original order, you can simply call the **reverse( )** method again. Reversing a list twice returns it to its original order.

# In[38]:


# Use the len() function to find the length of a list or how many elements are there in a list
print(len(list3))


# In[40]:


# Define an empty list
empty_list = []

# Use len() function to find the length of the list
length = len(empty_list)

# Print the length
print(len(empty_list))


# In[41]:


# Define a list with mixed data types
mixed_list = [1, "Python", 3.14, True, [1, 2, 3]]

# Use len() function to find the length of the list and print it
print(len(mixed_list))


# In[42]:


# Define a nested list
nested_list = [1, [2, 3], [4, 5, 6], 7]

# Use len() function to find the length of the list and print it
print(len(nested_list))


# <div style="background-color:beige;color:beige">
# 
# <h3 style="padding:1em;text-align:left;color:#00008B">Append/Insert Elements to a List</h3>
# 
# <ul><li><span style="color:#00008B;font-size:16px">
#     <b>append()</b> - Add elements in the end of a list. <b>Syntax</b>: listname.append(element)<br><br>
# <b>insert()</b> - Add elements at a specific position in a list. <b>Syntax</b>: listname.insert(index_position, element)<br><br>
# <b>extend()</b> - Append several elements in the list. <b>Syntax</b>: listname.extend([element1, element2, element3])<br><br>
#     <b>Concatenating lists</b> - Two or more lists can be concatenated using the "+" symbol. <b>Syntax</b>: list3 = list1+list2
# </span></li></ul> <br><br>
#  
# <br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[43]:


# Add a element in a list using append()
list4 = [1, 2, 3]
list4.append(4)
print(list4)


# In[44]:


# Insert a element in a list using insert()
list4.insert(1, 'Farah')
print(list4)


# In[47]:


# Append list of elements in a list using extend()
list4.extend([5, 6, 7, 'Umar', 'Maryam', 8, 9, 10])
print(list4)


# In[48]:


# Concatenate two lists using + symbol
list5 = [11, 12, 'Python']
new_list = list4 + list5
print(new_list)


# <div style="background-color:beige;color:beige">
# 
# <h3 style="padding:1em;text-align:left;color:#00008B">Remove Elements from a List</h3>
# 
# <ul><li><span style="color:#00008B;font-size:16px">
# <b>remove( )</b> - Removes the first occurence from a list, which matches the given value <b>Syntax</b> - listname.remove(element)<br><br>
# <b>pop( )</b> - By default removes the last value in a list, if a index position is not specified <b>Syntax</b> - listname.pop(index_position)<br><br>
# </span></li></ul> <br><br>
#  
# 
# <br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[49]:


# Remove a element from a list using remove method
list6 = [1, 2, 3, 4, 5, 4]
list6.remove(4)
print(list6)


# In[50]:


# Remove a element from a list using pop() and specify the index position [eg: element at index position 1]
list6.pop(1)
print(list6)


# In[53]:


# Remove a element from a list using del() 
del list6[3]
print(list6)


# In[57]:


list6.insert(1, 2)
print(list6)


# In[60]:


list6.insert(4, 8)
list6.insert(3, 4)
list6.insert(5, 7)
list6.insert(6, 8)
print(list6)

#3 - defines the position where it will be added
#4 - defines the number 4 will be added at position 3


# ### Important Note:
# - **insert( )** is used to insert elements at a specific position in the list.
# - **append( )** is used to add elements to the end of the list.

# <div style="background-color:beige;color:beige">
# 
# <h3 style="padding:1em;text-align:left;color:#00008B">Sorting a List</h3>
# 
# <ul><li><span style="color:#00008B;font-size:16px">
# <b>sort( )</b> - Use this function to sort a list. <br><br>
# <b>Syntax</b> - listname.sort( )<br><br>
# Note: Sorting doesn't work if your list includes a mix of numeric and string elements.
# 
# </span></li></ul> <br><br>
#  
# 
# <br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[61]:


# By default sort() performs sorting in ascending order
list7 = [11, 2, 43, 24, 55, 16]
list7.sort()
print(list7)


# In[62]:


# To sort in descending order, set reverse as True
list7.sort(reverse=True)
print(list7)


# In[63]:


# Sorting a list having string elements
list8 = ['ba', 'ax', '1m']
list8.sort()
print(list8)


# In[65]:


data = [10, 3.5, 'Alice', 7.2, 'Bob', 1.1, 5, 'Charlie', 2.5]

# Sort the list
sorted_data = sorted(data, key=lambda x: (isinstance(x, str), x))

# Print sorted list
print(sorted_data)


# The key function lambda x: (isinstance(x, str), x) is used to create a tuple for each element x in the list:
# 
# - isinstance(x, str) is True if x is a string, False otherwise. This ensures that strings are sorted after numeric values.
# - x itself is used as the secondary sorting criteria. This ensures that numbers are sorted numerically.
# 
# A **tuple** is an ordered collection of elements, similar to a list

# <div style="background-color:beige;color:beige">
# 
# <h3 style="padding:1em;text-align:left;color:#00008B">List Comprehension</h3>
# 
# <ul><li><span style="color:#00008B;font-size:16px">
# Loop over a sequence and create a new list. <br><br>
# Shorter and better alternative than using traditional loops (like, for) in Python.<br><br>
# 
# </span></li></ul> <br><br>
#  
# 
# <br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# 
# <h3 style="padding:1em;text-align:left;color:#00008B">List Comprehension</h3>
# 
# <ul><li><span style="color:#00008B;font-size:16px">
# List comprehensions are often preferred because they are more concise and readable compared to traditional for loops. They can also be faster in some cases.<br><br>
# 
# The basic syntax of a list comprehension is: [expression **for** item **in** iterable]
# Where:
# 
# - expression is the operation you want to perform on each item in the iterable,
# - item is the variable representing each item in the iterable,
# - iterable is the original iterable object. <br><br>
# 
# 
# 
# </span></li></ul> <br><br>
#  
# 
# <br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[66]:


# Calculating square values of elements in a list using a for loop
# ** used for squaring

object16 = [1, 2, 3, 4]
square_object16 = []
for x in object16:
    square_object16.append(x**2)
square_object16


# 1. **object16** = [1, 2, 3, 4]: This is the original list containing the numbers [1, 2, 3, 4].
# 
# 2. **square_object16** = []: This initializes an empty list square_object16 where we will store the squared numbers.
# 
# 3. **for x in object16:**: This starts a for loop that iterates over each element x in the list object16.
# 
# 4. **square_object16.append(x**2)**: Inside the loop, each number x is squared (x**2) and the result is added to the square_object16 list using the append() method.
# 
# 5. **square_object16**: Finally, when the loop is done, square_object16 contains [1, 4, 9, 16], which are the squares of [1, 2, 3, 4].

# In[67]:


# Calculating square values of elements using list comprehension
object17 = [1, 2, 3, 4]
square_object17 = [x**2 for x in object17]
square_object17

# you can also use pow() from math module, for calculating square values or a value to any power.


# 1. **object17** = [1, 2, 3, 4]: This is the original list containing the numbers [1, 2, 3, 4].
# 
# 2. **square_object17** = [x**2 for x in object17]: This line uses a list comprehension to create a new list square_object17 where each element is the square of the corresponding element in object17.
# 
# - [x**2 for x in object17]: This is the list comprehension syntax.
# - x**2 is the expression that calculates the square of x.
# - for x in object17 iterates over each element x in object17.
# So, this line essentially says: "For each x in object17, calculate x**2 and create a new list with these values."
# 
# 3. **square_object17**: Finally, square_object17 contains [1, 4, 9, 16], which are the squares of [1, 2, 3, 4].

# In[86]:


import math

object18 = [1, 2, 3, 4]
square_object18 = [math.pow(x, 2) for x in object18]
square_object18 
# Output: [1.0, 4.0, 9.0, 16.0]


# - import math imports the math module, which provides mathematical functions.
# - math.pow(x, 2) calculates x raised to the power of 2 (which is the square of x).

# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Tupple</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Stores heterogenous data items - numeric, string (within quotes), or other data structures. <br><br>
# They are immutable, i.e., elements cannot be changed after creation. <br><br>
# To create a tupple, elements are specified within (). <br>
# <b>Syntax:</b> Object_Name = (item1, item2, item3) <br><br>
# A tupple can also be created without parentheses, by using commas. <br>
# <b>Syntax:</b> Object_Name = item1, item2, item3 <br><br>
# Faster than Lists. <br><br>
# 
# </span></li></ul> <br><br>
#  
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[69]:


# Creating a tuple with parentheses
tuple1 = (1, 2, 3)
print(tuple1)


# In[70]:


# Creating a tuple without parentheses
tuple2 = 1, 2, 3
print(tuple2)


# In[71]:


# Accessing elements in a tuple
tuple3 = (1, 2, 3, 4, 5, 6)
print(tuple3)

# Accessing the third element [this is at index position 2]
print(tuple3[2])


# In[72]:


# Slicing to access the first four element
print(tuple3[0:4])


# In[73]:


# Concatenate two tuples using + symbol
tuple4 = (11, 12, 'Python')
new_tuple = tuple3 + tuple4
print(new_tuple)


# In[74]:


# Use the len() function to find the length or how many elements are there in a tuple
print(len(new_tuple))


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# Write python code(s) to perform the tasks as you did for List exercise, but instead of a list, use a tuple. Are you able to perform all the tasks? If not, why?<br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[75]:


# Creating a tuple
F_tuple = (1, 2, 3, 4, 5)

# Accessing elements of a tuple
print(F_tuple[0])  # Output: 1
print(F_tuple[1:3])  # Output: (2, 3)


# In[78]:


# Iterating through a tuple
for item in F_tuple:
    print(item)
    
    #Output: 1 2 3 4 5 


# In[79]:


# Concatenating tuples
tuple_M = (1, 2, 3)
tuple_N = (4, 5, 6)
tuple_concatenated = tuple_M + tuple_N
print(tuple_concatenated)  # Output: (1, 2, 3, 4, 5, 6)


# In[81]:


# Checking membership in a tuple
if 3 in F_tuple:
    print("Found")
else:
    print("Not found")
# Output: Found


# In[85]:


# Finding length of a tuple
print(len(F_tuple))  # Output: 5

# Counting occurrences of an element in a tuple
count = F_tuple.count(3)
print(count)  # Output: 1

# Finding index of an element in a tuple
index = F_tuple.index(3)
print(index)  # Output: 2

# Slicing tuples
slice_tuple = F_tuple[1:4]
print(slice_tuple)  # Output: (2, 3, 4)


# #### Appending or Removing Elements:
# - Tuples do not have methods like append() or remove() because these would modify the tuple, which is not allowed.
# 
# #### Sorting:
# - Tuples cannot be sorted in place. You can use the sorted() function to get a sorted list of tuple elements.
# 
# #### Modifying Elements:
# - You cannot change the elements of a tuple once it's created.
# 
# If your tasks involve operations that modify the tuple (like appending elements), then you won't be able to perform those specific tasks. Otherwise, for tasks that involve accessing, iterating, counting, and finding elements in a tuple, you can perform those operations just as you would with a list.

# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Dictionary</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# A collection data type that stores key-value pairs (sequence of items). <br><br>
# It is an unordered, mutable, and indexed collection.<br><br>
# Each item is made of a key and its corresponding value. <br><br>
# To create a dictonary, key and its value are specified within {}, seperated by a ":". Each key-value pair is seprated by a ",". <br>
# <b>Syntax:</b> Object_Name = {key1:item1, key2:item2, key3:item3} <br><br>
# Keys are unique and can't be duplicated. <br><br>
# </span></li></ul> <br><br>
#  
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Dictionary</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# <b>Unordered:</b> <br>
# The elements in a dictionary are not stored in a specific order. As of Python 3.7, dictionaries maintain the insertion order, but this is considered an implementation detail rather than a guarantee. <br><br>
# <b>Mutable:</b> <br>
# Dictionaries can be changed after they are created. You can add, remove, and modify key-value pairs.<br><br>
# <b>Indexed by Keys:</b> <br>
# Each element in a dictionary is a key-value pair. Keys are unique within a dictionary, and they are used to access the corresponding values.<br><br>
# <b>Efficient:</b> <br>
# Dictionaries provide fast lookups, insertions, and deletions because they are implemented using hash tables.
# </span></li></ul> <br><br>
#  
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[8]:


#starts at position 1 rather than 0 in dictionary


# In[87]:


# Creating a dictionary
dict1 = {1:"Farah", 2:"Python", 3:"Anaconda"}
print(dict1) 

# List all the items in a dictionary as key-value pair
print(dict1.items()) 

# Get the keys in a dictionary
print(dict1.keys()) 

# Get the values in a dictionary
print(dict1.values()) 

# Access a particular value
print(dict1[3]) 

# Add a new key-value pair
dict1[4]="Test"
print(dict1)


# In[88]:


# Add a new key-value pair
dict1[4]="Test"
print(dict1)


# In[90]:


# Use the help to check for other methods which can be used for dictionary
help(dict)


# In[91]:


# Using curly braces
my_dict = {"name": "Umar", "age": 12, "city": "Manchester"}

# Using the dict() function
my_dict = dict(name="Umar", age=12, city="Manchester")
my_dict #Output: {'name': 'Umar', 'age': 12, 'city': 'Manchester'}


# In[94]:


# Accessing Elements
print(my_dict["name"])  # Output: Umar

print(my_dict.get("name"))  # Output: Umar
print(my_dict.get("country", "United Kingdom"))  # Output: United Kingdom

# You can also use the get() method, which returns None or a specified default value if the key is not found.


# In[96]:


# Modifying Elements
my_dict["age"] = 13  # Modify existing key-value pair
my_dict["country"] = "USA"  # Add new key-value pair
my_dict #Output: {'name': 'Umar', 'age': 13, 'city': 'Manchester', 'country': 'USA'}


# In[98]:


# Removing Elements
# You can remove elements using the del statement or the pop() method.

del my_dict["city"]  # Removes the key "city"
print(my_dict)

age = my_dict.pop("age")  # Removes the key "age" and returns its value
print(age)  # Output: 13


# In[100]:


# Iterating through keys
for key in my_dict.keys():
    print(key)

# Iterating through values
for value in my_dict.values():
    print(value)

# Iterating through key-value pairs
for key, value in my_dict.items():
    print(f"{key}: {value}")
    
# Output: 
# name
# country
# Umar
# USA
# name: Umar
# country: USA


# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Iterating Through a Dictionary</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# <b>keys():</b> <br>
# Returns a view object containing the dictionary's keys. <br><br>
# <b>values():</b> <br>
# Returns a view object containing the dictionary's values.<br><br>
# <b>items():</b> <br>
# Returns a view object containing the dictionary's key-value pairs.<br><br>
# <b>update():</b> <br>
# Updates the dictionary with elements from another dictionary or an iterable of key-value pairs.<br><br>
# <b>clear():</b>
# Removes all elements from the dictionary.<br><br>
#     
# </span></li></ul> <br><br>
#  
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[101]:


# Creating a dictionary
person = {"name": "Alice", "age": 25, "city": "New York"}

# Accessing elements
print(person["name"])  # Output: Alice

# Modifying elements
person["age"] = 26
person["country"] = "USA"

# Removing elements
del person["city"]

# Iterating through the dictionary
for key, value in person.items():
    print(f"{key}: {value}")

# Output:
# name: Alice
# age: 26
# country: USA


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# Create a dictionary to store the following protein expression values observed in a patient: protein_a - 5.686, protein_b - 11.356, protein_c - 9.875, protein_d - 45.678. Print the expression value of protein_c (from this dictionary). <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[102]:


# Creating the dictionary with protein expression values
protein_expression = {"protein_a": 5.686, "protein_b": 11.356, "protein_c": 9.875, "protein_d": 45.678}

# Printing the expression value of protein_c
print(protein_expression["protein_c"])

#Output: 9.875


# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Sets</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Constructed from a sequence.<br><br>
# An unordered collection of unique elements. <br><br>
# Useful for storing elements when the presence of an element matters more than the order or number of occurrences.<br><br>
# Cannot have duplicate values in a particular set. <br><br>
# They support mathematical operations like union, intersection, difference, and symmetric difference. <br><br>
# <b>Syntax:</b> Object_Name = set([item1, item2]) <br><br>
# Set operations can be performed - Union, Intersection, Difference, Subset, Superset. <br><br>
# </span></li></ul> <br><br>
#  
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar</td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[103]:


# Creating two sets
set1 = set([1, 2, 3, 4])
print(set1)

set2 = set([4, 5, 6, 7, 8])
print(set2)


# In[104]:


# Set operation: Union
print(set1.union(set2))


# In[105]:


# Set operation: Intersection
print(set1.intersection(set2))


# In[106]:


# Set operation: Difference
print(set1.difference(set2))
print(set2.difference(set1))


# In[107]:


# Set operation: Subset
print(set1.issubset(set2))


# In[108]:


# Set operation: Superset
print(set1.issuperset(set2))


# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Key Features of Sets</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# <b>Unordered:</b> <br>
# The elements in a set do not have a specific order. <br><br>
# <b>Unique Elements:</b> <br>
# Each element in a set must be unique. Duplicate elements are automatically removed.<br><br>
# <b>Mutable:</b> <br>
# Sets can be changed after their creation. You can add or remove elements.<br><br>
# <b>Hashable Elements:</b> <br>
# Elements in a set must be immutable (hashable), which means you can only store hashable types such as numbers, strings, and tuples (if they contain only hashable types).<br><br>
#     
# </span></li></ul> <br><br>
#  
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[110]:


# Using curly braces
my_set = {1, 2, 3, 4, 5}

# Using the set() function
my_set = set([1, 2, 3, 4, 5])
my_set


# In[111]:


# You can add elements to a set using the add() method
my_set.add(6)
print(my_set)  # Output: {1, 2, 3, 4, 5, 6}


# In[112]:


# You can remove elements using the remove() method or the discard() method. 
# The remove() method will raise an error if the element is not found, while the discard() method will not.
my_set.remove(6)
my_set.discard(6)  # No error if 6 is not found
print(my_set)  # Output: {1, 2, 3, 4, 5}


# In[113]:


# The union of two sets contains all elements from both sets.
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union_set = set1.union(set2)
print(union_set)  # Output: {1, 2, 3, 4, 5}


# In[115]:


# The intersection of two sets contains only the elements that are present in both sets.
intersection_set = set1.intersection(set2)
print(intersection_set)  # Output: {3}


# In[117]:


# The difference between two sets contains the elements that are in the first set but not in the second set.
difference_set = set1.difference(set2)
print(difference_set)  # Output: {1, 2}


# In[118]:


# The symmetric difference contains elements that are in either of the sets but not in both.
sym_diff_set = set1.symmetric_difference(set2)
print(sym_diff_set)  # Output: {1, 2, 4, 5}


# In[119]:


# Creating two sets
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union: combines all elements from both sets
union_set = set1.union(set2)
print("Union:", union_set)
# Output: Union: {1, 2, 3, 4, 5, 6, 7, 8}

# Intersection: elements present in both sets
intersection_set = set1.intersection(set2)
print("Intersection:", intersection_set)
# Output: Intersection: {4, 5}

# Difference: elements in set1 but not in set2
difference_set = set1.difference(set2)
print("Difference (set1 - set2):", difference_set)
# Output: Difference (set1 - set2): {1, 2, 3}

# Difference: elements in set2 but not in set1
difference_set2 = set2.difference(set1)
print("Difference (set2 - set1):", difference_set2)
# Output: Difference (set2 - set1): {8, 6, 7}

# Symmetric Difference: elements in either set1 or set2 but not in both
sym_diff_set = set1.symmetric_difference(set2)
print("Symmetric Difference:", sym_diff_set)
# Output: Symmetric Difference: {1, 2, 3, 6, 7, 8}

# Subset: set1 is a subset of set2 (all elements of set1 are in set2)
is_subset = set1.issubset(set2)
print("set1 is subset of set2:", is_subset)
# Output: set1 is subset of set2: False

# Subset: {1, 2, 3} is a subset of set1
is_subset2 = {1, 2, 3}.issubset(set1)
print("{1, 2, 3} is subset of set1:", is_subset2)
# Output: {1, 2, 3} is subset of set1: True

# Superset: set1 is a superset of {1, 2, 3} (set1 contains all elements of {1, 2, 3})
is_superset = set1.issuperset({1, 2, 3})
print("set1 is superset of {1, 2, 3}:", is_superset)
# Output: set1 is superset of {1, 2, 3}: True

# Superset: set1 is a superset of set2
is_superset2 = set1.issuperset(set2)
print("set1 is superset of set2:", is_superset2)
# Output: set1 is superset of set2: False


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# 
# Write python code(s) to perform the following tasks:<br>
#   <ul><li><span style="color:#00008B;font-size:16px">
#    a. Create a set to store the following values: Gene_a, Gene_b, Gene_c. Create another set to store the following values: Gene_x, Gene_y, Gene_b, Gene_z.<br>
#    b. Perform the following set operations and print the output: union, intersection, difference. <br>
# </span></li></ul>
# </span></li></ul>
#     <br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[120]:


# a. Create sets
set1 = {"Gene_a", "Gene_b", "Gene_c"}
set2 = {"Gene_x", "Gene_y", "Gene_b", "Gene_z"}

# b. Perform set operations and print the output

# Union
union_set = set1.union(set2)
print("Union:", union_set)

# Intersection
intersection_set = set1.intersection(set2)
print("Intersection:", intersection_set)

# Difference (set1 - set2)
difference_set = set1.difference(set2)
print("Difference (set1 - set2):", difference_set)

# Difference (set2 - set1)
difference_set2 = set2.difference(set1)
print("Difference (set2 - set1):", difference_set2)

# Union: {'Gene_c', 'Gene_b', 'Gene_a', 'Gene_x', 'Gene_y', 'Gene_z'}
# Intersection: {'Gene_b'}
# Difference (set1 - set2): {'Gene_a', 'Gene_c'}
# Difference (set2 - set1): {'Gene_x', 'Gene_z', 'Gene_y'}


# #### Intersection:
# - The set of elements that are common to both sets.
# 
# #### Difference (set1 - set2):
# - The set of elements that are in `set1` but not in `set2`.
# 
# #### Difference (set2 - set1):
# - The set of elements that are in `set2` but not in `set1`.

# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Summary</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Data Structures in Python - List, Tuple, Dictionary, Set. <br><br>
# List, Dictionary and Set are mutable, while Tuple is immutable.<br><br>
# <a href="https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences">Python Data Structures - Resource 1</a> <br><br>
# <a href="https://thomas-cokelaer.info/tutorials/python/data_structures.html#:~:text=Data%20Structures%20(list%2C%20dict%2C%20tuples%2C%20sets%2C%20strings),-There%20are%20quite&text=The%20builtins%20data%20structures%20are,contain%20any%20type%20of%20objects.">Python Data Structures - Resource 2</a>
# </span></li>
#     </ul>
# 
# <br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>
