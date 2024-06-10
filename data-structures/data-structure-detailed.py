#!/usr/bin/env python
# coding: utf-8

# # Basics of Python Programming

# ### What is a "List"?

# **List is an ordered and changeable collection of data objects.**
# 
# - Stores heterogenous data items - numeric, string (within quotes), or other data structures. 
# 
# - They are mutable, i.e., elements can be changed after creation. 
# 
# - Elements can repeat in a list. 
# 
# To create a list, elements are specified within []. 
# 
# **Syntax: Object_Name = [item1, item2, item3]**
# 

# * Remember - In Python, anything which can be modified after creation is mutable, while which cannot is immutable.
# 
# Example: Lists, Sets, Dictionaries are mutable. 
# 
# Tuples and basic data types (numeric, boolean, strings) are non-mutable. 

# In[ ]:


# Mutable vs Non-Mutable


# An object that allows you to change its values without changing its identity is a mutable object. 
# - The changes that you can perform on a mutable object's value are known as mutations. 
# 
# In contrast, an object that doesn't allow changes in its value is an immutable object.

# In[ ]:


# Creating a list


# In[1]:


my_list = [2, 4, 6, 8, 10]
print(my_list)


# #### Accessing Elements in a List
# 
# 
# - Each element in a list is indexed depending on its position. 
# - Example: In the list list1 = [1, 2, 3, 2], element 1[1], is indexed at position 0, element 2[2] at 1, element 3[3] at 2, and element 4[2] at 3. 
# 
# **index()** -> Use this method to know position of any element. **Syntax** to access a particular element in a list: list_name[position]
# 
# **Slicing** - range of elements between two positions. **Syntax:** list_name[position1:position2]

# ### EXERCISE
# 
# **Write python code(s) to perform the following tasks:**
# 
# - Ask the user to input six numeric values (value1, value2, value3, value4, value5, value6). 
# - Store these values in a list. 

# In[2]:


list_1 = [12, 14, 16, 18, 20]
print(list_1)


# In[ ]:


# Index searches for an element in a list and returns its postion


# In[3]:


list_1 = [12, 14, 16, 18, 20]
list_1.insert(len(list_1) // 2, "Farah")
print(list_1)


# **This line uses the insert() method to add the element "Farah" to the list.**
# 
# Here's a breakdown of the line:
# 
# - len(list_1) returns the length of list_1, which is 5.
# - len(list_1) // 2 divides the length by 2 using integer division, resulting in the index where "Farah" will be inserted. In this case, it will be inserted at index 2.
# - The insert() method is then called on list_1 with two arguments: the index (2) and the element to be inserted ("Farah").
# - After executing this line, the list will be modified to [12, 14, "Farah", 16, 18, 20], with "Farah" inserted at index 2.

# In[4]:


list_1 = [12, 14, 16, "Farah", 18, 20]
print(list_1.index("Farah"))


# In[ ]:


# Accessing the third element of this list [this is at index position 2] 


# In[5]:


print(list_1[2])


# In[ ]:


# Try it with another example


# In[6]:


list_2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
print(list_2)


# In[7]:


# Inserting different names/numbers at different positions


# In[8]:


list_2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
list_2.insert(2, "Farah")
list_2.insert(6, "Nadeer")
print(list_2)


# In[9]:


# Accessing the seventh element of this list [this is at index position 6] 


# In[10]:


print(list_2[6])


# In[11]:


# Slicing to access the first four element of this list


# slice() function in python is used to extract a continous sequence(it may be an empty sequence, complete sequence or a partial sequence) of any object(array, string, tuple). 
# 
# It returns a slice object which can be used with the array/string which will return the continous sequence after slicing.

# In[12]:


list2_calling = list_2[0:4]
list2_calling


# In[13]:


list2_calling = list_2[2:7]
list2_calling


# In[14]:


list2_calling = [list_2[2], list_2[6]]
print(list2_calling)


# In[15]:


# Use the reverse() function to reverese the elements of this list


# In[16]:


list2_calling.reverse()
print(list2_calling)


# In[17]:


list_2.reverse()
print(list_2)


# In[18]:


# Same fucntion can be used to bring the list to its original state


# In[19]:


# Use the len() function to find the length of a list or how many elements are there in a list


# In[20]:


print(len(list_2))


# ### Append/Insert Elements to a List
# 
# 
# - **append()** - Add elements in the end of a list.  **Syntax:** listname.append(element)
# 
# - **insert()** - Add elements at a specific position in a list. **Syntax:** listname.insert(index_position, element)
# 
# - **extend()** - Append several elements in the list. **Syntax:** listname.extend([element1, element2, element3])
# 
# - **Concatenating lists** - Two or more lists can be concatenated using the "+" symbol. **Syntax:** list3 = list1+list2

# In[21]:


# Add a element in a list using append() 


# In[22]:


list_3 = [1, 2, 3]
list_3.append(4)
print(list_3)


# In[23]:


# Insert a element in a list using insert()


# In[24]:


list_3.insert(1, 'Nadeer')
print(list_3)


# In[25]:


# Append list of elements in a list using extend()


# In[26]:


list_3.extend([5, 6, 7])
print(list_3)


# In[27]:


# Concatenate two lists using + symbol


# In[28]:


list_4 = [8, 9, 10]
list_4.append(11)
print(list_4)


# In[29]:


fresh_list = list_3 + list_4
print(fresh_list)


# ### Now try using all of these fucntions with the previous lists. 

# - work with list_2

# In[30]:


# Add a element in a list using append() 


# In[31]:


list_2.append(22)
print(list_2)


# In[34]:


# Insert a element in a list using insert()


# In[33]:


list_2.insert(len(list_2), 24)
print(list_2)


# In[35]:


# Append list of elements in a list using extend()


# In[37]:


list_2.extend(["Fabz", 28, "Nadz"])
print(list_2)


# In[38]:


# Concatenate two lists using + symbol


# In[39]:


newlist = [30, 32, 34]
conc_list = list_2 + newlist
print (conc_list)


# ### Remove Elements from a List

# - **remove()** - Removes the first occurence from a list, which matches the given value 
# **Syntax** - listname.remove(element)
# 
# - **pop()** - By default removes the last value in a list, if a index position is not specified 
# **Syntax** - listname.pop(index_position)
# 

# In[40]:


# Work using the list_2 example


# In[41]:


# Remove a element from a list using remove method


# In[42]:


list_2.remove(22)
print(list_2)


# As there were 2 of the 22 numbers, so it only removed one.

# In[44]:


# Remove a element from a list using pop() and specify the index position [eg: element at index position 1]


# In[45]:


list_2.pop(12)
print(list_2)


# This helped to remove both 22 from the list. However, it can be used to remove anything else from the list too. 

# ### Sorting a List

# - **sort()** - Use this function to sort a list. 
# 
# - **Syntax** - listname.sort()
# 
# Note: Sorting doesn't work if your list includes a mix of numeric and string elements.

# In[46]:


print(list_2)


# In[47]:


# As this list has mix of numbers and string elements, sorting will not work on this.


# - Make a new list and work with it for now and then figure it out how to sort list_2.

# In[49]:


# By default sort() performs sorting in ascending order


# In[50]:


sortlist = [11, 2, 43, 24, 55, 16]
sortlist.sort()
print(sortlist)


# In[ ]:


# To sort in descending order, set reverse as True)


# In[51]:


sortlist.sort(reverse=True)
print(sortlist)


# In[53]:


# Sorting a list having string elements


# In[54]:


sortlist2 = ['ba', 'ax', '1m']
sortlist2.sort()
print(sortlist2)


# In[55]:


#try it with list_2 now.


# In[57]:


print(list_2)


# In[59]:


list_2 = sorted(list_2, key=lambda x: (isinstance(x, str), x))
print(list_2)


# The sorted() function is used to sort the elements of list_2 based on a custom key function. The key function is defined using a lambda function. 
# 
# 
# Let's break it down further:
# 
# **lambda x:** (isinstance(x, str), x): This lambda function takes an element x as input and returns a tuple (isinstance(x, str), x). 
# 
# The tuple consists of two elements:
# 
# 
# **isinstance(x, str):** This checks if the element x is an instance of the str (string) class. It returns True if x is a string, and False otherwise. This is used to separate the string elements from the numeric elements during the sorting process.
# 
# **x:** This is the original element itself. It ensures that the sorting within each data type (strings and numbers) is done based on the natural order of the elements.
# 
# 
# The sorted() function uses this key function to determine the order of the elements. It will first sort the elements based on the isinstance(x, str) comparison, which separates strings from numbers. Then, within each group, it will sort the elements based on their actual values using x.

# ### List Comprehension

# - Loop over a sequence and create a new list. 
# - Shorter and better alternative than using traditional loops (like, for) in Python.

# In[60]:


# Calculating square values of elements in a list using a for loop
# ** used for squaring


# In[61]:


object16 = [1, 2, 3, 4]
square_object16 = []
for x in object16:
    square_object16.append(x**2)
square_object16


# **`object16 = [1, 2, 3, 4]`**
# - In this line, a list named `object16` is created and initialized with four elements: 1, 2, 3, and 4. This is a simple list containing some numbers.
# 
# 
# **`square_object16 = []`**
# - Here, an empty list named `square_object16` is created. This list will be used to store the squared values of the numbers from `object16` list.
# 
# 
# **`for x in object16:`**
# - This line introduces a loop. It means that for each element `x` in the `object16` list, the following block of code will be executed.
# 
# 
# **`square_object16.append(x**2)`**
# - Inside the loop, the code takes the current element `x`, squares it using the `**` operator, and then appends the squared value to the `square_object16` list using the `append()` method. So, for each number in `object16`, its square is calculated and added to `square_object16`.
# 
# 
# **`square_object16`**
# - Finally, after the loop completes, the code prints the `square_object16` list, which contains the squared values of the numbers from `object16`.

# In[62]:


# Calculating square values of elements using list comprehension


# In[63]:


object17 = [1, 2, 3, 4]
square_object17 = [x**2 for x in object17]
square_object17


# In[64]:


# you can also use pow() from math module, for calculating square values or a value to any power.


# In[65]:


import math

object16 = [1, 2, 3, 4]
square_object16 = []

for x in object16:
    square_value = math.pow(x, 2)
    square_object16.append(square_value)

square_object16


# In[ ]:




