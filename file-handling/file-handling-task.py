#!/usr/bin/env python
# coding: utf-8

# # Basics of Python programming

# ### **How to open a file?**

# **Open a File**
# open() - to open a file. Need to define the following two parameters - filename, mode. 
# 
# **Close a File**
# close() - to close a file. 

# Modes for opening a file: 
# 1. "r" (Read is the default): Opens a file for reading. Will give an error if the file does not exist. 
# 
# 2. "w" (Write): Opens a file for writing. Creates the file if it does not exist. Will overwrite any existing content 
# 
# 3. "a" (Append): Opens a file for appending. Creates the file if it does not exist. Will append at the end of the existing content. 
# 
# 4. "x" (Create): Creates the specified file. Returns an error if the file exists. 
# 

# ##### Open a file - by default is handled as a text and opens in a read mode (for binary mode - specify mode as "b")

# In[3]:


file1 = open("/Users/farah/Downloads/testfile2.txt", "x")
file1


# As there are no file present in the download named as **"test1.text"**, this fucntion of **"test1.text"** with **"x"** will automatically create an empty file and if the same file is already present, it will result in an error. 

# After creating the file, it can be opened and used to make alterations, updates, or customizations to the information stored within the file based on your specific needs or preferences.
# 
# Any types of files are open to this function.

# In[4]:


file1 = open("/Users/farah/Downloads/testfile2.txt")
file1


# ### Read a File
# 
# **"r" (Read is the default)**: Opens a file for reading. Will give an error if the file does not exist. 
# 
# **read()** - by default reads all content of the file. Can specify number of characters to read. 
# 
# **readline** - reads line of a file. 
# 
# **read-lines** - returns a list of lines.

# In[5]:


file1 = open('/Users/farah/Downloads/testfile1.txt')
print (file1.read)


# ^ this fucntion will read the whole file type. 

# In[6]:


file1 = open('/Users/farah/Downloads/testfile1.txt')
print (file1.read())


# ^ As there is no information in the file, when the code ran, it shifted to the next cell.

# In[29]:


#Write something in the file using "w" fucntion and make sure to close it to see the outcome.


# In[7]:


file1 = open('/Users/farah/Downloads/testfile1.txt')
file1


# In[8]:


file1 = open('/Users/farah/Downloads/testfile1.txt', 'w')
file1


# In[9]:


file1.write("My name is Farah. \n")
file1.write("I study at the University of Westminster. \n")
file1.write("I am 22 years old. \n")
file1.write("My birthday is 16/05/2001. \n")
file1.write("I was born in Pakistan. \n")
file1.write("My favourite footballer is Messi. \n")

file1.close()


# In[106]:


#Now try to read the file with different "read" fucntions.


# In[10]:


file1 = open('/Users/farah/Downloads/testfile1.txt')
print (file1.read(3))


# In[11]:


file1 = open('/Users/farah/Downloads/testfile1.txt', 'r')

print (file1.read(3))


# In[12]:


# Open the file in binary mode
with open('/Users/farah/Downloads/testfile1.txt', 'rb') as file1:
    # Read the first three bytes and decode them as UTF-8
    content = file1.read(3).decode('utf-8', errors='ignore')
    print(content)


# In[108]:


file1 = open("/Users/farah/Downloads/test1.txt", 'r')
print (file1.readlines()[5])


# In[ ]:


#As sometimes the code might not work so it has to be customised in another way so all three ways are right. 


# In[13]:


file1 = open("/Users/farah/Downloads/test1.txt", 'r')

line = file1.readline()
print (line)

file1.close()


# In[14]:


file1 = open("/Users/farah/Downloads/test1.txt", 'r')

line = file1.readline(2)
print (line)

file1.close()


# In[15]:


file1 = open("/Users/farah/Downloads/test1.txt", 'r')

for x, line in enumerate(file1):
    if x == 2:  # Line numbers start from 0, so the third line has index 2
        print(line)
        break

file1.close()


# Sometimes the code might be working as loop so the **"for x in"** function can be used which also uses **"enumerate"** and **"break"** fucntion.

# Append function can also be used in order to write something in one file. 

# X fucntion can also be usedful when a new file needs to be created, it will return with an error if there is already same file exists. 

# In[16]:


file1 = open("/Users/farah/Downloads/test1.txt", 'r')
file1.read()


# ### Summary
# 
# File handling in Python - open, close, read, write, append, create. 
# A file by default is handled in text mode, for binary mode use the mode "b" (e.g. - for images).
# 
# 1. https://docs.python.org/3/tutorial/inputoutput.html 
# 2. https://realpython.com/read-write-files-python
# 3. https://docs.python.org/3/library/csv.html

# ### EXERCISE
# 
# **Write python code(s) to perform the following tasks:**
# 
# Create a file named "my_first_file.txt". 
# 
# Write the following lines in this file 
# [Line 1 - "The first argument is a string containing the filename." 
# Line 2 - "The second argument is another string containing a few characters describing the way in which the file will be used."] 
# 
# Print the first five characters of this file. 
# 
# Append the following line in this file [Line - "In text mode, the default when reading is to convert platform-specific line endings (\n on Unix, \r\n on Windows) to just \n."]
# 
# Print all the lines of this file.

# In[134]:


file2 = open("/Users/farah/Documents/my_first_file.txt", "x")
file2


# In[135]:


file2 = open("/Users/farah/Documents/my_first_file.txt", "w")
file2


# In[137]:


file2.write("This file is my first file using this technology. \n")
file2.write("I will use this file to make Python notes. \n")

file2.close()


# In[151]:


file2 = open("/Users/farah/Documents/my_first_file.txt", "r")
file2


# In[152]:


content = file2.read(5)
print (content)


# In[155]:


file2.close()


# In[156]:


file2 = open("/Users/farah/Documents/my_first_file.txt", "a")
file2


# In[157]:


file2.write("In text mode, the default when reading is to convert platform-specific line endings (\n on Unix, \r\n on Windows) to just \n.")
file2.close()


# In[158]:


file2 = open("/Users/farah/Documents/my_first_file.txt", "a")
file2


# In[160]:


file2 = open("/Users/farah/Documents/my_first_file.txt", "r")
line = file2.readline()
print(line)


# In[161]:


file2 = open("/Users/farah/Documents/my_first_file.txt", "r")
for line in file2:
    print(line)


# In[180]:


file2 = open("/Users/farah/Documents/my_first_file.txt", "r")
line = file2.readline()
print(line)


# In[186]:


words = line.split(' ')


# In[189]:


first_five_words = ' '.join(words[:5])
print(first_five_words)


# In[187]:


first_five = ''
for i in range(5):
    first_five = first_five + words[i] + ' '
print(first_five)


# In[193]:


for i in range(5):
    print(i)


# In[194]:


for i in range(0,5):
    print(i)


# In[195]:


for i in range(0,8):
    print(i)


# In[196]:


for i in range(2,8):
    print(i)


# In[197]:


for i in range(2,8,2):
    print(i)


# The range function works as follows:
# 
# **range(start, end, step)**
# 
# start,end,step refers to the values in the code. 

# In[198]:


for i in range(10,2,-2):
    print(i)


# ### Practice the **append** function in different ways on the same file.

# In[10]:


file3 = open("/Users/farah/Documents/testing_append.txt", "x")
file3


# In[14]:


file3 = open("/Users/farah/Documents/testing_append.txt", "a")
file3


# In[15]:


list = [1, 2, 3, 4, 5, 6]
list


# In[16]:


for x in list: 
    file3.write("%d" %x)
    file3.write("\n")
    

file3.close()


# In[ ]:


# 
get_ipython().run_line_magic('s=', 'string (values will be written as string)')
get_ipython().run_line_magic('d=', 'numbers (vlaues will be written as numbers)')

if you have mixture of string and numbers then it can be run as string. 

make sure to close the file so you can see the outcome. 


# In[18]:


print(file3)


# file3.write( indicates that we want to write something to the file represented by the file3 variable.
# 
# **"%d"** is a formatting placeholder in Python. It specifies that the value to be written should be treated as a decimal (integer) value.
# 
# **"%x"** is used to substitute the placeholder %d with the value of the variable x.
# 
# In simpler terms, this line takes the value of x, converts it to a string representation of a decimal (integer), and writes it to the file.
# 
# For example, if x is equal to 8, the line will be interpreted as file3.write("8"), and the character '8' will be written to the file.
# 
# This allows the code to write the numeric value of x to the file as a string.

# In[22]:


file3 = open("/Users/farah/Documents/testing_append.txt", "a")
file3


# In[23]:


newlist = [2,4,6,8,10]
square_newlist = [x**2 for x in newlist]
square_newlist


# In[24]:


for x in square_newlist:
    file3.write("%d" %x)
    file3.write("\n")
    
file3.close()


# In[25]:


file3 = open("/Users/farah/Documents/testing_append.txt", "a")


# In[26]:


file3.write("8")

