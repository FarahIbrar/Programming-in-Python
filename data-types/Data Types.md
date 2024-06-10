<div style="background-color:beige;color:beige">
<header>
<h1 style="padding:1em;text-align:center;color:#00008B">Basics of Python programming <br><br> Data Types </h1> 
</header>
<br><br><br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>

<div style="background-color:beige;color:beige">
<header>
<h1 style="padding:1em;text-align:left;color:#00008B">Outline</h1>
</header>

    
<ul><li><span style="color:#00008B;font-size:24px">Numeric - Integer, Float</span> <br><br>
        <li><span style="color:#00008B; font-size:24px">Strings </span> <br><br>
        <li><span style="color:#00008B; font-size:24px">Boolean </span>
        </li>
       </ul>

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>

<div style="background-color:beige;color:beige">

<h2 style="padding:1em;text-align:left;color:#00008B">Numeric - Integer</h2>

<ul><li><span style="color:#00008B;font-size:20px">
Represents range of mathematical integer numbers. <br><br>
Whole number (not a fraction) that can be positive, negative, or zero. E.g.: 565, -55, 0. <br><br>
Works similar as in other programming languages. <br><br>
<b>Syntax:</b> Variable_Name = Value <br>
</span></li></ul> <br><br>
 

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Creating a variable x which stores a value of 5
x = 5
```


```python
# To check, specify the variable name or use print function, either of the following works fine: x or print(x)
x
print (x)
```

    5



```python
# Another example
y = -2
y
```




    -2




```python
# Creating a variable x which stores a value of 5
x = 5

# To check, specify the variable name or use print function, either of the following works fine: x or print(x)
x

# Another example
y = -2


# Let's try some arithmetic operations - add, subtract, multiply and divde are variable names, which stores the 
# result values for addition, subtraction, multiplication and division respectively
add = x+y
subtract = x-y
multiply = x*y
divide = x/y

print("Addition value is:", add)
print("Subtraction value is:", subtract)
print("Multiplication value is:", multiply)
print("Divison value is:", divide)
```

    Addition value is: 3
    Subtraction value is: 7
    Multiplication value is: -10
    Divison value is: -2.5


<div style="background-color:beige;color:beige">

<h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>

<ul><li><span style="color:#00008B;font-size:20px">
Write a python code to perform the following tasks:<br>
  <ul><li><span style="color:#00008B;font-size:16px">
   a. Create two variables to store the following values: 450 (var1), -625 (var2). <br>
   b. Perform the following arithmetic operations on var1 and var2: multiplication, addition. <br>
   c. Perform the following arithmetic operation and print the result: (var1*var2)/(var1-var2). <br><br>
  </span></li></ul>
 

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Part a: Create variables
var1 = 450
var2 = -625

# Part b: Perform arithmetic operations
multiplication_result = var1 * var2
addition_result = var1 + var2

# Part c: Perform arithmetic operation and print the result
result_c = (var1 * var2) / (var1 - var2)
print("Result of (var1 * var2) / (var1 - var2):", result_c)

# Printing the other results as well
print("Multiplication result (var1 * var2):", multiplication_result)
print("Addition result (var1 + var2):", addition_result)
```

    Result of (var1 * var2) / (var1 - var2): -261.6279069767442
    Multiplication result (var1 * var2): -281250
    Addition result (var1 + var2): -175


<div style="background-color:beige;color:beige">

<h2 style="padding:1em;text-align:left;color:#00008B">Numeric - Float</h2>

<ul><li><span style="color:#00008B;font-size:20px">
Represents range of mathematical rational numbers. <br><br>
Values are specified with a decimal point. E.g. - 56.5, 10.686. <br><br>
Exponential notation: 'e'/'E' followed by a positive or negative integer value may be appended to specify scientific notation. e.g. - 1e6, 5.6e-5,  <br><br>
<b>Syntax:</b> Variable_Name = Value <br>
</span></li> <br><br>
    </ul> 

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Creating a variable f1 which stores a value of 56.5
f1 = 56.5

# To check, specify the variable name or use print function, either of the following works fine: f or print(f)
f1
```




    56.5




```python
# Creating a variable f1 which stores a value of 56.5
f1 = 56.5

# To check, specify the variable name or use print function, either of the following works fine: f1 or print(f1)
f1

# Creating a variable f2 which stores a value of 0.00056
f2 = 5.6e-4
f2
```




    0.00056



<div style="background-color:beige;color:beige">

<h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>

<ul><li><span style="color:#00008B;font-size:20px">
Write a python code to perform the following tasks:<br>
  <ul><li><span style="color:#00008B;font-size:16px">
   a. Create three variables to store the following values: 450 (var1), -625 (var2), 12.78 (var3). <br>
   b. Perform the following arithmetic operations on var1, var2 and var3: subtraction, divison. <br>
   d. Perform the following arithmetic operation and print the result: (var1*var2+var3)/(var1-var3). <br><br>
  </span></li></ul>
 

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Part a: Create variables
var1 = 450
var2 = -625
var3 = 12.78

# Part b: Perform arithmetic operations
subtraction_result = var1 - var2
division_result = var1 / var3

# Part d: Perform arithmetic operation and print the result
result_d = (var1 * var2 + var3) / (var1 - var3)
print("Result of (var1 * var2 + var3) / (var1 - var3):", result_d)

# Printing the other results as well
print("Subtraction result (var1 - var2):", subtraction_result)
print("Division result (var1 / var3):", division_result)
```

    Result of (var1 * var2 + var3) / (var1 - var3): -643.2396047756278
    Subtraction result (var1 - var2): 1075
    Division result (var1 / var3): 35.21126760563381


<div style="background-color:beige;color:beige">

<h2 style="padding:1em;text-align:left;color:#00008B">Strings</h2>

<ul><li><span style="color:#00008B;font-size:20px">
Used to represent text. <br><br>
Text is delimited using either single(') or double (") or triple quotes("""). eg: 'My name is Nadeer.'<br><br>
<b>Syntax:</b> Variable_Name = Value <br>
</span></li> <br><br>
    </ul>

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Creating a variable line1 which stores a string
line1 = "my name is nadeer."

line1
```




    'my name is nadeer.'




```python
# Creating a variable which stores a string with quotes - my name is "nadeer"

# If the string has a quotation, don't use the same quote as this will give an error
# Use combination of single and double or triple quotes
line2 = "my name is "nadeer.""

line2

# it will throw an error because quote marks on different places
```


      Cell In[10], line 5
        line2 = "my name is "nadeer.""
                             ^
    SyntaxError: invalid syntax




```python
# Creating a variable which stores a string with quotes - my name is "nadeer"

# If the string has a quotation, don't use the same quote as this will give an error
# Use combination of single and double or triple quotes
line2 = 'my name is "nadeer."'

line2

#the error can be fixed using single quotes.
```




    'my name is "nadeer."'




```python
# Creating a variable line1 which stores a string with a single quote
line1 = "my name is nadeer."

# Capitalize a string - use 'capitalize' function
print(line1.capitalize())
```

    My name is nadeer.



```python
# Creating a variable line1 which stores a string with a single quote
line1 = "my name is nadeer."

# Convert the string to uppercase - use 'upper' function
print(line1.upper())
```

    MY NAME IS NADEER.


<div style="background-color:beige;color:beige">

<h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>

<ul><li><span style="color:#00008B;font-size:20px">
Write a python code to store the following text in a variable: <b>"Python 3.8 was released on October 14th, 2019" </b>. Convert and print this entire text in lowercase. <br>
  [Hint: use a function which does opposite of upper() function]. <br><br>
</span></li></ul>
 

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Store the text in a variable
text = "Python 3.8 was released on October 14th, 2019"

# Convert the text to lowercase
lowercase_text = text.lower()

# Print the lowercase text
print(lowercase_text)
```

    python 3.8 was released on october 14th, 2019


<div style="background-color:beige;color:beige">

<h2 style="padding:1em;text-align:left;color:#00008B">Boolean</h2>

<ul><li><span style="color:#00008B;font-size:20px">
Intend to represent the truth values of logic/boolean algebra.<br><br>
This datatype can have one of the two possible values - <b>TRUE</b> or <b>FALSE</b> <br><br>
<b>Syntax:</b> Variable = Value [True/False] <br><br>
<br><br>
</span></li>
    </ul>

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Creating variables x and y, which stores boolean values
x = True
y = False
print(x,y)
```

    True False


<div style="background-color:beige;color:beige">

<h2 style="padding:1em;text-align:left;color:#00008B">Boolean</h2>

<ul><li><span style="color:#00008B;font-size:20px">
Intend to represent the truth values of logic/boolean algebra.<br><br>
This datatype can have one of the two possible values - <b>TRUE</b> or <b>FALSE</b> <br><br>
<b>Syntax:</b> Variable = Value [True/False] <br><br>
Comparison of variables(objects) returns boolean data.<br><br>
    
#### Check whether a number is greater than 50 or not [a=input_value, b=50]<br><br>
    
#### Comparission Operations (Relational Operators)
1. "less than" [eg: a < b] <br><br>
2. "greater than" [eg: a > b] <br><br>
3. "less than or equal" [eg: a <= b] <br><br>
4. "greater than or equal" [eg: a>= b] <br><br>
5. "equal" [eg: a == b] <br><br>
6. "not equal" [eg: a!= b] <br><br>

#### Boolean Operations (Boolean Operators)
1. or [eg: a or b] <br><br>
2. and [eg: a and b]<br><br>
3. not [eg: a not b]<br><br>
    
</span></li>
    </ul>
  
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Boolean operations on two variables
x = 50
y = 75
x > y
```




    False




```python
# Creating variables x and y, which stores boolean values
x = True
y = False

# Check whether a number is greater than 50 or not
z = int(input("Enter a value:"))
if z > 50:
    print(x)
else:
    print(y)
```

    Enter a value:79
    True


<div style="background-color:beige;color:beige">

<h2 style="padding:1em;text-align:left;color:#00008B">Boolean</h2>

<ul><li><span style="color:#00008B;font-size:20px">
Intend to represent the truth values of logic/boolean algebra.<br><br>
This datatype can have one of the two possible values - <b>TRUE</b> or <b>FALSE</b> <br><br>
<b>Syntax:</b> Variable = Value [True/False] <br><br>
Comparison of variables(objects) returns boolean data.<br><br>
Can be used to convert numbers [use function: bool()]. A non-zero value is TRUE, while zero is always FALSE.
</span></li>
    </ul> 

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Convert numeric to a boolean value
bool(5432)
```




    True




```python
# Convert 0 to a boolean value
bool(0)
```




    False



<div style="background-color:beige;color:beige">

<h1 style="padding:1em;text-align:center;color:#00008B">EXERCISE</h1>

<ul><li><span style="color:#00008B;font-size:20px">
Write a python code asking the user to input two values (value1 and value2). Check value1 is greater than equal to value2.<br>
</span></li></ul>
 

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>


```python
# Ask the user to input two values
value1 = float(input("Enter the first value (value1): "))
value2 = float(input("Enter the second value (value2): "))

# Check if value1 is greater than or equal to value2
if value1 >= value2:
    print(f"{value1} is greater than or equal to {value2}")
else:
    print(f"{value1} is less than {value2}")
```

    Enter the first value (value1): 89
    Enter the second value (value2): 19
    89.0 is greater than or equal to 19.0


<div style="background-color:beige;color:beige">

<h2 style="padding:1em;text-align:left;color:#00008B">Summary</h2>

<ul><li><span style="color:#00008B;font-size:20px">
Common data types in Python - Numeric[Integer, Float, Complex], String, Boolean. <br><br>
Arithmetic Operations.<br><br>
Boolean and Comparison Operations.<br><br>
Functions - print(), type(), input(), capitalize(), upper(), bool(). <br><br>
</span></li>
    </ul>

<br><br><br><br>
<footer>
<table style="border:none;width:100%">
<tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
</footer>

</div>
