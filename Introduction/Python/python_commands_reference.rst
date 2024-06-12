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
   * - `remove`
     - Removes the first occurrence of a value from a list.
     - ``numbers.remove(3)``
   * - `extend`
     - Adds all elements of a list to another list.
     - ``numbers.extend([6, 7, 8])``
   * - `insert`
     - Inserts an item at a specified position in a list.
     - ``numbers.insert(0, 1)``
   * - `index`
     - Returns the index of the first occurrence of a value in a list.
     - ``index = numbers.index(2)``
   * - `loc`
     - Returns label-based indexer.
     - ``print(df.loc[[1, 3]])``
   * - `for in (for loop)`
     - Iterates over a sequence.
     - ``for item in my_list: print(item)``
   * - `for in range`
     - Iterates over a sequence of numbers.
     - ``for i in range(5): print(i)``
   * - `factorial`
     - Returns the factorial of a number.
     - ``import math; factorial = math.factorial(5)``
   * - `if else`
     - Executes a block of code if a condition is true, otherwise another block.
     - ``x = 10; result = "Positive" if x > 0 else "Non-positive"``
   * - `square/**`
     - Raises a number to the power of two.
     - ``square = 5 ** 2``
   * - `[ ]`
     - Creates a list or accesses elements of a list.
     - ``my_list = [1, 2, 3]; x = my_list[0]``
   * - `{ }`
     - Creates a dictionary or sets.
     - ``my_dict = {'a': 1, 'b': 2}; my_set = {1, 2, 3}``
   * - `key`
     - Accesses the value associated with a key in a dictionary.
     - ``value = my_dict['a']``
   * - `union`
     - Returns a set containing the union of two or more sets.
     - ``set1 = {1, 2, 3}; set2 = {3, 4, 5}; union_set = set1 | set2``
   * - `intersection`
     - Returns a set containing the intersection of two or more sets.
     - ``intersection_set = set1 & set2``
   * - `difference`
     - Returns a set containing the difference between two or more sets.
     - ``difference_set = set1 - set2``
   * - `subset`
     - Returns True if all elements of a set are present in another set.
     - ``is_subset = set1 <= set2``
   * - `superset`
     - Returns True if a set has all elements of another set.
     - ``is_superset = set1 >= set2``
   * - `close`
     - Closes a file.
     - ``file.close()``
   * - `write “w” ()`
     - Writes to a file (creates a new file if it does not exist).
     - ``with open("file.txt", "w") as f: f.write("Hello, World!")``
   * - `create “x” ()`
     - Creates a new file.
     - ``with open("file.txt", "x") as f: pass``
   * - `close ()`
     - Closes a file.
     - ``file.close()``
   * - `open ()`
     - Opens a file and returns a file object.
     - ``with open("file.txt", "r") as f: content = f.read()``
   * - `read “r” ()`
     - Reads from a file.
     - ``with open("file.txt", "r") as f: content = f.read()``
   * - `append “a” ()`
     - Appends to a file.
     - ``with open("file.txt", "a") as f: f.write("New line")``
   * - `readline ()`
     - Reads a single line from a file.
     - ``with open("file.txt", "r") as f: line = f.readline()``
   * - `\n`
     - Represents a newline character.
     - ``multiline_str = "Line 1\nLine 2"``
   * - `strip ()`
     - Removes leading and trailing whitespace from a string.
     - ``clean_str = "   Hello   ".strip()``
   * - `%d`
     - Format specifier for integer.
     - ``num = 5; print("Number: %d" % num)``
   * - `%x`
     - Format specifier for hexadecimal integer.
     - ``num = 10; print("Hexadecimal: %x" % num)``
   * - `with`
     - Simplifies exception handling by encapsulating common preparation and cleanup tasks.
     - ``with open("file.txt") as f: content = f.read()``
   * - `string`
     - Defines a string.
     - ``my_str = "Hello, World!"``
   * - `enumerate`
     - Returns an enumerate object.
     - ``for i, value in enumerate(['a', 'b', 'c']): print(i, value)``
   * - `break`
     - Terminates the loop statement and transfers execution to the statement immediately following the loop.
     - ``for i in range(10): if i == 5: break``
   * - `binary mode`
     - Opens a file in binary mode.
     - ``with open("file.bin", "wb") as f: f.write(b'binary data')``
   * - `split`
     - Splits a string into a list.
     - ``words = "Hello World".split()``
   * - `join`
     - Joins elements of a list into a string.
     - ``sentence = " ".join(words)``
   * - `for in range`
     - Iterates over a sequence of numbers.
     - ``for i in range(5): print(i)``
   * - `range(start, end, step)`
     - Generates a sequence of numbers with a specified start, end, and step.
     - ``for i in range(1, 10, 2): print(i)``
   * - `isinstance`
     - Returns True if the specified object is of the specified type.
     - ``is_num = isinstance(5, int)``
   * - `sorted`
     - Returns a new sorted list from the elements of any iterable.
     - ``sorted_list = sorted([3, 1, 2])``
   * - `bool`
     - Converts a value to a boolean.
     - ``b = bool(1)``
   * - `if`
     - Executes a block of code if a condition is true.
     - ``if x > 0: print("Positive")``
   * - `if-elif`
     - Checks another condition if the preceding `if` condition is false.
     - ``if x > 0: print("Positive") elif x == 0: print("Zero")``
   * - `if-else`
     - Executes a block of code if a condition is true, otherwise another block.
     - ``x = 10; result = "Positive" if x > 0 else "Non-positive"``
   * - `else`
     - Executes a block of code if the preceding `if` condition(s) are false.
     - ``if x > 0: print("Positive") else: print("Non-positive")``
   * - `elif`
     - Checks another condition if the preceding `if` condition is false.
     - ``if x > 0: print("Positive") elif x == 0: print("Zero")``
   * - `weather forecast`
     - Provides weather information.
     - ``weather_forecast = {"temperature": 25, "conditions": "sunny"}``
   * - `for`
     - Iterates over a sequence.
     - ``for item in my_list: print(item)``
   * - `break`
     - Terminates the loop statement and transfers execution to the statement immediately following the loop.
     - ``for i in range(10): if i == 5: break``
   * - `continue`
     - Skips the rest of the loop and continues with the next iteration.
     - ``for i in range(10): if i == 5: continue``
   * - `else in for loop`
     - Executes a block of code when the loop is finished executing.
     - ``for i in range(3): print(i) else: print("Finished")``
   * - `nested`
     - A loop inside another loop.
     - ``for i in range(3): for j in range(2): print(i, j)``
   * - `nested loop`
     - A loop inside another loop.
     - ``for i in range(3): for j in range(2): print(i, j)``
   * - `def`
     - Defines a function.
     - ``def greet(): print("Hello")``
   * - `return`
     - Exits a function and returns a value.
     - ``def add(a, b): return a + b``
   * - `info`
     - Provides a concise summary of a DataFrame.
     - ``data.info()``
   * - `shape`
     - Returns a tuple representing the dimensionality of a DataFrame.
     - ``shape = data.shape``
   * - `head`
     - Returns the first n rows of a DataFrame.
     - ``top_rows = data.head()``
   * - `tail`
     - Returns the last n rows of a DataFrame.
     - ``bottom_rows = data.tail()``
   * - `.columns`
     - Returns the column labels of a DataFrame.
     - ``columns = data.columns``
   * - `.index()`
     - Returns the index labels of a DataFrame.
     - ``index = data.index``
   * - `.describe()`
     - Generates descriptive statistics of a DataFrame.
     - ``stats = data.describe()``
   * - `.iloc`
     - Purely integer-location based indexing for selection by position.
     - ``data.iloc[1]``
   * - `data.iloc[1]`
     - Selects a specific row in a DataFrame by index location.
     - ``row = data.iloc[1]``
   * - `data.iloc[:, 0]`
     - Selects a specific column in a DataFrame by index location.
     - ``column = data.iloc[:, 0]``
   * - `.copy()`
     - Creates a copy of a DataFrame.
     - ``data_copy = data.copy()``
   * - `.concat()`
     - Concatenates two or more DataFrames.
     - ``combined_data = pd.concat([data1, data2])``
   * - `.dropna()`
     - Removes rows or columns with missing values (NaN).
     - ``clean_data = data.dropna()``
   * - `.mean()`
     - Computes the mean of numeric columns in a DataFrame.
     - ``avg = data.mean()``
   * - `.rename()`
     - Renames columns or index labels of a DataFrame.
     - ``data.rename(columns={'A': 'a', 'B': 'b'})``
   * - `.plot()`
     - Plots the data in a DataFrame.
     - ``data.plot()``
   * - `correlation_matrix`
     - Displays a correlation matrix.
     - ``corr_matrix = data.corr()``
   * - `annot`
     - Annotates the cells of a heatmap or other plot.
     - ``sns.heatmap(corr_matrix, annot=True)``
   * - `cmap`
     - Specifies the colormap for a plot.
     - ``sns.heatmap(corr_matrix, cmap='coolwarm')``
   * - `fmt`
     - Formats the text or numbers in a plot.
     - ``sns.heatmap(corr_matrix, fmt='.2f')``
   * - `.idxmax`
     - Returns the index of the first occurrence of the maximum value.
     - ``max_index = data['column'].idxmax()``
   * - `subplot`
     - Creates a subplot in a plot.
     - ``plt.subplot(1, 2, 1)``
   * - `countplot`
     - Shows the counts of observations in each categorical bin.
     - ``sns.countplot(x='column', data=data)``
   * - `kind`
     - Specifies the type of plot to be created.
     - ``data.plot(kind='scatter', x='A', y='B')``
   * - `bbox_to_anchor`
     - Specifies the bounding box of a legend.
     - ``plt.legend(bbox_to_anchor=(1.05, 1))``
   * - `plot.map`
     - Maps a function to each element of a plot.
     - ``sns.pairplot(data.map(func))``
   * - `map`
     - Applies a function to each element of a series or DataFrame.
     - ``data['column'].map(func)``
   * - `correlation`
     - Measures the strength and direction of the linear relationship between two variables.
     - ``corr = data['A'].corr(data['B'])``
   * - `matrix`
     - Represents a matrix.
     - ``matrix = [[1, 2], [3, 4]]``
   * - `K-Nearest Neighbors (KNN)`
     - A supervised machine learning algorithm used for classification and regression.
     - ``from sklearn.neighbors import KNeighborsClassifier``
   * - `import pandas as pd`
     - Import the pandas library for data manipulation.
     - `import pandas as pd`
   * - `pd.read_csv`
     - Read a comma-separated values (CSV) file into a DataFrame.
     - `iris_dataset = pd.read_csv('/path/to/file.csv')`
   * - `print`
     - Print the specified message to the console.
     - `print("Hello, World!")`
   * - `iris_dataset.head`
     - Return the first n rows of the DataFrame.
     - `print(iris_dataset.head())`
   * - `iris_dataset.isnull`
     - Detect missing values in the DataFrame.
     - `missing_values = iris_dataset.isnull().sum()`
   * - `iris_dataset.nunique`
     - Count unique values in each column.
     - `unique_counts = iris_dataset.nunique()`
   * - `iris_dataset['species'].value_counts`
     - Return a Series containing counts of unique values.
     - `species_distribution = iris_dataset['species'].value_counts()`
   * - `iris_dataset.skew`
     - Return the skewness of each numeric column.
     - `skewness = iris_dataset.skew()`
   * - `iris_dataset.kurt`
     - Return the kurtosis of each numeric column.
     - `kurtosis = iris_dataset.kurt()`
   * - `shapiro`
     - Perform the Shapiro-Wilk test for normality.
     - `stat, p = shapiro(iris_dataset['column'])`
   * - `StandardScaler`
     - Standardize features by removing the mean and scaling to unit variance.
     - `scaler = StandardScaler()` 
   * - `PolynomialFeatures`
     - Generate a new feature matrix consisting of all polynomial combinations.
     - `poly = PolynomialFeatures(degree=2)`
   * - `PCA`
     - Perform Principal Component Analysis.
     - `pca = PCA(n_components=2)`
   * - `train_test_split`
     - Split arrays or matrices into random train and test subsets.
     - `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`
   * - `LogisticRegression`
     - Perform logistic regression.
     - `model = LogisticRegression()`
   * - `accuracy_score`
     - Compute the accuracy classification score.
     - `accuracy = accuracy_score(y_true, y_pred)`
   * - `KNeighborsClassifier`
     - Classify using k-nearest neighbors.
     - `knn = KNeighborsClassifier(n_neighbors=5)`
   * - `DecisionTreeClassifier`
     - Build a decision tree classifier.
     - `tree = DecisionTreeClassifier()`
   * - `RandomForestClassifier`
     - Build a random forest classifier.
     - `forest = RandomForestClassifier(n_estimators=100)`
   * - `SVC`
     - Perform support vector classification.
     - `svm = SVC(kernel='linear')`
   * - `cross_val_score`
     - Evaluate a score by cross-validation.
     - `cv_scores = cross_val_score(model, X, y, cv=5)`
   * - `GridSearchCV`
     - Perform grid search with cross-validation for hyperparameter tuning.
     - `grid = GridSearchCV(SVC(), param_grid, refit=True)`
   * - `confusion_matrix`
     - Compute confusion matrix to evaluate accuracy.
     - `cm = confusion_matrix(y_true, y_pred)`
   * - `ConfusionMatrixDisplay`
     - Plot the confusion matrix.
     - `disp = ConfusionMatrixDisplay(confusion_matrix=cm)`
   * - `classification_report`
     - Generate a classification report.
     - `report = classification_report(y_true, y_pred)`
   * - `roc_curve`
     - Compute Receiver Operating Characteristic (ROC).
     - `fpr, tpr, _ = roc_curve(y_true, y_score)`
   * - `auc`
     - Compute Area Under the Curve (AUC) for ROC.
     - `roc_auc = auc(fpr, tpr)`
   * - `label_binarize`
     - Binarize labels in a one-vs-all fashion.
     - `y_bin = label_binarize(y, classes=[0, 1, 2])`
   * - `OneVsRestClassifier`
     - One-vs-the-rest (OvR) multiclass strategy.
     - `classifier = OneVsRestClassifier(SVC())`
   * - `cycle`
     - Cycle through an iterable indefinitely.
     - `colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])`
   * - `plt.figure`
     - Create a new figure.
     - `plt.figure()`
   * - `plt.plot`
     - Plot y versus x as lines and/or markers.
     - `plt.plot(fpr, tpr, label='ROC curve')`
   * - `plt.xlim`
     - Set the x-axis view limits.
     - `plt.xlim([0.0, 1.0])`
   * - `plt.ylim`
     - Set the y-axis view limits.
     - `plt.ylim([0.0, 1.05])`
   * - `plt.xlabel`
     - Set the label for the x-axis.
     - `plt.xlabel('False Positive Rate')`
   * - `plt.ylabel`
     - Set the label for the y-axis.
     - `plt.ylabel('True Positive Rate')`
   * - `plt.title`
     - Set the title of the current axes.
     - `plt.title('Receiver Operating Characteristic')`
   * - `plt.legend`
     - Place a legend on the axes.
     - `plt.legend(loc='lower right')`
   * - `plt.savefig`
     - Save the current figure.
     - `plt.savefig('/path/to/figure.png')`
   * - `plt.show`
     - Display all open figures.
     - `plt.show()`
   * - `KMeans`
     - Perform K-Means clustering.
     - `kmeans = KMeans(n_clusters=3)`
   * - `Missing Value Analysis`
     - Check for missing values in the dataset.
     - `missing_values = iris_dataset.isnull().sum()`
   * - `Unique Value Counts`
     - Count the number of unique values in each column.
     - `unique_counts = iris_dataset.nunique()`
   * - `Species Distribution`
     - Calculate the distribution of each species in the dataset.
     - `species_distribution = iris_dataset['species'].value_counts()`
   * - `Skewness and Kurtosis`
     - Calculate skewness and kurtosis for each feature.
     - `skewness = iris_dataset.skew(); kurtosis = iris_dataset.kurt()`
   * - `Normality Test`
     - Perform a normality test (Shapiro-Wilk test) on each feature.
     - `stat, p = shapiro(iris_dataset['column'])`
   * - `Feature Scaling`
     - Scale the features using StandardScaler.
     - `scaler = StandardScaler(); scaled_features = scaler.fit_transform(iris_dataset)`
   * - `Feature Engineering: Polynomial Features`
     - Create polynomial features to increase model complexity.
     - `poly = PolynomialFeatures(degree=2); poly_features = poly.fit_transform(iris_dataset)`
   * - `Principal Component Analysis (PCA)`
     - Reduce dimensionality using PCA and explain variance.
     - `pca = PCA(n_components=2); pca_components = pca.fit_transform(iris_dataset)`
   * - `Logistic Regression`
     - Build a logistic regression model to classify species.
     - `model = LogisticRegression(); model.fit(X_train, y_train)`
   * - `K-Nearest Neighbors (KNN)`
     - Build and evaluate a KNN classifier.
     - `knn = KNeighborsClassifier(n_neighbors=5); knn.fit(X_train, y_train)`
   * - `Decision Tree Classifier`
     - Build and evaluate a decision tree classifier.
     - `tree = DecisionTreeClassifier(); tree.fit(X_train, y_train)`
   * - `Random Forest Classifier`
     - Build and evaluate a random forest classifier.
     - `forest = RandomForestClassifier(n_estimators=100); forest.fit(X_train, y_train)`
   * - `Support Vector Machine (SVM)`
     - Build and evaluate an SVM classifier.
     - `svm = SVC(kernel='linear'); svm.fit(X_train, y_train)`
   * - `Cross-Validation`
     - Perform cross-validation to evaluate model performance.
     - `cv_scores = cross_val_score(model, X, y, cv=5)`
   * - `Hyperparameter Tuning: Grid Search`
     - Perform grid search for hyperparameter tuning.
     - `grid = GridSearchCV(SVC(), param_grid, refit=True); grid.fit(X_train, y_train)`
   * - `Confusion Matrix`
     - Generate a confusion matrix to evaluate classification performance.
     - `cm = confusion_matrix(y_true, y_pred); disp = ConfusionMatrixDisplay(confusion_matrix=cm)`
   * - `Classification Report`
     - Generate a classification report with precision, recall, and F1-score.
     - `report = classification_report(y_true, y_pred)`
   * - `Feature Importance`
     - Calculate and display feature importance from a tree
     - `importances = model.feature_importances_; plt.barh(range(len(importances)), importances)`
   * - `ROC Curve and AUC`
     - Plot the ROC curve and calculate the AUC for model evaluation.
     - `fpr, tpr, _ = roc_curve(y_true, y_score); roc_auc = auc(fpr, tpr)`
   * - `Multiclass ROC Curve`
     - Plot ROC curves for multiclass classification problems.
     - `colors = cycle(['aqua', 'darkorange', 'cornflowerblue']); for i, color in zip(range(n_classes), colors): plt.plot(fpr[i], tpr[i], color=color)`
   * - `Clustering with K-Means`
     - Perform K-Means clustering and visualize clusters.
     - `kmeans = KMeans(n_clusters=3); kmeans.fit(X); plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)`
