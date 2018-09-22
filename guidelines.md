# Deeplodocus coding guide

## Source File Encoding

TBC

## Maximum Line Length

- All lines are limited to 79 characters

    The preferred way of wrapping long lines is by using Python's implied line continuation inside parentheses, brackets and braces. 
    Long lines can be broken over multiple lines by wrapping expressions in parentheses. 
    These should be used in preference to using a backslash for line continuation.
    
    Backslashes may still be appropriate at times. For example, long, multiple with-statements cannot use implicit continuation, so backslashes are acceptable:

    ```buildoutcfg
    with open('/path/to/some/file/you/want/to/read') as file_1, \
         open('/path/to/some/file/being/written', 'w') as file_2:
        file_2.write(file_1.read())
    ```
    This way it is easier to see which items are added and which are subtracted.
    
## Blank Lines

- Top-level function and class definitions are surrounded by two blank lines
- Method definitions are surrounded by a single blank line
- Blank lines may be used sparingly to separate groups of related functions


## Imports

- Separate packages should be imported on separate lines

    Yes:
    
    ```buildoutcfg
    import os
    import sys
    ```
    
    No:
    ```buildoutcfg
    import os, sys
    ```
    
- Its okay to import multiple modules from a package on one line
    
    ```buildoutcfg
    from os import mkdir, listdir
    ```

- Imports should be grouped in the following order:
    1. Standard library imports
    2. Related third party imports
    3. Local application/library specific imports
   
    Each group should be separated by a single blank line

- Absolute imports are preferred:

    ```buildoutcfg
    from deeplodocus.utils import example
    ```

- But relative imports are acceptable when dealing with complex package layouts where absolute imports become unnecessarily verbose  

    ```buildoutcfg
    from . import example
    ```


## Line Breaks at Binary Operators

- Lines should break before binary operators

    ```buildoutcfg
    # Yes: easy to match operators with operands
    income = (gross_wages
              + taxable_interest
              + (dividends - qualified_dividends)
              - ira_deduction
              - student_loan_interest)
    ```
    
    ```buildoutcfg
    # No: operators sit far away from their operands
    income = (gross_wages +
              taxable_interest +
              (dividends - qualified_dividends) -
              ira_deduction -
              student_loan_interest)
    ```

## Module Level Dunder Names

- Module level 'dunders' such ad __author__, __version__ should be placed after the module doc string, but before any import statements *except* from __future__ imports

    ```buildoutcfg
    """This is the example module.
    
    This module does stuff.
    """
    
    from __future__ import barry_as_FLUFL
    
    __all__ = ['a', 'b', 'c']
    __version__ = '1.3'
    __author__ = 'Professor Plum'
    
    import os
    import sys
    ```

## String Quotes

- Use double quotes, "foobar" not single quotes, 'foobar'

## Comments

- Always make a priority of keeping the comments up-to-date when the code changes.
- Comments should be complete sentences.
- The first word should be capitalized, unless it is an identifier that begins with a lower case letter (never alter the case of identifiers).
- Block comments generally consist of one or more paragraphs built out of complete sentences, with each sentence ending in a period.

### Block Comments

- Block comments generally apply to some (or all) code that follows them, and are indented to the same level as that code. 
- Each line of a block comment starts with a # and a single space.
- Paragraphs inside a block comment are separated by a line containing a single #

### Inline Comments
- An inline comment is on the same line as the code statement.
- Use inline comments sparingly.
- Do not use inline comments to state the obvious:
    
    Yes:

    ```buildoutcfg
    x += 1        # Compensate for boarder
    ```

    No:
    ```buildoutcfg
    x += 1         # Increment x
    ```
    
### Documentation Strings

- Write docstrings for all modules, functions, classes, and methods.

Yes:
    
    ```buildoutcfg
    def load_data(path)
    """
    Loads data from the given directory path into a list of file paths.
    :param path: str: path to the directory containing data files
    :return: list of str: path to each data file
    """
    ```
    
## Naming Conventions

- Never use l (lower case L), O, o (lower or upper case o), or I (upper case i) as single character variable names.
- Use the CapWords convention for class names.
- Use lower case with underscores for function and variable names.
- Constants are defined on a module level and written in all capitals with underscores separating words.
- Private attributes should have one leading underscore.
- Public attributes should have not leading underscores.
- Private methods should have two leading underscores.
- Public methods should have no leading underscores.

## Strings

- Use %s, %i or %f when inserting variables into a string:

    Yes: 
    
    ```buildoutcfg
    "The frame rate is: "%.2f" % frame_rate
    ```
    
    No:
    ```buildoutcfg
    "The frame rate is: " + str(frame_rate)
    ```

- Use string methods instead of the string module.
- Use ''.startswith() and ''.endswith() instead of string slicing to check for prefixes or suffixes.
    
    Yes:
    ```buildoutcfg
    if foo.startswith('bar'):
    ```
    
    No:
    ```buildoutcfg
    if foo[:3] == 'bar':
    ```

## If Statements

### Object Type Comparisons

- Object type comparisons should always use isinstance() instead of comparing types directly.

    Yes: 

    ```buildoutcfg
    if isinstance(obj, int):
    ```
    
    No:
    ```buildoutcfg
    if type(obj) is int:
    ```

### Boolean Values

- Don't compare boolean values to True or False using ==

   Yes:
    
    ```buildoutcfg
    if greeting:
    ```
    
   No:
   
    ```buildoutcfg
    if greeting == True:
    ```
    
   Worse:
   
    ```buildoutcfg
    if greeting is True:
    ``` 

## Try Except

- Always specify the specific exception where possible:

    Yes:
    ```buildoutcfg
    try:
        a = b / c
    except ValueError:
        a = 0
    ```
    
    No:
    ```buildoutcfg
    try:
        a = b / c
    except:
        a = 0
    ```
    
- When binding caught exceptions to a name, prefer the explicit name binding syntax added in Python 2.6:

    ```buildoutcfg
    try:
        process_data()
    except Exception as exc:
        raise DataProcessingFailedError(str(exc))

    ```
- Limit the try clause to the absolute minimum amount of code necessary, this avoids masking bugs:

    Yes:
    ```buildoutcfg
    try:
        value = collection[key]
    except KeyError:
        return key_not_found(key)
    else:
        return handle_value(value)
    ```
    
    No:
    ```buildoutcfg
    try:
        # Too broad!
        return handle_value(collection[key])
    except KeyError:
        # Will also catch KeyError raised by handle_value()
        return key_not_found(key)
    ```


## Notification, Warning and Error Messages

- Only raise Errors manually when the resultant message is more informative than would be otherwise be given.
- Only raise Errors manually if you are 120% certain that the given information will always be correct.

If something is not covered in this document, please refer to the link below for further details: 

https://www.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds
