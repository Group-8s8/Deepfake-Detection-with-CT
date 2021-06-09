# Deepfake-Detection-with-CT
-----------------------------

* This is the main program. When you commit something, do so in another branch. I will review and pull them later.
* Everything I do is commented on where needed. read them. 

### Notations and stuff.
-------------------------
* Single underscore `_` before a variable means that the variable should be treated privately by the programmer reading the code. It doesn't have a meaning for python.
* User proper class names and functions. Class names should not have issue with python modules.
* Comment where possible.
* Write codes into `/utils`. except main.

## How to run?
---------------

* Run `pip install -r requirements.txt`.
* Run `python run.py -f <path_to_fake_image> -r <path_to_real_image>`
	* `-r` argument is optional.

## ToDo.
--------

- Optimize the code.