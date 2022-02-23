# NOTICE

THIS PROJECT DOES NOT PRODUCE A VIABLE RESULT AS OF NOW AND NEEDS REWORKING. JUST USE IT AS A POC FOR THE ORIGINAL PAPER. NO GUARANTEE IS PROVIDED TO THE USERS OF THIS PROJECT.

# Deepfake-Detection-with-CT - Public
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
	* eg: `python run.py -f data/Fake/fake.png`.

* To increase accuracy. increase iteration at `utils/EM.py:29`

## Expected output with iteration = 1
-------------------------------------

![image](assets/output.png)

## ToDo.
--------

- Optimize the code.
