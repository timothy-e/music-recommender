# Virtual Env

In the directory, to set up the virtual environment:

```
python3 -m venv .venv
```

Then to use the environment, run

```
source .venv/bin/activate
```

or

```
.\.venv\bin\activate.bat
```

to enter the virtual environment.

Use

```
pip install -r requirements.txt
```

to get the requirements and

```
deactivate
```

to exit the virtual environment.

# Collaborative Dataset

Run

```
python collaborative.py [-s]
```

The number of `s`s in the flag indicate how small of a dataset we want. No `-s` is the whole dataset, `-s` is one tenth, `-ss` is one hundredth, ...
