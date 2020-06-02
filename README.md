# brain-connectivity-analysis

This repository contains the methodes that I have developped during my 2020
internship in the Empenn team.

## Organization of the Repository
The folder `wasserstein_analysis` contains code demonstrating the use of
the Wasserstein distance, as well as the barycenter computation.

* `wasserstein_analysis/demonstration.py` demonstrates the basic functions that I implemented.
* *more files to come, implementing the experiments that I conducted.*

## Data
This code is designed to analyze data collected from the LONGIDEP study. In
this repository I provide some synthetical data in order to show how the
code interacts (as in: *loads*, *displays*, *splits*, etc) with data. Those
matrices, provided in the `fake_data/`, folder are not relevant for any
kind of clinical study, and are just random matrices that somehow mimic the
structure of real connectivity matrices.

## Technical details to run this code

### Environment and dependancies

The best way to run this code is to setup a Python3 virtual environment.

```
virtalenv env
```

Then one can source it as follows.

```
source env/bin/activate
```

Install all dependancies:

```
pip install -r requierements.txt
```

### Notebooks

Because pure notebook versioning is somehow a mess, they are handled using
the `jupytext` extension (this should have been installed via `pip` if the
above procedure has been followed). A notebook can be created from the `.py`
light python script files using `jupytext --to notebook python_file.py`.

In order to get `jupyter` to recognize the extension and behave well with it,
also run once
```
jupyter labextension install jupyterlab-jupytext
```

One can generate all the notebooks from a specific folder using (fish shell script...)

```
for file in *.py; jupytext --to notebook $file; end
```

Note that light python scripts are correct python scripts, and they can be
run without using a notebook.


## Code is getting cleaned, and purged of non-releaseable data, and will appear here during the first week of June
