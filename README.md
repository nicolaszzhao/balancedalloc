# Assignment 2: Balanced allocation

> [!IMPORTANT]
> The code is tested for CPython 3.11.3 on Ubuntu 22.04.

## Project structure

- `src/notebook.ipynb`: Jupyter notebook showing all the generated plots.
- `src/driver.py`: Script that runs the experiments and generates the plots.
- `src/strategies.py`: Functions defining the different strategies of choosing
  bins.
- `src/experiment.py`: Functions for running the experiments.
- `report/`: Related Latex source files and assets for creating the project
  report.

## Instructions for reproducing the experiments

1. Create Python virtual environment

```sh
$ python3 -m venv balancedalloc-env
```

2. Activate virtual environment

```sh
# Bash
$ source balancedalloc-env/bin/activate

# cmd.exe 
C:\> balancedalloc-env\Scripts\activate.bat

# PowerShell
PS C:\> balancedalloc-env\Scripts\Activate.ps1
```

3. Install dependencies

```sh
$ cd src
$ pip install -r requirements.txt
```

4. Run the code in `src/notebook.ipynb` to visualize the plots or run the
  script `src/driver.py` to generate the plots as PDFs.

```sh
# Plot visualization
$ jupyter notebook notebook.ipynb

# Plot generation
$ python driver.py
```

5. Once done reproducing the experiment deactivate virtual environment by
   typing `deactivate` in your shell.
