# Data analysis project template

ctrl+shift+v to toggle between markup and plain text

## Starting a new project from this template

This repo is a GitHub template repository. To start a new analysis project:

1. On GitHub, click **Use this template -> Create a new repository**.
2. Pick the destination (e.g. the NHS GitHub org) and a name for the new project.
3. Clone the new repo locally and follow the setup steps below.
4. Drop your data into `data/` (git-ignored, so it never gets committed) and start from `notebooks/data_exploration.ipynb`.

## Tools available

| Package    | Version | Why                                                        |
|------------|---------|-------------------------------------------------------------|
| duckdb     | 1.5.5   | SQL engine for querying CSV/dataframes directly              |
| pandas     | 3.0.5   | dataframes                                                   |
| pyreadr    | 0.5.6   | reads/writes R .rda/.rds files (the compiled one)             |
| openpyxl   | 3.1.5   | Excel .xlsx read/write engine pandas calls under the hood     |
| matplotlib | 3.11.1  | plotting                                                     |
| jupyterlab | 4.6.3   | the notebook UI                                              |
| pytest     | 9.1.1   | test runner                                                  |
| python-stdnum | 2.2  | NHS number checker                                           |

## Make the virtual environment from project root
python3 -m venv .venv

## Activate virtual environment
source .venv/bin/activate

## Install requirements
pip install -r requirements.txt

## Deactivate virtual environment
deactivate

## Start jupyter notebook
jupyter lab
or
nohup jupyter lab > jupyter.log 2>&1 &
disown

## Find what jupyter servers are running
jupyter server list

## Stop Jupyter process
jupyter lab stop [pid]
