Welcome to the repository for "Derivative-Based Neural Modelling of Cumulative
Distribution Functions for Survival Analysis", to appear at AISTATS 2022.

## Getting started

Create a virtual environment and activate it:
```
python -m venv desurv_env
source desurv_env/bin/activate
```

Install dependencies:
```
python -m pip install -r requirements.txt
```

Access the source code:
```
cd src
```

Use DeCDF for CDF/density estimation:
```
python -m jupyter notebook fig2.ipynb
```

Train a DeSurv model:
```
python desurv.py
```

### Additional info:

The core code components of DeCDF/DeSurv can be found in `classes.py`.
The `ODESurvSingle` class implements a DeSurv model for the single-risk setting.
The `ODESurvMultiple` class implements a DeSurv model for the competing-risk setting.
`desurv.py` illustrates how a DeSurv model can be trained and is designed to be easily
adaptable to a typical user's problem setting. `fig2.ipynb` illustrates the basic concept
behind DeCDF and hence DeSurv in an interactive setting.

### Acknowledgements

We would like to acknowledge the [pycox](https://github.com/havakv/pycox) library of
Kvamme et al. for its excellent dataset, model and evaluation provisions which
proved very useful when undertaking this project.
