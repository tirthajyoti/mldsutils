# mldsutils



This file will become your README and also the index of your documentation.

## Install

`pip install mldsutils`

## How to use

- Import the library
- Define a list of Scikit-learn estimators with your choice of hyperparameters
- Generate some synthetic data
- Run the `run_regressor` function to iterate through each of them and evaluate the given datatset

### Import

```python
from mldsutils.mldsutils import *
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
```

### Regressors and their names

```python
reg_names = ["Linear regression","L1 (LASSO) regression","Ridge regression"]
regressors = [LinearRegression(n_jobs=-1),
              Lasso(alpha=0.1),
              Ridge(alpha=0.1)]
```

### Some data

```python
X = np.random.normal(size=200)
y = 2*X+3+np.random.uniform(1,2,size=200) # A linear relationship with a small noise added
```

### Run
Note, you will get back a Pandas DataFrame from this

```python
d1 = run_regressors(X,y,regressors,metric='r2',runtime=False,verbose=True)
```

### Let's see the DataFrame

It shows the $R^2$ score for three estimators with 10 runs.

```python
d1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LinearRegression</th>
      <th>Lasso</th>
      <th>Ridge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.823522</td>
      <td>0.880346</td>
      <td>0.712075</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.926596</td>
      <td>0.937878</td>
      <td>0.975332</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.918350</td>
      <td>0.936040</td>
      <td>0.945131</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.972663</td>
      <td>0.862610</td>
      <td>0.942700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.977795</td>
      <td>0.929077</td>
      <td>0.966707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.929413</td>
      <td>0.969269</td>
      <td>0.954407</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.970386</td>
      <td>0.906072</td>
      <td>0.972059</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.957160</td>
      <td>0.978470</td>
      <td>0.945512</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.964562</td>
      <td>0.925020</td>
      <td>0.970284</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.962224</td>
      <td>0.967389</td>
      <td>0.967825</td>
    </tr>
  </tbody>
</table>
</div>


