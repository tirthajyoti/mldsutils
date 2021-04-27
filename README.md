# `mldsutils`: A machine-learning and data science utility package



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
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
```

### Classifiers and their names

```python
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=10),]

clf_names = ['k-Nearest Neighbors(3)',
             'Support Vector Machine with Linear Kernel',
            'Support Vector Machine with RBF Kernel']
```

### Some data

```python
X1, y1 = make_classification(n_features=20, n_samples=2000,n_redundant=0, n_informative=20,
                             n_clusters_per_class=1)
```

### Run
Note, you will get back a Pandas DataFrame from this

```python
d1,d2 = run_classifiers(X1,y1,
                        clf_lst=classifiers,names = clf_names,
                        metric='f1',verbose=True)
```

    Finished 10 runs for k-Nearest Neighbors(3) algorithm
    ---------------------------------------------------------------------------
    Finished 10 runs for Support Vector Machine with Linear Kernel algorithm
    ---------------------------------------------------------------------------
    Finished 10 runs for Support Vector Machine with RBF Kernel algorithm
    ---------------------------------------------------------------------------
    

## Examining the result

### Checking the dataframe of F1-scores

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
      <th>k-Nearest Neighbors(3)</th>
      <th>Support Vector Machine with Linear Kernel</th>
      <th>Support Vector Machine with RBF Kernel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.987013</td>
      <td>0.979021</td>
      <td>0.642857</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.987893</td>
      <td>0.972705</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.984293</td>
      <td>0.983213</td>
      <td>0.655462</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.984925</td>
      <td>0.972093</td>
      <td>0.644068</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.992908</td>
      <td>0.976636</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.992405</td>
      <td>0.967901</td>
      <td>0.009950</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.973761</td>
      <td>0.966507</td>
      <td>0.009804</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.977887</td>
      <td>0.990291</td>
      <td>0.648649</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.991870</td>
      <td>0.990521</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.987593</td>
      <td>0.974747</td>
      <td>0.264317</td>
    </tr>
  </tbody>
</table>
</div>



### Stats of the `d1` to compare algorithms

```python
d1.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>k-Nearest Neighbors(3)</th>
      <td>10.0</td>
      <td>0.986055</td>
      <td>0.006236</td>
      <td>0.973761</td>
      <td>0.984451</td>
      <td>0.987303</td>
      <td>0.990876</td>
      <td>0.992908</td>
    </tr>
    <tr>
      <th>Support Vector Machine with Linear Kernel</th>
      <td>10.0</td>
      <td>0.977364</td>
      <td>0.008442</td>
      <td>0.966507</td>
      <td>0.972246</td>
      <td>0.975691</td>
      <td>0.982165</td>
      <td>0.990521</td>
    </tr>
    <tr>
      <th>Support Vector Machine with RBF Kernel</th>
      <td>10.0</td>
      <td>0.287511</td>
      <td>0.320052</td>
      <td>0.000000</td>
      <td>0.002451</td>
      <td>0.137134</td>
      <td>0.643765</td>
      <td>0.655462</td>
    </tr>
  </tbody>
</table>
</div>



### Checking the dataframe of fitting/training time

```python
d2
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
      <th>k-Nearest Neighbors(3)</th>
      <th>Support Vector Machine with Linear Kernel</th>
      <th>Support Vector Machine with RBF Kernel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>19.529</td>
      <td>128.902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000</td>
      <td>17.581</td>
      <td>138.664</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000</td>
      <td>27.340</td>
      <td>142.638</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.977</td>
      <td>19.531</td>
      <td>125.963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.956</td>
      <td>19.528</td>
      <td>139.639</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000</td>
      <td>16.598</td>
      <td>124.063</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000</td>
      <td>15.672</td>
      <td>124.171</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.976</td>
      <td>20.506</td>
      <td>135.868</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000</td>
      <td>22.464</td>
      <td>152.332</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.000</td>
      <td>20.490</td>
      <td>123.982</td>
    </tr>
  </tbody>
</table>
</div>



```python
d2.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>k-Nearest Neighbors(3)</th>
      <td>10.0</td>
      <td>0.2909</td>
      <td>0.468427</td>
      <td>0.000</td>
      <td>0.00000</td>
      <td>0.000</td>
      <td>0.71700</td>
      <td>0.977</td>
    </tr>
    <tr>
      <th>Support Vector Machine with Linear Kernel</th>
      <td>10.0</td>
      <td>19.9239</td>
      <td>3.286444</td>
      <td>15.672</td>
      <td>18.06775</td>
      <td>19.530</td>
      <td>20.50200</td>
      <td>27.340</td>
    </tr>
    <tr>
      <th>Support Vector Machine with RBF Kernel</th>
      <td>10.0</td>
      <td>133.6222</td>
      <td>9.733031</td>
      <td>123.982</td>
      <td>124.61900</td>
      <td>132.385</td>
      <td>139.39525</td>
      <td>152.332</td>
    </tr>
  </tbody>
</table>
</div>



## Visualizing the results with the `plot_bars` function

Make sure to pass the correct titles of the plots. Otherwise, default strings will be plotted which may indicate wrong thing for your experiment.

```python
plot_bars(d1,t1="Mean F1 score of algorithms",
              t2="Std.dev of the F1 scores of algorithms")
```


![png](docs/images/output_20_0.png)


```python
plot_bars(d2,t1="Mean fitting time of algorithms",
              t2="Std.dev of the fitting time of algorithms")
```


![png](docs/images/output_21_0.png)

