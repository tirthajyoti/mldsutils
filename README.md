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
                        runtime=True,
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
      <td>0.970667</td>
      <td>0.946292</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.978378</td>
      <td>0.961637</td>
      <td>0.661074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.971576</td>
      <td>0.947891</td>
      <td>0.644068</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.992556</td>
      <td>0.956743</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.995000</td>
      <td>0.948980</td>
      <td>0.646362</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.979275</td>
      <td>0.939314</td>
      <td>0.648649</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.978417</td>
      <td>0.949062</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.984456</td>
      <td>0.931217</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.961240</td>
      <td>0.963351</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.976526</td>
      <td>0.958115</td>
      <td>0.665552</td>
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
      <td>0.978809</td>
      <td>0.010089</td>
      <td>0.961240</td>
      <td>0.972814</td>
      <td>0.978398</td>
      <td>0.983161</td>
      <td>0.995000</td>
    </tr>
    <tr>
      <th>Support Vector Machine with Linear Kernel</th>
      <td>10.0</td>
      <td>0.950260</td>
      <td>0.010063</td>
      <td>0.931217</td>
      <td>0.946691</td>
      <td>0.949021</td>
      <td>0.957772</td>
      <td>0.963351</td>
    </tr>
    <tr>
      <th>Support Vector Machine with RBF Kernel</th>
      <td>10.0</td>
      <td>0.326570</td>
      <td>0.344294</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.322034</td>
      <td>0.648077</td>
      <td>0.665552</td>
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
      <td>0.978</td>
      <td>23.525</td>
      <td>142.145</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000</td>
      <td>26.323</td>
      <td>138.665</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.978</td>
      <td>26.365</td>
      <td>135.692</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.976</td>
      <td>26.403</td>
      <td>113.238</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.978</td>
      <td>23.444</td>
      <td>123.922</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000</td>
      <td>21.479</td>
      <td>150.983</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000</td>
      <td>21.483</td>
      <td>127.956</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.000</td>
      <td>26.400</td>
      <td>132.321</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000</td>
      <td>22.457</td>
      <td>123.329</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.981</td>
      <td>22.422</td>
      <td>119.792</td>
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
      <td>0.4891</td>
      <td>0.515558</td>
      <td>0.000</td>
      <td>0.00000</td>
      <td>0.4880</td>
      <td>0.97800</td>
      <td>0.981</td>
    </tr>
    <tr>
      <th>Support Vector Machine with Linear Kernel</th>
      <td>10.0</td>
      <td>24.0301</td>
      <td>2.124234</td>
      <td>21.479</td>
      <td>22.43075</td>
      <td>23.4845</td>
      <td>26.35450</td>
      <td>26.403</td>
    </tr>
    <tr>
      <th>Support Vector Machine with RBF Kernel</th>
      <td>10.0</td>
      <td>130.8043</td>
      <td>11.377251</td>
      <td>113.238</td>
      <td>123.47725</td>
      <td>130.1385</td>
      <td>137.92175</td>
      <td>150.983</td>
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

