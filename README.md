# mnist-example feature/tests
```
========================================================================= test session starts ==========================================================================
platform linux -- Python 3.6.9, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
rootdir: /mnt/d/IIT JODHPUR/Semester-1/ML-Ops/mnist-example/tests
collected 2 items

test_utils.py .F                                                                                                                                                 [100%]

=============================================================================== FAILURES ===============================================================================
_______________________________________________________________________ test_small_data_overfit ________________________________________________________________________

    def test_small_data_overfit():
        iris = load_iris()
        random_indice = np.random.permutation(len(iris.data))
        X = iris.data[random_indice[:10]]
        y = iris.target[random_indice[:10]]
        acc = run_classification_experiment(X, y, X, y, .001, 1)
>       assert acc > 0.99
E       assert 0.5 > 0.99

test_utils.py:22: AssertionError
======================================================================= short test summary info ========================================================================
FAILED test_utils.py::test_small_data_overfit - assert 0.5 > 0.99
===================================================================== 1 failed, 1 passed in 2.97s ======================================================================
```
