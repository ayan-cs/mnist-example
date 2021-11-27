# Final Exam

The best hyperparameter is 0.1 because it gets best generalization capability. For higher values of hyperparameter, it overfits training data and for lower values, it underfits the same.

Dockerfile

```
FROM ubuntu:18.04
COPY mnist-code /exp/mnist-example
COPY /requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3-pip && apt-get clean
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["python3", "./mnist-example/q2.py"]
```
