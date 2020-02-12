# 2019ncov_prediction
Try to predict when the outbreak would be eliminated

Two models are in predict.py

1). "Growth": the Gompertz growth model

https://en.wikipedia.org/wiki/Gompertz_function

Y(x) = Ym*(Y0/YM)^exp(-K*X)


2). "Sigmoid": simple sigmoid function

Y(x) = YM/(1+exp(-b*(x-c)))


Gompertz growth would give "worse" situations of the outbreak than Sigmoid model
