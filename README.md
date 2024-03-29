# MLAdaBatch

## Here is the code for Muti label Batch selection
Adabatch train and baseline represent training using adaptive batch selection and model default batch selection, respectively.
### Describe:
model.py layer.py and util.py
Adaptivetrain.py(Adaptivetrain.ipy) and basetrain.py(basetrain.ipy) respectively consider training using the adaptive batch selection method or training using default batch selection method of the model.
### Use:
class CFG: Parameter configuration class,Help you configure the parameters of the basic model.

Adaptive parameter use:

warm_epoch:default=3

selection presure: [2,4,16,64]

just run Adaptivetrain.ipy



