### Model Explanation Capabilities

Model Evaluation Service (MES) exposes various capabilities to aid in
understanding the results of a black-box model. Model explanations
can be of two types:
* Local Explanations and,
* Global Explanations.

#### Local Explanations

Local explanations involve understanding and explaining the model
results for any single data point. For example, for a loan applicant
classification model, local explanation could imply understanding
the reasons for declaring an applicant as high-risk.

##### Usage

Local explanation can be made available through implementations in
packages such as Oracle MLX, LIME or Skater. User needs to encapsulate
the functionality provided by such packages and expose it through
specific methods in the encapsulating class.

An example template would be:
```python
class LocalExplainer:

    def __init__(self):
        """
        Code to initialize local explainer.
        """

    def explanation(self, X):
        """
        Code to generate explanations for the record X in parameters
        """
```

Such an implementation can then be used from an environment like
Jupyter notebook. The implementation logic for generating local
explanations can be changed and the consumer code for the above
mentioned class will be able to use it without any changes.

#### Global Explanations

Global explanations involve explaining the over-all behavior of a
model. Examples of global explanations include determination of
importance of features as well as partial dependence plots which
describe how strong of an influence a subset of features have on the
response variable.

##### Usage

As with the local explanations, the global explanations can be made
available through implementations in packages such as Oracle MLX,
Skater etc. User needs to encapsulate the functionality provided by
such packages and expose it through specific methods in the
encapsulating class.

An example template would be:
```python
class GlobalExplainer:

    def __init__(self):
        """
        Initialize the global explainer
        """

    def compute_feature_importance(self, X):
        """
        Logic to compute feature importance for records in sample X
        """
    def compute_partial_dependence(self, X):
        """
        Logic to compute partial dependence among the variables of
        a sample X
        """
```

Again, similar to local explanations, such an implementation can then
be used from an environment like Jupyter notebook. The implementation
logic for generating global explanations can be changed and the
consumer code for the above mentioned class will be able to use it
without any changes.
