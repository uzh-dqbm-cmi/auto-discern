from typing import Callable, Dict


class Predictor:

    def __init__(self, model: Callable, encoders: Dict, preprocess_func: Callable, transform_func: Callable):
        """
        Base class for a machine learning model predictor.

        Args:
            model: A trained model with a `predict` method.
            encoders: Dictionary of encoders to be used during feature transformation.
            transform_func: Function to transform input data into features for the model.
        """
        self.model = model
        self.preprocess_fun = preprocess_func
        self.encoders = encoders
        self.transform_func = transform_func

    def predict(self, data_point: Dict):
        data_processed = self.preprocess_fun(data_point)
        x, feature_cols = self.transform_func([data_processed], self.encoders)
        return self.model.predict(x)
