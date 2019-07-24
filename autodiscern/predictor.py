from typing import Any, Callable, Dict


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

    def predict(self, data_point: Dict[str, Any]):
        """
        Make a prediction.

        Args:
            data_point: A dictionary containing all the keys necessary to describe the data point.

        Returns: prediction value.

        """
        # wrap the data_point in a dict to mimic the structure expected by preprocess_func,
        #   which is a dict of data point dicts
        data_point['entity_id'] = 0
        wrapped_data_point = {0: data_point}
        wrapped_data_processed = self.preprocess_fun(wrapped_data_point)

        # unwrap the data point
        data_processed = wrapped_data_processed[0]

        # but then put it in a list, because that's what the transform_func expects :(
        rewrapped_data_processed = [data_processed]

        x, feature_cols = self.transform_func(rewrapped_data_processed, self.encoders)
        return self.model.predict(x)
