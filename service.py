import bentoml
from bentoml.io import PandasDataFrame, JSON, NumpyNdarray
import pandas as pd
import numpy as np

model_runner = bentoml.picklable_model.get("cardiovascular_knn:latest").to_runner()

service = bentoml.Service("cardiovascular_knn", runners=[model_runner])


@service.api(input=NumpyNdarray(), output=JSON())
def classify(input_value: np.ndarray) -> np.ndarray:
    model_input=pd.DataFrame(input_value)
    result = model_runner.run(model_input.T)

    print(result)

    return result
