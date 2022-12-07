from mqt.predictor.ml import helper
from mqt.predictor.ml.Predictor import Predictor


def compile(qc, model="ML"):
    if model == "ML":
        predictor = Predictor()
        prediction = predictor.predict(qc)
        return predictor.compile_predicted_compilation_path(qc, prediction)
