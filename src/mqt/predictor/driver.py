from mqt.predictor import ml


def compile(qc, model="ML"):
    if model == "ML":
        predictor = ml.Predictor()
        prediction = predictor.predict(qc)
        return predictor.compile_predicted_compilation_path(qc, prediction)
