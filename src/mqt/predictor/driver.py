from mqt.predictor.ml import helper
from mqt.predictor.ml.Predictor import Predictor


def compile(qc, model="ML"):
    if model == "ML":
        ML_predictor = Predictor()
        prediction = ML_predictor.predict(qc)
        qc_compiled = ML_predictor.compile_predicted_compilation_path(qc, prediction)
        device = helper.get_index_to_comppath_LUT()[prediction][
            1
        ]  # index '1' corresponds to device
        return qc_compiled, device
