from mqt.predictor import ML_utils
from mqt.predictor.ML_Predictor import ML_Predictor
from mqt.predictor.RL_Predictor import RL_Predictor


def compile(qc, model="ML", opt_objective="fidelity"):
    if model == "ML":
        ML_predictor = ML_Predictor()
        prediction = ML_predictor.predict(qc)
        qc_compiled = ML_predictor.compile_predicted_compilation_path(qc, prediction)
        device = ML_utils.get_index_to_comppath_LUT()[prediction][
            1
        ]  # index '1' corresponds to device
        return qc_compiled, device
    elif model == "RL":
        RL_predictor = RL_Predictor()
        qc_compiled, device = RL_predictor.compile(qc, opt_objective=opt_objective)
        return qc_compiled, device

    else:
        raise ValueError("Choose between 'ML' and 'RL' Model.")
