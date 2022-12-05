import argparse

from mqt.predictor import ML_utils
from mqt.predictor.RL_Predictor import ML_Predictor, RL_Predictor


def compile(qc, model="ML"):
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
        qc_compiled, device = RL_predictor.predict(qc)
        return qc_compiled, device

    else:
        raise ValueError("Choose between 'ML' and 'RL' Model.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MQT Predictor")
    # parser.add_argument("--timeout", type=int, default=120)
    # args = parser.parse_args()
