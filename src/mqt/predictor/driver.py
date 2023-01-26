from mqt.predictor import ml, rl


def compile(qc, model="ML", opt_objective="fidelity"):
    if model == "ML":
        predictor = ml.Predictor()
        prediction = predictor.predict(qc)
        compiled_qc = predictor.compile_predicted_compilation_path(qc, prediction)
        compile_information = ml.helper.get_index_to_comppath_LUT()[prediction]
        return compiled_qc, compile_information

    elif model == "RL":
        predictor = rl.Predictor()
        compiled_qc, compile_information = predictor.compile(qc, opt_objective)
        return compiled_qc, compile_information

    else:
        raise ValueError("Choose between 'ML' and 'RL' Model.")
