from mqt.predictor import ml, rl


def compile(qc, model="ML", opt_objective="fidelity"):
    if model == "ML":
        predictor = ml.Predictor()
        prediction = predictor.predict(qc)
        return predictor.compile_predicted_compilation_path(qc, prediction)
    elif model == "RL":
        predictor = rl.Predictor()
        qc_compiled, device = predictor.compile(qc, opt_objective=opt_objective)
        return qc_compiled, device

    else:
        raise ValueError("Choose between 'ML' and 'RL' Model.")
