from mqt.predictor import ml, rl


def compile(qc, model="ML", opt_objective="fidelity"):
    if model == "ML":
        predictor = ml.Predictor()
        prediction = predictor.predict(qc)
        return predictor.compile_predicted_compilation_path(qc, prediction)
    elif model == "RL":
        predictor = rl.Predictor()
        return predictor.compile(qc, opt_objective=opt_objective)

    else:
        raise ValueError("Choose between 'ML' and 'RL' Model.")
