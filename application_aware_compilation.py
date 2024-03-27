from mqt.predictor import rl
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QCBM")

    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--dev", type=str, default="ibm_quito")
    args = parser.parse_args()
    device = args.dev
    num_qubits = args.num_qubits
    rl.Predictor(figure_of_merit="KL", device_name=device, num_qubits=num_qubits).train_model(timesteps=30000)