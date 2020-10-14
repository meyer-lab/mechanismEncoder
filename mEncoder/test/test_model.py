from .. import load_model
import amici


def test_model_compilation():
    model, solver = load_model()
    amici.runAmiciSimulation(model, solver)
