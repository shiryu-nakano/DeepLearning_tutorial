
from sacred import Experiment

ex = Experiment('tutorial')

@ex.config
def cfg():
    message = "hello"
    num_steps = 5

@ex.automain
def run(_run, message, num_steps):
    _run.info = []
    for i in range(num_steps):
        loss = 1.0 / (i + 1)

        # ここがロギングの本体: dict を _run.info に追記する
        _run.info.append({"step": i, "loss": loss})                                                 
        print(f"Step {i}: loss={loss:.4f}, message={message}")
                                                                        
