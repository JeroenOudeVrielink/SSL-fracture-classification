import numpy as np


def save_model_output(out, save_dir):
    body_batches = [batch[0] for batch in out]
    view_batches = [batch[1] for batch in out]

    body_out = [item for sublist in body_batches for item in sublist]
    view_out = [item for sublist in view_batches for item in sublist]

    np.save(save_dir / "body_preds.npy", body_out)
    np.save(save_dir / "view_preds.npy", view_out)
    return body_out, view_out
