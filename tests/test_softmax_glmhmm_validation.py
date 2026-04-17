from pathlib import Path
import sys

import jax.numpy as jnp
import pytest

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from glmhmmt.model import SoftmaxGLMHMM


def _build_model():
    model = SoftmaxGLMHMM(
        num_states=2,
        num_classes=2,
        emission_input_dim=2,
        transition_input_dim=0,
        m_step_num_iters=1,
    )
    params, props = model.initialize()
    return model, params, props


def test_fit_em_multisession_rejects_negative_class_labels():
    model, params, props = _build_model()
    emissions = jnp.array([-1, 1, -1, 1], dtype=jnp.int32)
    inputs = jnp.ones((4, 2), dtype=jnp.float32)
    session_ids = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

    with pytest.raises(ValueError, match=r"categorical class indices"):
        model.fit_em_multisession(
            params,
            props,
            emissions,
            inputs,
            session_ids=session_ids,
            num_iters=1,
            verbose=False,
        )


def test_smoother_multisession_rejects_out_of_range_class_labels():
    model, params, _props = _build_model()
    emissions = jnp.array([0, 2, 1, 0], dtype=jnp.int32)
    inputs = jnp.ones((4, 2), dtype=jnp.float32)
    session_ids = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

    with pytest.raises(ValueError, match=r"categorical class indices"):
        model.smoother_multisession(
            params,
            emissions,
            inputs,
            session_ids=session_ids,
        )
