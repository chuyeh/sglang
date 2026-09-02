# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ROCm defaults a GDN model's SSM state to bf16, and only where it is safe.

The recurrent decode kernel is bound by state bandwidth, so on gfx9xx the state
dtype sets its floor and the mamba pool's capacity. This default overrides a
`float32` sitting in the checkpoint config, which makes the negative cases the
interesting ones: an explicit dtype must win, a non-GDN model must be untouched,
and the speculative paths that need fp32 must not be flipped underneath.

GDN is detected off `linear_num_key_heads` on the *text* config, which is what
the qwen3_next / qwen3_5 model files branch on. Two nearby signals are wrong
here and are pinned below: the linear-attn registry is still empty during
argument resolution (it fills when the model classes import), and these config
classes drop `layer_types`, rebuilding the layout from `full_attention_interval`.

    python -m pytest test/registered/unit/server_args/test_rocm_mamba_ssm_dtype_default.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups import attention_hook
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Platform:
    """Every capability is off unless a test names it."""

    def __init__(self, **flags):
        self._flags = flags

    def __getattr__(self, name):
        return self._flags.get(name, False)


def _model_config(*, gdn=True, nested=True):
    # Qwen3.5 ships as a ...ForConditionalGeneration, so its language config is
    # nested; flat configs have to keep working too.
    text = SimpleNamespace(linear_num_key_heads=16) if gdn else SimpleNamespace()
    hf_config = SimpleNamespace(get_text_config=lambda: text) if nested else text
    return SimpleNamespace(hf_config=hf_config)


def _declared_ssm_dtype(*, is_hip=True, model_config=None, **overrides):
    """Run the handler over a stand-in and report what it declared, if anything.

    Reads the declaration stash rather than the field: a declared resolution
    writes no attribute, so the raw record keeps whatever the caller passed.
    """
    sa = ServerArgs.__new__(ServerArgs)
    fields = {
        "disaggregation_mode": "null",
        "enable_linear_replayssm": False,
        "enable_linear_replayssm_spec": False,
        "linear_attn_backend": None,
        "linear_attn_decode_backend": None,
        "linear_attn_prefill_backend": None,
        "linear_attn_verify_backend": None,
        "linear_replayssm_cache_len": None,
        "mamba_radix_cache_strategy": "auto",
        "mamba_ssm_dtype": None,
        "speculative_algorithm": None,
        "speculative_eagle_topk": None,
    }
    fields.update(overrides)
    for name, value in fields.items():
        object.__setattr__(sa, name, value)

    config = _model_config() if model_config is None else model_config
    with mock.patch.object(
        attention_hook, "get_platform", lambda: _Platform(is_hip=is_hip)
    ), mock.patch.object(attention_hook, "model_config_of", lambda _: config):
        try:
            attention_hook.handle_linear_attn_backend(sa)
        except Exception:
            # The bf16 branch is the first thing the handler does, so whatever
            # a later branch wants from this stand-in, it has already declared.
            pass

    declared = [
        fields_["mamba_ssm_dtype"]
        for _source, fields_ in getattr(sa, "_resolved_overrides", None) or []
        if "mamba_ssm_dtype" in fields_
    ]
    return declared[-1] if declared else None


class TestRocmMambaSsmDtypeDefault(unittest.TestCase):
    def test_rocm_gdn_defaults_to_bfloat16(self):
        self.assertEqual(_declared_ssm_dtype(), "bfloat16")

    def test_flat_config_also_detected(self):
        """A GDN model whose config is not nested must resolve the same."""
        self.assertEqual(
            _declared_ssm_dtype(model_config=_model_config(nested=False)),
            "bfloat16",
        )

    def test_non_rocm_is_untouched(self):
        """CUDA and friends keep opting in by hand; nothing is declared here."""
        self.assertIsNone(_declared_ssm_dtype(is_hip=False))

    def test_explicit_dtype_wins(self):
        for chosen in ("float32", "bfloat16"):
            self.assertIsNone(_declared_ssm_dtype(mamba_ssm_dtype=chosen))

    def test_non_gdn_model_is_untouched(self):
        """No linear-attention layers means no SSM state to shrink."""
        self.assertIsNone(_declared_ssm_dtype(model_config=_model_config(gdn=False)))

    def test_replayssm_spec_keeps_fp32(self):
        """Replay folds the state against an fp32 recurrent baseline."""
        self.assertIsNone(_declared_ssm_dtype(enable_linear_replayssm_spec=True))

    def test_unmeasured_speculative_paths_are_left_alone(self):
        """sgl-project/sglang#36889 reports shorter accept lengths with a bf16
        state under DFlash, and neither verify path is measured on gfx9xx."""
        for algorithm in ("DFLASH", "dflash", "DSPARK"):
            self.assertIsNone(_declared_ssm_dtype(speculative_algorithm=algorithm))

    def test_measured_speculative_path_still_defaults(self):
        """NEXTN is measured on gfx950 at unchanged accept length, so it keeps
        the default rather than being swept up with the untested algorithms."""
        self.assertEqual(_declared_ssm_dtype(speculative_algorithm="NEXTN"), "bfloat16")


if __name__ == "__main__":
    unittest.main()
