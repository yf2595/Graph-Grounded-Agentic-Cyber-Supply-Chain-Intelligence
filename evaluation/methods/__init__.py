from evaluation.methods.b0_rules_graph import B0Method
from evaluation.methods.b1_single_llm import B1Method
from evaluation.methods.b2_single_llm_graph_tool import B2Method
from evaluation.methods.b3_graph_rag import B3Method
from evaluation.methods.ours import A1Method, A2Method, A3Method, A4Method, OursMethod


def _ours_variant(method_id: str):
    class _V(OursMethod):
        def __init__(self, ctx):
            super().__init__(ctx, method_id=method_id)

    _V.method_id = method_id
    return _V


def _b2_variant(method_id: str):
    class _V(B2Method):
        def __init__(self, ctx):
            super().__init__(ctx)
            self.method_id = method_id

    _V.method_id = method_id
    return _V


METHOD_REGISTRY = {
    "B0": B0Method,
    "B1": B1Method,
    "B2": B2Method,
    "B3": B3Method,
    "OURS": OursMethod,
    "A1": A1Method,
    "A2": A2Method,
    "A3": A3Method,
    "A4": A4Method,
    "OURS_QWEN3_14B": _ours_variant("OURS_QWEN3_14B"),
    "OURS_GEMMA3_12B": _ours_variant("OURS_GEMMA3_12B"),
    "B2_QWEN3_14B": _b2_variant("B2_QWEN3_14B"),
}

