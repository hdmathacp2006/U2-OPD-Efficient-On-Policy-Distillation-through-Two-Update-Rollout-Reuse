"""Check native model context and analytical KV capacity before allocation."""


def validate_capacity(model: object, config: dict) -> dict:
    length = config["maximum_model_length"]
    if (
        config["max_prompt_tokens"] + config["max_new_tokens"] > length
        or length > model.max_position_embeddings
    ):
        raise ValueError("Prompt/output budget exceeds the pinned native context")
    per_token = (
        2 * model.num_hidden_layers * model.num_key_value_heads * model.head_dim * 2
    )
    required = per_token * length * config["concurrency"]
    if required > config["cache_bytes"] * 0.85:
        raise ValueError("Declared KV cache cannot hold the longest active requests")
    return {
        "bf16_kv_bytes_per_token": per_token,
        "worst_case_batch_kv_bytes": required,
        "cache_bytes": config["cache_bytes"],
        "scope": "analytic_KV_only_not_total_device_peak",
    }
