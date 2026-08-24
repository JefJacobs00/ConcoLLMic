from app.model import claude, common, gemini, gpt, local


def register_all_models() -> None:
    """
    Register all models. This is called in main.
    """
    common.register_model(gpt.Gpt4o_20241120())
    common.register_model(gpt.Gpt4o_20240806())
    common.register_model(gpt.Gpt4o_20240513())
    common.register_model(gpt.Gpt4o_mini_20240718())
    common.register_model(gpt.Gpt4_Turbo20240409())
    common.register_model(gpt.Gpt4_0125Preview())
    common.register_model(gpt.Gpt4_1106Preview())
    common.register_model(gpt.Gpt35_Turbo0125())
    common.register_model(gpt.Gpt35_Turbo1106())
    common.register_model(gpt.Gpt35_Turbo16k_0613())
    common.register_model(gpt.Gpt35_Turbo0613())
    common.register_model(gpt.Gpt4_0613())
    common.register_model(gpt.Gpt_o1mini())

    common.register_model(claude.Claude3Opus())
    common.register_model(claude.Claude3Sonnet())
    common.register_model(claude.Claude3Haiku())
    common.register_model(claude.Claude3_5Sonnet())
    common.register_model(claude.Claude3_7Sonnet())
    common.register_model(claude.Claude3_7Sonnet_128k())
    common.register_model(claude.Claude4_5Sonnet())
    common.register_model(gemini.Gemini2_0Flash())
    common.register_model(gemini.Gemini2_5Flash())
    common.register_model(gemini.Gemini2_5Pro())
    common.register_model(gemini.Gemini3Flash())
    common.register_model(gemini.Gemini3_1Pro())
    common.register_model(gemini.Gemini3_5Flash())

    # locally-served, OpenAI-compatible models (vLLM, SGLang, ...)
    common.register_model(local.Qwen3_5_9B())
    common.register_model(local.Qwen3_8_27B_NVFP4())

    # register default model as selected
    common.SELECTED_MODEL = claude.Claude4_5Sonnet()
