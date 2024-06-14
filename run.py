from vllm import LLM, SamplingParams
 
# ===========================================================================
# batch prompts
# ===========================================================================
prompts = ["The president of the United States is",
           "The capital of France is",
           "The future of AI is",]
 
# ===========================================================================
# Sampling parameters
# ===========================================================================
sampling_params = SamplingParams(temperature=0.2,
                                 max_tokens=512,
                                top_p=0.95)
 
# ===========================================================================
# Initialize vLLM offline batched inference instance, and load the model
# ===========================================================================
llm = LLM(
    # model="/models/Meta-Llama-3-8B-Instruct-FP8/",
    model="/models/llama-2-7b-chat-hf/",
    #   if args.kv_cache_scales_path!='' else None,
    quantization="fp8",
    # quantization_param_path='quantized/quark/llama.safetensors',
    quantization_param_path='quark/llama.safetensors',
    enforce_eager=True,
    tensor_parallel_size=4
)
    # enforce_eager=True)
 
# ===========================================================================
# Do the inference
# ===========================================================================
outputs = llm.generate(prompts, sampling_params)
 
# ===========================================================================
# Print the result for each prompt
# ===========================================================================
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")