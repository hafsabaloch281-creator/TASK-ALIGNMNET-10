import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_USER = "Ahm-d18"
BASE_MODEL = "Qwen/Qwen2.5-0.5B"

MODELS = {
    "base": BASE_MODEL,
    "dpo_0.1": f"{HF_USER}/qwen2.5-0.5b-dpo-beta01",
    "dpo_0.5": f"{HF_USER}/qwen2.5-0.5b-dpo-beta05",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPTS = [
    "Explain what In-Contextual Learning.",
    "What is the difference between Direct Preference Optimization and Supervised Fine Tubing?",
    "I have an assignment to complete within a week on RLHF. What should I do, which model should I select and which dataset should I select for this topic?",
    "Write a brief description about weather conditon of Gwadar.",
    "A shop sells 4 cham for 25 rupees. How much will 15 cham cost? Explain step by step.",
    "What is your favourite subject and why was it chosen as the favourite subject?",
]

_cache = {}


def load_model(key: str):
    if key not in _cache:
        tok = AutoTokenizer.from_pretrained(MODELS[key])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODELS[key], dtype=DTYPE).to(DEVICE).eval()
        _cache[key] = (tok, model)
    return _cache[key]


def run_generation(key: str, prompt: str, max_new_tokens: int, temperature: float, rep_penalty: float) -> str:
    if not prompt or not prompt.strip():
        return "Please enter or select a prompt first."

    tok, model = load_model(key)
    chat_text = tok.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(chat_text, return_tensors="pt").to(DEVICE)

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [tok.eos_token_id] + ([im_end_id] if im_end_id is not None else [])

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(rep_penalty),
        "eos_token_id": stop_ids,
        "pad_token_id": tok.pad_token_id,
    }
    if temperature and float(temperature) > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def gen_base(prompt, max_tokens, temp, rep):
    return run_generation("base", prompt, max_tokens, temp, rep)


def gen_aligned(prompt, beta_choice, max_tokens, temp, rep):
    key = "dpo_0.1" if beta_choice == "beta = 0.1" else "dpo_0.5"
    return run_generation(key, prompt, max_tokens, temp, rep)


def gen_side_by_side(prompt, max_tokens, temp, rep):
    return (
        run_generation("base", prompt, max_tokens, temp, rep),
        run_generation("dpo_0.1", prompt, max_tokens, temp, rep),
        run_generation("dpo_0.5", prompt, max_tokens, temp, rep),
    )


CUSTOM_CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; }
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0284c7 100%);
    padding: 24px 28px;
    border-radius: 16px;
    color: #ffffff;
    margin-bottom: 18px;
}
.hero-banner h1 { margin: 0 0 8px 0; font-size: 1.75rem; color: #ffffff; }
.hero-banner p { margin: 0; color: #e2e8f0; font-size: 0.98rem; }
"""

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="sky")

with gr.Blocks(title="Qwen2.5-0.5B DPO Alignment Lab") as demo:
    gr.HTML(
        """
        <div class="hero-banner">
            <h1>Qwen2.5-0.5B Preference Alignment Lab (DPO)</h1>
            <p>Interactive comparison of <strong>Base Model</strong> vs. <strong>DPO (β = 0.1)</strong> vs. <strong>DPO (β = 0.5)</strong> trained on <code>trl-lib/ultrafeedback_binarized</code>.</p>
        </div>
        """
    )

    with gr.Accordion("Experiment Overview & Prompt Bank", open=False):
        gr.Markdown(
            """
            * **Base Model:** `Qwen/Qwen2.5-0.5B`
            * **Alignment Method:** Direct Preference Optimization (DPO) with LoRA (`r=16, lora_alpha=32`)
            * **Dataset:** `trl-lib/ultrafeedback_binarized`
            * **KL Penalty (`β`):** Controls how closely the policy stays near the reference model (`0.1` vs `0.5`).
            """
        )

    with gr.Tabs():
        with gr.TabItem("1. Before Alignment (Base)"):
            with gr.Row():
                with gr.Column(scale=1):
                    p_base = gr.Textbox(label="Prompt", lines=4, placeholder="Type a prompt or click an example below...")
                    mt_base = gr.Slider(32, 400, value=200, step=8, label="Max New Tokens")
                    t_base = gr.Slider(0.0, 1.2, value=0.0, step=0.1, label="Temperature (0 = Deterministic)")
                    r_base = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="Repetition Penalty")
                    btn_base = gr.Button("Generate Base Response", variant="primary")
                with gr.Column(scale=1):
                    out_base = gr.Textbox(label="Base Model Output", lines=14, interactive=False)
            btn_base.click(gen_base, [p_base, mt_base, t_base, r_base], out_base)
            gr.Examples([[p] for p in PROMPTS], inputs=p_base, label="Click Any Assignment Prompt")

        with gr.TabItem("2. After Alignment (DPO)"):
            with gr.Row():
                with gr.Column(scale=1):
                    p_dpo = gr.Textbox(label="Prompt", lines=4, placeholder="Type a prompt or click an example below...")
                    beta_radio = gr.Radio(["beta = 0.1", "beta = 0.5"], value="beta = 0.1", label="Select DPO Beta")
                    mt_dpo = gr.Slider(32, 400, value=200, step=8, label="Max New Tokens")
                    t_dpo = gr.Slider(0.0, 1.2, value=0.0, step=0.1, label="Temperature (0 = Deterministic)")
                    r_dpo = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="Repetition Penalty")
                    btn_dpo = gr.Button("Generate Aligned Response", variant="primary")
                with gr.Column(scale=1):
                    out_dpo = gr.Textbox(label="DPO Model Output", lines=14, interactive=False)
            btn_dpo.click(gen_aligned, [p_dpo, beta_radio, mt_dpo, t_dpo, r_dpo], out_dpo)
            gr.Examples([[p] for p in PROMPTS], inputs=p_dpo, label="Click Any Assignment Prompt")

        with gr.TabItem("3. Side-by-Side Comparison"):
            p_cmp = gr.Textbox(label="Prompt", lines=3, placeholder="Enter a prompt or pick one from the examples below...")
            with gr.Row():
                mt_cmp = gr.Slider(32, 400, value=200, step=8, label="Max New Tokens")
                t_cmp = gr.Slider(0.0, 1.2, value=0.0, step=0.1, label="Temperature")
                r_cmp = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="Repetition Penalty")
            btn_cmp = gr.Button("Compare All Three Models", variant="primary")
            with gr.Row():
                col_base = gr.Textbox(label="Base (Qwen2.5-0.5B)", lines=13, interactive=False)
                col_b01 = gr.Textbox(label="DPO (beta = 0.1)", lines=13, interactive=False)
                col_b05 = gr.Textbox(label="DPO (beta = 0.5)", lines=13, interactive=False)
            btn_cmp.click(gen_side_by_side, [p_cmp, mt_cmp, t_cmp, r_cmp], [col_base, col_b01, col_b05])
            gr.Examples([[p] for p in PROMPTS], inputs=p_cmp, label="Click Any Assignment Prompt")

if __name__ == "__main__":
    demo.queue().launch(theme=theme, css=CUSTOM_CSS)
