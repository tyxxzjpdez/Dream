import torch
import time
import gradio as gr
from transformers import AutoModel, AutoTokenizer
import copy

# --- (Keep Model Loading and Helper Functions the same) ---
# Load model and tokenizer
model_path = "Dream-org/Dream-v0-Instruct-7B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    dtype = torch.bfloat16
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
except TypeError:
    print("Falling back to float32 due to potential bfloat16 issues.")
    dtype = torch.float32
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device).eval()

# Helper: Format Gradio history [[user, assistant], ...] to messages [{"role": ..., "content": ...}, ...]
def format_gradio_history_to_messages(history):
    messages = []
    if not history:
        return messages
    for pair in history:
        user_msg, assistant_msg = pair
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": str(assistant_msg)})
    return messages

# Helper: Add user message to Gradio history
def add_user_message_to_gradio_history(history, message):
    if not history:
        history = []
    new_history = history + [[message, None]]
    return new_history

# Main Generation Function (Generator) with Debugging
def dream_generate_with_visualization(history, max_new_tokens, steps, temperature, top_p, top_k, delay):
    print("\n--- Starting dream_generate_with_visualization ---") # DEBUG
    print(f"Parameters: max_new_tokens={max_new_tokens}, steps={steps}, temp={temperature}, top_p={top_p}, top_k={top_k}, delay={delay}") # DEBUG
    
    messages_for_model = format_gradio_history_to_messages(history)
    
    try:
        inputs = tokenizer.apply_chat_template(messages_for_model, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        prompt_length = input_ids.shape[1]
        print(f"Prompt length: {prompt_length}") # DEBUG
    except Exception as e:
        print(f"Error during input tokenization: {e}") # DEBUG
        error_message = f"输入处理错误: {e}"
        history[-1][1] = error_message
        yield format_gradio_history_to_messages(history), error_message, history
        return

    visualization_token_states = []
    hook_call_count = 0 # DEBUG

    def my_generation_tokens_hook(step, x, logits):
        nonlocal hook_call_count # DEBUG
        hook_call_count += 1 # DEBUG
        # Limit storing states if it grows excessively (sanity check)
        # if hook_call_count <= steps + 5: # Allow a small buffer
        visualization_token_states.append(x[0].clone().cpu())
        # Optional: Print hook call info, can be verbose
        # print(f"  Hook called: step={step}, count={hook_call_count}, seq_len={x.shape[1]}") # DEBUG (verbose)
        return x

    effective_top_k = top_k if top_k > 0 else None

    print(f"Calling model.diffusion_generate...") # DEBUG
    start_time = time.time() # DEBUG
    try:
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=effective_top_k,
            generation_tokens_hook_func=my_generation_tokens_hook
        )
        end_time = time.time() # DEBUG
        print(f"model.diffusion_generate finished in {end_time - start_time:.2f} seconds.") # DEBUG
        print(f"Hook was called {hook_call_count} times.") # DEBUG
        print(f"Number of states captured: {len(visualization_token_states)}") # DEBUG
        # Expected count is steps + 1 (step=None plus steps 0 to steps-1)
        print(f"Expected number of states based on steps ({steps}): {steps + 1}") # DEBUG

    except Exception as e:
        print(f"Error during diffusion_generate: {e}") # DEBUG
        error_message = f"模型生成出错: {e}"
        history[-1][1] = error_message # Update history state with error
        yield format_gradio_history_to_messages(history), error_message, history
        print("--- Exiting dream_generate_with_visualization due to error ---") # DEBUG
        return # Stop generation

    # --- Yield intermediate visualization steps ---
    intermediate_history = copy.deepcopy(history)
    num_states_to_process = len(visualization_token_states)
    print(f"Starting intermediate yield loop for {num_states_to_process -1} states...") # DEBUG

    # Loop over states captured by the hook, excluding the very last one maybe?
    # Let's iterate up to num_states_to_process - 1, which corresponds indices 0 to num_states_to_process - 2
    # If hook called steps+1 times, this iterates steps times. Seems right.
    yield_count = 0
    for i, state_tensor in enumerate(visualization_token_states[:-1]):
        yield_count += 1 # DEBUG
        # print(f"  Yielding intermediate step {yield_count}/{num_states_to_process - 1} (index {i})") # DEBUG (verbose)
        generated_tokens = state_tensor[prompt_length:]
        decoded_vis = tokenizer.decode(generated_tokens.tolist(), skip_special_tokens=False)
        decoded_vis = decoded_vis.replace(tokenizer.eos_token, "<EOS>")
        if tokenizer.mask_token:
            decoded_vis = decoded_vis.replace(tokenizer.mask_token, "❓")

        intermediate_history[-1][1] = f"⏳ Step {i+1}/{num_states_to_process - 1}" # Keep UI update minimal
        messages_for_chatbot_update = format_gradio_history_to_messages(intermediate_history)

        time.sleep(delay)
        yield messages_for_chatbot_update, decoded_vis, history # Yield original history state

    print(f"Finished intermediate yield loop after {yield_count} yields.") # DEBUG

    # --- Process and yield final result ---
    print("Processing final result...") # DEBUG
    try:
        final_tokens_tensor = output.sequences[0][prompt_length:]
        final_text = tokenizer.decode(final_tokens_tensor.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        final_vis_display = final_text # Show clean text in vis box at the end

        history[-1][1] = final_text # Update original history state
        final_messages_for_chatbot = format_gradio_history_to_messages(history)

        print("Yielding final result.") # DEBUG
        yield final_messages_for_chatbot, final_vis_display, history # Yield final state
    except Exception as e:
        print(f"Error processing final output: {e}") # DEBUG
        error_message = f"处理最终输出时出错: {e}"
        history[-1][1] = error_message
        yield format_gradio_history_to_messages(history), error_message, history


    print("--- Exiting dream_generate_with_visualization normally ---") # DEBUG


# User message submission handler
def user_message_submitted(message, history):
    if not message or not message.strip():
        return history, format_gradio_history_to_messages(history), ""
    new_history = add_user_message_to_gradio_history(history, message)
    messages_for_chatbot = format_gradio_history_to_messages(new_history)
    # Clear visualization box when user submits new message
    return new_history, messages_for_chatbot, "" # Return state, chatbot msgs, clear input


# Bot response generator wrapper
def bot_response_generator(history, max_new_tokens, steps, temperature, top_p, top_k, delay):
    if not history or history[-1][1] is not None:
        print("Skipping bot response: No history or last message already has response.") # DEBUG
        yield format_gradio_history_to_messages(history), "", history # Yield current state to avoid errors
        return

    # Call the main generation function (generator)
    yield from dream_generate_with_visualization(history, max_new_tokens, steps, temperature, top_p, top_k, delay)


# --- (Keep Gradio UI Build the same) ---
css = """
/* Make chatbot text selectable */
.gradio-container .prose ::selection { background-color: #ACE6FF; }
.gradio-container .prose ::-moz-selection { background-color: #ACE6FF; }
#vis_output_box textarea { font-family: monospace; font-size: 0.9em; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Dream Diffusion 模型 Demo (Text-to-Text)")
    gr.Markdown("与 Dream-v0-Instruct-7B 模型进行多轮对话，并观察扩散生成过程。")
    gr.Markdown("模型链接: [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)")

    chat_history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_display = gr.Chatbot(
                label="对话", bubble_full_width=False, height=600, type="messages"
            )
            with gr.Group():
                with gr.Row():
                    user_input_textbox = gr.Textbox(
                        label="你的消息", placeholder="请输入消息后按 Enter 或点击发送...", scale=4, show_label=False, container=False
                    )
                    send_button = gr.Button("发送", scale=1, variant="primary")

        with gr.Column(scale=2):
            vis_output_textbox = gr.Textbox(
                label="扩散生成过程 (中间步骤)", placeholder="模型生成时，这里会逐步显示中间状态...", lines=25, max_lines=25, interactive=False, elem_id="vis_output_box"
            )

    with gr.Accordion("生成参数", open=False):
        max_new_tokens_slider = gr.Slider(16, 512, value=128, step=16, label="最大生成长度 (Max New Tokens)")
        steps_slider = gr.Slider(8, 512, value=128, step=8, label="扩散步数 (Steps)") # Reduced minimum steps
        temperature_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="温度 (Temperature, 0 = 确定性)")
        top_p_slider = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p (0 = 禁用)")
        top_k_slider = gr.Slider(0, 100, value=0, step=1, label="Top-k (0 = 禁用)")
        delay_slider = gr.Slider(0.0, 0.5, value=0.02, step=0.01, label="可视化延时 (秒)")

    clear_button = gr.Button("🗑️ 清空对话")

    # Clear function needs to clear visualization box too now
    def clear_conversation():
        return [], [], "", "" # chat_history_state, chatbot_display, user_input_textbox, vis_output_textbox

    clear_button.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chat_history_state, chatbot_display, user_input_textbox, vis_output_textbox],
        queue=False
    )

    generation_params = [
        max_new_tokens_slider, steps_slider, temperature_slider, top_p_slider, top_k_slider, delay_slider
    ]

    # When user submits, clear the visualization box output as well
    submit_event_args = dict(
        fn=user_message_submitted,
        inputs=[user_input_textbox, chat_history_state],
        outputs=[chat_history_state, chatbot_display, user_input_textbox], # Removed vis_output clear here, do it below
    )

    bot_response_event_args = dict(
        fn=bot_response_generator,
        inputs=[chat_history_state] + generation_params,
        outputs=[chatbot_display, vis_output_textbox, chat_history_state]
    )

    # Chain: Submit -> Clear Input & Update Chat -> Clear Vis -> Run Generator
    submit_action = user_input_textbox.submit(**submit_event_args)
    submit_action.then(lambda: "", inputs=None, outputs=[vis_output_textbox]) # Clear vis box
    submit_action.then(**bot_response_event_args) # Run generator

    send_action = send_button.click(**submit_event_args)
    send_action.then(lambda: "", inputs=None, outputs=[vis_output_textbox]) # Clear vis box
    send_action.then(**bot_response_event_args) # Run generator


if __name__ == "__main__":
    # Reduce concurrency if you only have one GPU, helps prevent OOM or slowdowns
    demo.queue(max_size=10, default_concurrency_limit=1).launch(share=True, debug=True)
