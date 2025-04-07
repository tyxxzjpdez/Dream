import torch
import time
import gradio as gr
from transformers import AutoModel, AutoTokenizer

# 加载 Dream 模型和 tokenizer
model_path = "Dream-org/Dream-v0-Instruct-7B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device).eval()

# 格式化对话历史：输入为列表，每个元素为 {"role": "user"/"assistant", "content": "..."}
def format_chat_history(history):
    # 这里假设 history 为一个列表，每个元素是一对 [user_message, assistant_message]，assistant_message 为 None 表示还未回复
    messages = []
    for pair in history:
        user_msg, assistant_msg = pair
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages

# 定义生成过程函数（带中间状态记录）
def dream_generate_with_visualization(history, max_new_tokens, steps, temperature, top_p, top_k, delay):
    # history: 多轮对话历史，格式为 [[user, assistant], ...]，最后一轮assistant为None表示待生成回复
    # 格式化对话
    messages = format_chat_history(history)
    # 生成 prompt，注意 Dream 模型要求使用 return_tensors="pt"
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # 用于保存扩散过程中每一步的中间文本状态
    visualization_states = []

    # 定义自定义 hook，用于捕捉每一步生成状态
    def my_generation_tokens_hook(step, x, logits):
        # x: 当前生成的 token 序列 (形状 [1, L])
        # 解码当前序列（不跳过特殊 token，可保留 mask_token 信息）
        decoded = tokenizer.decode(x[0].tolist(), skip_special_tokens=False)
        visualization_states.append(decoded)
        # 返回原始 x，不做修改
        return x

    # 调用 diffusion_generate，注意传入自定义 hook 函数
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        output_history=True,
        return_dict_in_generate=True,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        # 其他 diffusion 参数可根据需要调整
        generation_tokens_hook_func=my_generation_tokens_hook
    )

    # 生成最终文本：从生成的序列中去除 prompt 部分
    final_tokens = output.sequences[0][input_ids.shape[1]:]
    final_text = tokenizer.decode(final_tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # 将最终状态也添加到可视化状态列表中
    visualization_states.append(final_text)

    # 使用生成器方式逐步返回可视化状态，每步之间延时 delay 秒
    for state in visualization_states:
        time.sleep(delay)
        yield history, state, final_text

    # 最后返回更新后的对话历史（将最新一轮的 assistant 回复更新为最终文本）
    history[-1][1] = final_text
    yield history, visualization_states[-1], final_text

# 帮助函数：添加一轮对话到历史记录中
def add_message(history, message, response):
    history = history.copy()
    history.append([message, response])
    return history

# 用户消息提交，先更新历史（不生成回复）
def user_message_submitted(message, history):
    if not message.strip():
        return history, history, ""
    # 将用户消息加入历史，assistant 回复设为 None
    history = add_message(history, message, None)
    return history, history, ""

# 生成回复，调用 dream_generate_with_visualization
def bot_response(history, max_new_tokens, steps, temperature, top_p, top_k, delay):
    if not history:
        return history, "", ""
    # 获取最新用户消息（最后一轮 assistant 回复为 None）
    # 使用生成器返回每个中间状态
    for out in dream_generate_with_visualization(history, max_new_tokens, steps, temperature, top_p, top_k, delay):
        yield out

# 构建 Gradio Demo
css = """
/* 可根据需要自定义样式 */
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Dream Diffusion 模型 Demo")
    gr.Markdown("[模型链接](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)")
    
    # 会话历史（多轮对话记录）
    chat_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=500, type="messages")
            with gr.Group():
                with gr.Row():
                    user_input = gr.Textbox(label="你的消息", placeholder="请输入...", show_label=False)
                    send_btn = gr.Button("发送")
        with gr.Column(scale=2):
            # 展示扩散生成过程，每一步的文本
            vis_output = gr.Textbox(label="扩散生成过程", placeholder="生成过程...", lines=10)
    
    with gr.Accordion("生成参数", open=False):
        max_new_tokens_slider = gr.Slider(16, 256, value=128, step=16, label="生成长度")
        steps_slider = gr.Slider(16, 512, value=256, step=16, label="扩散步数")
        temperature_slider = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="温度")
        top_p_slider = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p")
        top_k_slider = gr.Slider(0, 100, value=0, step=5, label="Top-k (0 表示不开启)")
        delay_slider = gr.Slider(0.0, 1.0, value=0.05, step=0.05, label="可视化延时 (秒)")
    
    clear_btn = gr.Button("清空对话")
    
    # 清空对话函数
    def clear_conversation():
        return [], [], ""
    
    clear_btn.click(fn=clear_conversation, inputs=[], outputs=[chat_history, chatbot, vis_output])
    
    # 用户消息提交：先更新对话历史，显示用户消息
    user_submit = user_input.submit(
        fn=user_message_submitted,
        inputs=[user_input, chat_history],
        outputs=[chat_history, chatbot, user_input]
    )
    send_btn.click(
        fn=user_message_submitted,
        inputs=[user_input, chat_history],
        outputs=[chat_history, chatbot, user_input]
    )
    
    # 在用户消息更新后，生成回复（支持生成器）
    user_submit.then(
        fn=bot_response,
        inputs=[chat_history, max_new_tokens_slider, steps_slider, temperature_slider, top_p_slider, top_k_slider, delay_slider],
        outputs=[chatbot, vis_output, chat_history]
    )
    send_btn.click(
        fn=bot_response,
        inputs=[chat_history, max_new_tokens_slider, steps_slider, temperature_slider, top_p_slider, top_k_slider, delay_slider],
        outputs=[chatbot, vis_output, chat_history]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
