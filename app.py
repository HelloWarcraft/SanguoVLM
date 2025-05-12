import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from PIL import Image

# 加载模型部分
# modelscope download livehouse/SanguoVLM --local_dir /home/xlab-app-center/SanguoVLM
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('livehouse/SanguoVLM')

path = model_dir

device_map = None
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True,
    device_map="auto" if device_map is None else device_map
).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# 预处理图片函数
def preprocess(image, max_num=12, image_size=448):
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(image)
    pixel_values = pixel_values.unsqueeze(0).to(torch.bfloat16).cuda()
    return pixel_values

# 推理函数
def predict(image, message, history):
    if image is None:
        return history + [("请上传一张图片", "")]

    pixel_values = preprocess(image)
    
    if history is None or len(history) == 0:
        user_input = f"<image>\n{message}"
        response, new_history = model.chat(
            tokenizer, pixel_values, user_input,
            generation_config={"max_new_tokens": 1024, "do_sample": True},
            history=None, return_history=True
        )
    else:
        response, new_history = model.chat(
            tokenizer, pixel_values, message,
            generation_config={"max_new_tokens": 1024, "do_sample": True},
            history=history, return_history=True
        )
    
    return new_history

# Gradio界面
with gr.Blocks(title="InternVL3-9B-sft Web Demo") as demo:
    gr.Markdown("<h1 align='center'>📷 InternVL3-9B-sft Chat Demo</h1>")
    gr.Markdown("<p align='center'>⚠️ 务必上传<strong>正方形图片</strong>，否则人脸压缩，识别不准。推荐分辨率为 <code>448×448</code> 尺寸仍然能看清人脸的图片。</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传一张图片")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, label="对话")
            message_input = gr.Textbox(placeholder="请输入你的问题...", label="提问")
            send_btn = gr.Button("发送")
    
    state = gr.State([])  # 保存history

    def on_send(image, message, history):
        if message.strip() == "":
            return history
        history = predict(image, message, history)
        return history

    send_btn.click(
        fn=on_send,
        inputs=[image_input, message_input, state],
        outputs=[chatbot]
    )
    message_input.submit(
        fn=on_send,
        inputs=[image_input, message_input, state],
        outputs=[chatbot]
    )

demo.launch(server_name="127.0.0.1", server_port=6000, share=True)