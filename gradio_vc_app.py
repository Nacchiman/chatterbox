import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = ChatterboxVC.from_pretrained(DEVICE)
def generate(audio, target_voice_path, apply_watermark):
    wav = model.generate(
        audio,
        target_voice_path=target_voice_path,
        apply_watermark=apply_watermark,
    )
    return model.sr, wav.squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file"),
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file (if none, the default voice is used)", value=None),
        gr.Checkbox(value=True, label="Perth ウォーターマークを付ける"),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch()
