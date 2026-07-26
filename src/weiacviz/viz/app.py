"""Gradio application entry point."""
from __future__ import annotations

from typing import Optional

import gradio as gr

from .charts import channel_heatmap, comparison_histograms, distribution_histogram
from .progress import ProgressReporter, gradio_progress_adapter
from ..loading.calibration import load_calibration_texts
from ..loading.model_loader import load_model
from ..loading.module_resolver import resolve_modules
from ..loading.runner import run_calibration
from ..loading.weights import get_weight
from ..quant.fake_quant import fake_quantize_tensor
from ..shared.config import Settings, get_settings
from ..shared.types import Granularity, QuantConfig, Symmetry
from ..stats.weight_stats import weight_stats


class App:
    """Stateful application holding the loaded model and captured stats."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model = None
        self._tokenizer = None
        self._modules = []
        self._aggregator = None

    def load(self, model_name_or_path: str, dtype: str, device: str):
        self._model, self._tokenizer = load_model(
            model_name_or_path, dtype=dtype, device=device,
        )
        result = resolve_modules(self._model)
        self._modules = result.modules
        return (f"Loaded. family={result.family}, degraded={result.degraded}, "
                f"{len(self._modules)} target modules.")

    def module_choices(self):
        return [m.path for m in self._modules]

    def view_weight(self, module_path: str, num_bins: int):
        w = get_weight(self._model, module_path)
        fig = distribution_histogram(w, name=module_path, num_bins=int(num_bins))
        stats = weight_stats(w, module_path, granularity=Granularity.PER_CHANNEL)
        hm = channel_heatmap([s.mean for s in stats], title=f"per-channel mean: {module_path}")
        return fig, hm

    def run_calib(self, dataset: str, num_samples: int, batch_size: int, progress=gr.Progress()):
        paths = [m.path for m in self._modules]
        texts = load_calibration_texts(dataset, num_samples=num_samples)
        reporter = ProgressReporter(gradio_progress_adapter(progress))
        self._aggregator = run_calibration(
            self._model, self._tokenizer, texts, paths,
            config=None, progress_cb=reporter.callback,
        )
        return f"Calibration done. {len(paths)} modules captured."

    def quant_compare(self, module_path: str, bits: int):
        w = get_weight(self._model, module_path)
        cfg = QuantConfig(bits=int(bits), granularity=Granularity.PER_TENSOR,
                          symmetry=Symmetry.SYMMETRIC)
        q = fake_quantize_tensor(w, cfg)
        return comparison_histograms(w, q, name=module_path)


def build_app(settings: Optional[Settings] = None) -> gr.Blocks:
    app_obj = App(settings)

    with gr.Blocks() as demo:
        gr.Markdown("# WeiActVisualize — 量化适配性分析")
        with gr.Tab("模型加载"):
            m_in = gr.Textbox(label="model name or path", value=app_obj.settings.model_name_or_path)
            d_in = gr.Dropdown(["fp32", "fp16", "bf16"], value=app_obj.settings.dtype, label="dtype")
            dev_in = gr.Dropdown(["cpu", "cuda", "auto"], value=app_obj.settings.device, label="device")
            load_btn = gr.Button("加载模型")
            load_out = gr.Textbox(label="状态")
            load_btn.click(app_obj.load, [m_in, d_in, dev_in], load_out)

        with gr.Tab("权重分布"):
            mod_dd = gr.Dropdown([], label="module")

            def refresh():
                choices = app_obj.module_choices()
                return gr.update(choices=choices, value=choices[0] if choices else None)

            gr.Button("刷新 module 列表").click(refresh, outputs=mod_dd)
            bins_in = gr.Slider(16, 1024, value=256, step=16, label="histogram bins")
            w_btn = gr.Button("查看分布")
            w_fig = gr.Plot()
            w_hm = gr.Plot()
            w_btn.click(app_obj.view_weight, [mod_dd, bins_in], [w_fig, w_hm])

        with gr.Tab("校准与激活"):
            ds_in = gr.Textbox(value=app_obj.settings.calibration_dataset, label="dataset")
            ns_in = gr.Slider(1, 1024, value=app_obj.settings.calibration_samples, step=1, label="samples")
            bs_in = gr.Slider(1, 64, value=app_obj.settings.calibration_batch_size, step=1, label="batch size")
            cal_btn = gr.Button("运行校准")
            cal_out = gr.Textbox(label="状态")
            cal_btn.click(app_obj.run_calib, [ds_in, ns_in, bs_in], cal_out)

        with gr.Tab("量化模拟"):
            q_mod = gr.Dropdown([], label="module")

            def refresh_q():
                choices = app_obj.module_choices()
                return gr.update(choices=choices, value=choices[0] if choices else None)

            gr.Button("刷新 module 列表").click(refresh_q, outputs=q_mod)
            bits_in = gr.Dropdown([4, 8], value=4, label="bits")
            q_btn = gr.Button("量化对比")
            q_fig = gr.Plot()
            q_btn.click(app_obj.quant_compare, [q_mod, bits_in], q_fig)

    demo.queue()
    return demo


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
