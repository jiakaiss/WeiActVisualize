"""Gradio application entry point."""
from __future__ import annotations

from typing import Optional

import gradio as gr
import pandas as pd

from .charts import (
    channel_heatmap,
    channel_stats_heatmap,
    channel_violin,
    comparison_histograms,
    distribution_histogram,
)
from .progress import ProgressReporter, gradio_progress_adapter
from .structure import build_module_table, build_overview
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
        self._resolve_result = None
        self._aggregator = None

    def load(self, model_name_or_path: str, dtype: str, device: str):
        self._model, self._tokenizer = load_model(
            model_name_or_path, dtype=dtype, device=device,
        )
        self._resolve_result = resolve_modules(self._model)
        self._modules = self._resolve_result.modules
        overview = build_overview(self._resolve_result, self._model)
        table = pd.DataFrame(build_module_table(self._resolve_result, self._model))
        status = (f"Loaded. family={self._resolve_result.family}, "
                  f"{len(self._modules)} target modules.")
        return status, overview, table, str(self._model)

    def module_choices(self):
        return [m.path for m in self._modules]

    def view_weight(self, module_path: str, num_bins: int, granularity: str,
                    group_size):
        if self._model is None:
            raise gr.Error("请先在「模型加载」tab 加载模型后再查看分布")
        if not module_path:
            raise gr.Error("请先点「刷新 module 列表」并选择一个 module")
        gran = Granularity(granularity) if isinstance(granularity, str) else granularity
        gs = int(group_size) if group_size else None
        if gran == Granularity.PER_GROUP and (not gs or gs <= 0):
            raise gr.Error("per-group 粒度下 group_size 需为正整数")
        w = get_weight(self._model, module_path)
        nb = int(num_bins)
        label = "per-group" if gran == Granularity.PER_GROUP else "per-channel"
        fig = distribution_histogram(w, name=module_path, num_bins=nb)
        stats = weight_stats(w, module_path, granularity=gran, group_size=gs, num_bins=nb)
        hm = channel_stats_heatmap(stats, title=f"{label} shape: {module_path}")
        vio = channel_violin(w, granularity=gran, group_size=gs, stats=stats,
                             name=f"{label} violin: {module_path}")
        return fig, hm, vio

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
        gr.Markdown("# WeiActVisualize - 量化适配性分析")
        with gr.Tab("模型加载"):
            m_in = gr.Textbox(label="model name or path", value=app_obj.settings.model_name_or_path)
            d_in = gr.Dropdown(["fp32", "fp16", "bf16"], value=app_obj.settings.dtype, label="dtype")
            dev_in = gr.Dropdown(["cpu", "cuda", "auto"], value=app_obj.settings.device, label="device")
            load_btn = gr.Button("加载模型")
            load_out = gr.Textbox(label="状态")
            gr.Markdown("### 模型结构")
            overview_md = gr.Markdown("（加载后显示概览）")
            gr.Markdown("### 完整网络结构（print model）")
            model_code = gr.Code(label="model architecture", language="python", interactive=False)
            structure_df = gr.Dataframe(
                headers=["layer", "path", "kind", "shape", "dtype", "params"],
                datatype=["number", "str", "str", "str", "str", "number"],
                interactive=False,
                wrap=True,
            )
            load_btn.click(app_obj.load, [m_in, d_in, dev_in],
                           [load_out, overview_md, structure_df, model_code])

        with gr.Tab("权重分布"):
            mod_dd = gr.Dropdown([], label="module")

            def refresh():
                choices = app_obj.module_choices()
                return gr.update(choices=choices, value=choices[0] if choices else None)

            gr.Button("刷新 module 列表").click(refresh, outputs=mod_dd)
            bins_in = gr.Slider(16, 1024, value=256, step=16, label="histogram bins")
            gran_dd = gr.Dropdown(["per-channel", "per-group"], value="per-channel", label="粒度")
            gs_in = gr.Number(value=128, label="group_size (per-group 时生效)")
            with gr.Accordion("术语说明（kurtosis / q1 / q3 / kde 等含义）", open=False):
                gr.Markdown("""### 分布形态指标
- **excess kurtosis（超额峰度，轴上 `k=`）**：度量重尾程度。正态分布 = 0；> 0 重尾（> 3 标记“重尾”）；< 0 轻尾。越大 = 尾部越重 = 量化时 scale 越被离群值浪费。
- **skewness（偏度）**：分布不对称性。|偏度| > 0.5 时提示“偏态，非对称量化可能更优”。
- **tail-ratio**：`(p99.9 − p0.1) / (2·std)`，尾部动态范围 / 主体宽度。大 = 少数极值撑开 range。
- **outlier-ratio**：超过 99.9 百分的离群值比例（每切片约 0.1%，跨切片区分度低，仅供参考）。
- **形态标签**：正态 / 重尾 / 轻尾（基于 kurtosis 阈值）。

### violin 图（鼠标悬停显示）
- **kde（核密度估计）**：violin 的轮廓形状，某处宽度 = 该值附近数据密度（越宽 = 数据越多）。
- **q1 / q3**：25% / 75% 分位数，`q3 − q1` = IQR（主体 50% 数据的宽度）。
- **median / mean**：中位数 / 均值。
- **top-k 采样**：切片数超过 64 时，按 kurtosis 取最重尾的 64 个展示，每个标注 `k=` 值。

### 整张量直方图
整张权重展平分桶，看全局分布形状。

**怎么看**：`k=` 大 + violin 两端拉长 + IQR 窄 = 主体集中但有重尾 = 量化最难的切片。""")
            w_btn = gr.Button("查看分布")
            w_fig = gr.Plot(label="整张量分布")
            w_hm = gr.Plot(label="形态指标")
            w_vio = gr.Plot(label="分布 violin")
            w_btn.click(app_obj.view_weight, [mod_dd, bins_in, gran_dd, gs_in],
                        [w_fig, w_hm, w_vio])

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
