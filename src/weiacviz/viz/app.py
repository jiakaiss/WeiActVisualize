"""Gradio application entry point."""
from __future__ import annotations

from typing import List, Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .charts import (
    channel_heatmap,
    channel_stats_heatmap,
    channel_violin,
    comparison_histograms,
    distribution_histogram,
    render_histogram_result,
    render_token_absmax_violin,
    sensitivity_heatmap,
)
from .progress import ProgressReporter, gradio_progress_adapter
from .structure import build_module_table, build_overview
from ..loading.calibration import load_calibration_texts
from ..loading.model_loader import load_model
from ..loading.module_resolver import resolve_modules
from ..loading.runner import (
    capture_module_output_sample,
    capture_sample_inputs,
    run_calibration,
)
from ..loading.weights import get_weight
from ..quant.fake_quant import fake_quantize_tensor
from ..quant.scheme_compare import compare_schemes
from ..quant.sensitivity import layer_sensitivity_output
from ..report.export import export_csv, export_json, export_markdown
from ..report.recommend import recommend
from ..shared.config import Settings, get_settings
from ..shared.types import CaptureConfig, Granularity, QuantConfig, Symmetry
from ..stats.activation_stats import (
    activation_stats_per_token,
    activation_token_outliers,
    activation_token_summary,
)
from ..stats.weight_stats import weight_stats

# Single-sample text for the on-demand per-token slice view. Repeated so a real
# tokenizer yields enough tokens (truncated to seq_length) for a top-k violin;
# the synthetic test tokenizer ignores the content.
_ACT_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Large language models are composed of stacked transformer blocks. "
    "Activation distributions inform quantization suitability decisions. "
    "Outlier channels dominate the activation range and drive quantization error. "
) * 8


class App:
    """Stateful application holding the loaded model and captured stats."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model = None
        self._tokenizer = None
        self._modules = []
        self._resolve_result = None
        self._aggregator = None
        self._last_report = None

    def load(self, model_name_or_path: str, dtype: str, device: str):
        self._model, self._tokenizer = load_model(
            model_name_or_path, dtype=dtype, device=device,
        )
        self._model_name = model_name_or_path
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

    def run_calib(self, dataset: str, num_samples: int, batch_size: int,
                  collect_histogram: bool, num_bins: int,
                  collect_token_stats: bool,
                  progress=gr.Progress()):
        paths = [m.path for m in self._modules]
        texts = load_calibration_texts(dataset, num_samples=num_samples)
        reporter = ProgressReporter(gradio_progress_adapter(progress))
        cfg = CaptureConfig(max_samples=len(texts), batch_size=int(batch_size))
        self._aggregator = run_calibration(
            self._model, self._tokenizer, texts, paths,
            config=cfg, progress_cb=reporter.callback,
            collect_histogram=bool(collect_histogram),
            num_bins=int(num_bins),
            collect_token_stats=bool(collect_token_stats),
        )
        n_hist = sum(
            1 for p in paths
            if self._aggregator.histograms.get(p, {}).get("output") is not None
        )
        n_tok = sum(
            1 for p in paths
            if self._aggregator.token_stats.get(p, {}).get("output") is not None
            and self._aggregator.token_stats[p]["output"].count > 0
        )
        return (f"Calibration done. {len(paths)} modules captured, "
                f"{n_tok} with per-token stats, {n_hist} with histogram.")

    def view_activation(self, module_path: str, seq_length: int = 512):
        """Per-token activation analysis at weight-side depth.

        Per-token slice view (violin + shape heatmap) uses an on-demand
        single-module activation sample -- available after model load, no
        calibration needed. Per-token abs_max distribution violin + outlier
        magnitude + global histogram need calibration (collect_histogram /
        collect_token_stats).

        Returns (advice, global_fig, token_absmax_violin, slice_violin,
        slice_heatmap).
        """
        if self._model is None:
            raise gr.Error("请先在「模型加载」tab 加载模型")
        if not module_path:
            raise gr.Error("请先刷新并选择一个 module")
        agg = self._aggregator

        # --- per-token slice view (on-demand sample, no calibration needed) ---
        slice_violin = go.Figure().update_layout(template="plotly_white")
        slice_heatmap = go.Figure().update_layout(template="plotly_white")
        if self._tokenizer is None:
            slice_violin = slice_violin.update_layout(title="缺少 tokenizer，无法捕获激活样本")
            slice_heatmap = slice_heatmap.update_layout(title="缺少 tokenizer，无法捕获激活样本")
        else:
            try:
                sample = capture_module_output_sample(
                    self._model, self._tokenizer, module_path,
                    text=_ACT_SAMPLE_TEXT, seq_length=int(seq_length),
                )
            except Exception:  # noqa: BLE001  forward/shape errors -> placeholder
                sample = None
            if sample is None:
                slice_violin = slice_violin.update_layout(
                    title=f"未捕获到 {module_path} 的激活样本")
                slice_heatmap = slice_heatmap.update_layout(
                    title=f"未捕获到 {module_path} 的激活样本")
            else:
                tok_stats = activation_stats_per_token(sample, module_path)
                slice_violin = channel_violin(
                    sample, granularity=Granularity.PER_CHANNEL, stats=tok_stats,
                    name=f"per-token 分布: {module_path}")
                slice_heatmap = channel_stats_heatmap(
                    tok_stats, title=f"per-token 形态: {module_path}")

        # --- per-token abs_max distribution + outlier magnitude (calibration) ---
        ts = agg.token_stats.get(module_path, {}).get("output") if agg else None
        if ts is None or ts.count == 0:
            token_fig = go.Figure().update_layout(
                title="未采集 per-token 统计（勾选「采集 per-token 统计」后重新校准）",
                template="plotly_white")
            advice = "未采集 per-token 统计（勾选后重新校准）；per-token 切片视图可用"
        else:
            summ = activation_token_summary(ts, module_path, "output")
            outlier_info = activation_token_outliers(ts)
            token_fig = render_token_absmax_violin(
                ts, outlier_info=outlier_info,
                name=f"per-token abs_max: {module_path}")
            parts = [
                f"形态={summ.shape_label}",
                f"mean={ts.mean:.3f} std={ts.std:.3f}",
                f"kurtosis={ts.kurtosis:.2f} skewness={ts.skewness:.2f}",
            ]
            sev = outlier_info.get("severity", float("nan"))
            ratio = outlier_info.get("outlier_ratio", float("nan"))
            if sev == sev:
                parts.append(f"离群比例={ratio:.1%} severity={sev:.2f}")
            advice = "per-token abs_max | " + " | ".join(parts)

        # --- global activation histogram (needs collect_histogram) ---
        h = agg.histograms.get(module_path, {}).get("output") if agg else None
        if h is not None:
            global_fig = render_histogram_result(
                h.to_result(), name=f"activation: {module_path}")
        else:
            global_fig = go.Figure().update_layout(
                title="未采集全局直方图（勾选「采集直方图」后重新校准）",
                template="plotly_white")

        return advice, global_fig, token_fig, slice_violin, slice_heatmap

    def scheme_compare_view(self, module_path: str, bits_list, gran_list,
                            sym_list, group_size):
        """Run a cartesian product of quant schemes and rank by error.

        Returns (error table, best-scheme text, before/after histogram of the
        best scheme). The best scheme is the lowest-MSE row; its quantized
        tensor is rendered against the original so the user sees the concrete
        distribution shift of the recommended scheme.
        """
        if self._model is None:
            raise gr.Error("请先在「模型加载」tab 加载模型后再做方案对比")
        if not module_path:
            raise gr.Error("请先点「刷新 module 列表」并选择一个 module")
        w = get_weight(self._model, module_path)
        bits_opts = [int(b) for b in bits_list] if bits_list else [4, 8]
        grans = [Granularity(g) for g in gran_list] if gran_list else list(Granularity)
        syms = [Symmetry(s) for s in sym_list] if sym_list else list(Symmetry)
        gs = int(group_size) if group_size else 128
        rows = compare_schemes(
            w, bits_options=bits_opts, granularities=grans,
            symmetries=syms, group_size=gs,
        )
        df = pd.DataFrame(rows)
        valid = df[df["mse"].notna()] if "mse" in df else df
        if len(valid):
            best_idx = valid["mse"].idxmin()
            best = df.loc[best_idx]
            best_bits = int(best["bits"])
            best_gran = Granularity(best["granularity"])
            best_sym = Symmetry(best["symmetry"])
            best_text = (
                f"最优方案: {best_bits}-bit {best_gran.value} {best_sym.value} | "
                f"mse={best['mse']:.4e} cosine={best['cosine']:.6f}"
            )
            cfg = QuantConfig(
                bits=best_bits, granularity=best_gran, symmetry=best_sym,
                group_size=gs if best_gran == Granularity.PER_GROUP else None,
            )
            q = fake_quantize_tensor(w, cfg)
            fig = comparison_histograms(
                w, q, name=f"{module_path} ({best_bits}bit {best_gran.value} {best_sym.value})",
            )
        else:
            best_text = "无有效方案（请检查粒度/group_size 设置）"
            fig = go.Figure()
        return df, best_text, fig

    def build_report(self, bits: int, seq_length: int = 128):
        """Build a per-module quantization recommendation report over the
        whole model. Uses output_diff sensitivity + weight shape, and emits a
        'why + how' reason.
        """
        if self._model is None:
            raise gr.Error("请先在「模型加载」tab 加载模型后再生成报告")
        if self._tokenizer is None:
            raise gr.Error("缺少 tokenizer，请确认模型已完整加载")
        if not self._modules:
            raise gr.Error("没有目标 module，请检查模型是否为支持的架构")
        rows = self._sensitivity_rows(bits, seq_length)
        cfg = QuantConfig(bits=int(bits), granularity=Granularity.PER_CHANNEL,
                          symmetry=Symmetry.SYMMETRIC)
        model_name = getattr(self, "_model_name", "") or self.settings.model_name_or_path
        rep = recommend(rows, model_name=model_name, config=cfg)
        self._last_report = rep
        df = pd.DataFrame([{
            "module": r.module_path, "kind": r.kind,
            "bits": r.recommended_bits, "granularity": r.recommended_granularity,
            "symmetry": r.recommended_symmetry,
            "joint_output_mse": r.joint_output_mse,
            "reason": r.reason,
        } for r in rep.recommendations])
        return df, rep.summary

    def export_report(self, fmt: str):
        """Export the last report to a temp file and return its path for download."""
        if self._last_report is None:
            raise gr.Error("请先生成报告再导出")
        import os
        import tempfile
        ext = {"markdown": "md", "json": "json", "csv": "csv"}[fmt]
        tmpdir = tempfile.mkdtemp(prefix="weiacviz_report_")
        path = os.path.join(tmpdir, f"quant_report.{ext}")
        if fmt == "markdown":
            export_markdown(self._last_report, path)
        elif fmt == "json":
            export_json(self._last_report, path)
        else:
            export_csv([r.__dict__ for r in self._last_report.recommendations], path)
        return path

    def _sensitivity_rows(self, bits: int, seq_length: int = 128) -> List[dict]:
        """Build enriched per-module sensitivity rows for the whole model.

        Combines output_diff sensitivity with per-channel weight shape
        (kurtosis/skewness median). Shared by the sensitivity overview and the
        recommendation report.
        """
        paths = [m.path for m in self._modules]
        weights = {p: get_weight(self._model, p) for p in paths}
        cfg = QuantConfig(bits=int(bits), granularity=Granularity.PER_CHANNEL,
                          symmetry=Symmetry.SYMMETRIC)
        texts = load_calibration_texts(self.settings.calibration_dataset, num_samples=1)
        sample_inputs = capture_sample_inputs(
            self._model, self._tokenizer, paths,
            text=texts[0] if texts else "", seq_length=int(seq_length),
        )
        kinds = {m.path: m.kind.value for m in self._modules}
        rows = layer_sensitivity_output(
            self._model, paths, weights, sample_inputs, cfg, kinds=kinds,
        )
        for r in rows:
            p = r["module_path"]
            ws = weight_stats(weights[p], p, granularity=Granularity.PER_CHANNEL)
            kurts = [s.kurtosis for s in ws if s.kurtosis == s.kurtosis]
            skews = [s.skewness for s in ws if s.skewness == s.skewness]
            if kurts:
                kurts_arr = np.asarray(kurts, dtype=np.float64)
                r["weight_kurtosis_max"] = float(kurts_arr.max())
                # heavy channel = per-channel excess kurtosis > 3 (Laplace-like);
                # the median would hide these (LLM weight medians are ~0.5 but a
                # few channels reach kurtosis 30-200+)
                r["heavy_channel_ratio"] = float((kurts_arr > 3.0).mean())
            else:
                r["weight_kurtosis_max"] = float("nan")
                r["heavy_channel_ratio"] = float("nan")
            r["weight_skewness"] = float(np.median(skews)) if skews else float("nan")
        return rows

    def run_sensitivity(self, bits: int, sort_by: str, seq_length: int = 128):
        """Whole-model sensitivity ranking by output_diff (main) + weight shape.

        See ``_sensitivity_rows`` for the per-module signals. The table shows
        *why* a module is sensitive in one glance: ``output_mse`` high +
        ``weight_kurtosis_max`` high => weight heavy-tail; ``joint_output_mse``
        远大于 ``output_mse`` => activation-side quantization loss.
        """
        if self._model is None:
            raise gr.Error("请先在「模型加载」tab 加载模型")
        if self._tokenizer is None:
            raise gr.Error("缺少 tokenizer，请确认模型已完整加载")
        if not self._modules:
            raise gr.Error("没有目标 module")
        rows = self._sensitivity_rows(bits, seq_length)
        key_field = "joint_output_mse" if sort_by == "joint_output_mse" else "output_mse"
        rows.sort(key=lambda r: r[key_field] if r[key_field] == r[key_field]
                  else float("-inf"), reverse=True)
        display = ["module_path", "kind", "output_mse", "joint_output_mse",
                   "weight_kurtosis_max", "heavy_channel_ratio"]
        df = pd.DataFrame(rows)[display] if rows else pd.DataFrame(columns=display)
        hm = sensitivity_heatmap(rows)
        return df, hm


def build_app(settings: Optional[Settings] = None) -> gr.Blocks:
    app_obj = App(settings)

    with gr.Blocks() as demo:
        gr.Markdown("# WeiActVisualize - 量化适配性分析")
        with gr.Tab("模型加载"):
            m_in = gr.Textbox(label="model name or path", value=app_obj.settings.model_name_or_path)
            d_in = gr.Dropdown(["fp32", "fp16", "bf16"], value=app_obj.settings.dtype, label="dtype")
            dev_in = gr.Dropdown(["auto", "cpu"], value=app_obj.settings.device, label="device")
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
            hist_in = gr.Checkbox(value=app_obj.settings.calibration_collect_histogram,
                                  label="采集直方图（两遍校准，较慢；per-token 直方图也依赖此项）")
            tok_in = gr.Checkbox(value=app_obj.settings.calibration_collect_token_stats,
                                 label="采集 per-token 统计（默认，单遍，O(1) 内存）")
            hbins_in = gr.Slider(16, 1024, value=app_obj.settings.calibration_histogram_bins,
                                 step=16, label="histogram bins")
            cal_btn = gr.Button("运行校准")
            cal_out = gr.Textbox(label="状态")
            cal_btn.click(app_obj.run_calib,
                          [ds_in, ns_in, bs_in, hist_in, hbins_in, tok_in],
                          cal_out)

            gr.Markdown(
                "激活分布分析：per-token 切片形态 + per-token abs_max 分布与离群幅度。"
            )
            act_mod = gr.Dropdown([], label="module")

            def refresh_act():
                choices = app_obj.module_choices()
                return gr.update(choices=choices, value=choices[0] if choices else None)

            gr.Button("刷新 module 列表").click(refresh_act, outputs=act_mod)
            act_btn = gr.Button("查看激活分布")
            act_advice = gr.Textbox(label="per-token abs_max 分布", lines=2)
            with gr.Row():
                act_global = gr.Plot(label="激活全局直方图")
                act_token = gr.Plot(label="per-token abs_max 分布 violin")
            with gr.Row():
                act_slice_vio = gr.Plot(label="per-token 切片 violin（top-k by kurtosis）")
                act_slice_hm = gr.Plot(label="per-token 形态热力图")
            act_btn.click(app_obj.view_activation, [act_mod],
                          [act_advice, act_global, act_token,
                           act_slice_vio, act_slice_hm])

        with gr.Tab("量化模拟"):
            q_mod = gr.Dropdown([], label="module")

            def refresh_q():
                choices = app_obj.module_choices()
                return gr.update(choices=choices, value=choices[0] if choices else None)

            gr.Button("刷新 module 列表").click(refresh_q, outputs=q_mod)
            gr.Markdown("勾选要对比的方案维度，跑笛卡尔积。表格按 mse 升序看哪种方案误差最小。")
            bits_cb = gr.CheckboxGroup(["4", "8"], value=["4", "8"], label="bits")
            gran_cb = gr.CheckboxGroup(
                ["per-tensor", "per-channel", "per-group"],
                value=["per-tensor", "per-channel"], label="粒度",
            )
            sym_cb = gr.CheckboxGroup(
                ["symmetric", "asymmetric"], value=["symmetric", "asymmetric"],
                label="对称性",
            )
            gs_in = gr.Number(value=128, label="group_size (per-group 时生效)")
            q_btn = gr.Button("方案对比")
            q_best = gr.Textbox(label="最优方案")
            q_table = gr.Dataframe(
                headers=["bits", "granularity", "symmetry", "group_size", "mse", "cosine"],
                datatype=["number", "str", "str", "number", "number", "number"],
                interactive=False, wrap=True,
            )
            q_fig = gr.Plot(label="最优方案 量化前后分布叠加")
            q_btn.click(app_obj.scheme_compare_view,
                        [q_mod, bits_cb, gran_cb, sym_cb, gs_in],
                        [q_table, q_best, q_fig])

        with gr.Tab("敏感性总览"):
            gr.Markdown(
                "全模型量化敏感性排序（默认按 **joint_output_mse**：W8A8 权重+激活联合量化层输出误差）。"
                "``output_mse``=只权重量化(W4A16)，``joint_output_mse``=权重+per-token激活(W8A8)，"
                "差值≈激活量化损失。``heavy_channel_ratio`` 高=权重重尾通道。需先加载模型。"
            )
            with gr.Row():
                sens_bits = gr.Dropdown([4, 8], value=4, label="参考 bits")
                sens_sort = gr.Radio(["joint_output_mse", "output_mse"],
                                     value="joint_output_mse", label="排序依据")
            sens_btn = gr.Button("运行敏感性分析")
            sens_table = gr.Dataframe(
                headers=["module_path", "kind", "output_mse", "joint_output_mse",
                         "weight_kurtosis_max", "heavy_channel_ratio"],
                datatype=["str", "str", "number", "number", "number", "number"],
                interactive=False, wrap=True,
            )
            sens_hm = gr.Plot(label="层 × 指标 热力图（每行独立归一化）")
            sens_btn.click(app_obj.run_sensitivity, [sens_bits, sens_sort],
                           [sens_table, sens_hm])

        with gr.Tab("报告"):
            gr.Markdown(
                "基于 output_diff 敏感性 + 权重形态，"
                "给出每层「为什么难量化 + 怎么量化(bits/粒度/对称性)」的诊断。"
                "reason 中的「激活量化损失占比」为诊断信号，不替你选算法。"
                "需先加载模型。"
            )
            rep_bits = gr.Dropdown([4, 8], value=4, label="参考 bits（敏感性评估用）")
            rep_btn = gr.Button("生成报告")
            rep_summary = gr.Textbox(label="概览")
            rep_table = gr.Dataframe(
                headers=["module", "kind", "bits", "granularity", "symmetry",
                         "joint_output_mse", "reason"],
                datatype=["str", "str", "number", "str", "str",
                          "number", "str"],
                interactive=False, wrap=True,
            )
            with gr.Row():
                exp_fmt = gr.Radio(["markdown", "json", "csv"], value="markdown",
                                   label="导出格式")
                exp_btn = gr.Button("导出")
            exp_file = gr.File(label="下载报告")
            rep_btn.click(app_obj.build_report, [rep_bits], [rep_table, rep_summary])
            exp_btn.click(app_obj.export_report, [exp_fmt], [exp_file])

    demo.queue()
    return demo


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
