import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
from torch.nn.utils import parametrizations

# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="交通事故风险预测与可视化预警系统",
    layout="wide"
)

# =========================================================
# 全局样式
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}

.main {
    background: linear-gradient(180deg, #f4f8fc 0%, #eef4fa 100%);
}

.banner {
    width: 100%;
    padding: 28px 32px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(15,32,39,0.18);
}

.banner h1 {
    font-size: 34px;
    margin: 0 0 8px 0;
    font-weight: 800;
}

.banner p {
    margin: 4px 0;
    font-size: 15px;
    opacity: 0.95;
}

.soft-card {
    background: white;
    padding: 16px 18px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(30,60,90,0.08);
    border: 1px solid rgba(20,60,120,0.06);
    margin-bottom: 14px;
}

.func-card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(30,60,90,0.08);
    border: 1px solid rgba(20,60,120,0.06);
    min-height: 128px;
}

.func-icon {
    font-size: 30px;
    margin-bottom: 6px;
}

.func-title {
    font-size: 18px;
    font-weight: 700;
    color: #16324f;
    margin-bottom: 4px;
}

.func-desc {
    font-size: 13px;
    color: #5f7287;
}

.sub-title {
    font-size: 22px;
    font-weight: 800;
    color: #17324d;
    margin: 8px 0 10px 0;
}

.small-note {
    font-size: 13px;
    color: #607080;
    line-height: 1.7;
}

.risk-box-green {
    background: linear-gradient(90deg, #1f9d55, #34c759);
    color: white;
    border-radius: 14px;
    padding: 18px;
    font-weight: 700;
    font-size: 22px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(31,157,85,0.22);
}

.risk-box-yellow {
    background: linear-gradient(90deg, #d4a017, #f5c542);
    color: white;
    border-radius: 14px;
    padding: 18px;
    font-weight: 700;
    font-size: 22px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(212,160,23,0.22);
}

.risk-box-red {
    background: linear-gradient(90deg, #c62828, #ef5350);
    color: white;
    border-radius: 14px;
    padding: 18px;
    font-weight: 700;
    font-size: 22px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(198,40,40,0.22);
}

.section-gap {
    margin-top: 8px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 顶部横幅
# =========================================================
st.markdown("""
<div class="banner">
    <h1>🚦 城市道路交通事故风险预测与可视化预警系统</h1>
    <p>基于 TSEBG 深度学习模型的历史回放式动态预测平台</p>
    <p>支持历史时点选择、未来风险预测、风险等级预警与辅助决策展示</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 基础配置
# =========================================================
DATA_PATH = "data/nyc_2022_hourly.csv"
MODEL_PATH = "outputs/best_tsebg_nyc.pt"

SEQ_LEN = 48

FEATURE_COLS = [
    "accident_count",
    "accident_log1p",
    "injury_ratio",
    "death_ratio",
    "abnormal_factor_ratio",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "accident_ma_3",
    "accident_ma_6",
    "accident_lag_24",
    "accident_lag_168",
]

TARGET_COL = "accident_log1p"
TIME_COL = "time_hour"

# =========================================================
# 模型定义
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x.contiguous()
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super().__init__()

        self.conv1 = parametrizations.weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = parametrizations.weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        try:
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
        except Exception:
            pass
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel, max(1, channel // reduction))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, channel // reduction), channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1)
        return x * out


class MHSEBlock(nn.Module):
    def __init__(self, channel, heads=4, reduction=16):
        super().__init__()
        assert channel % heads == 0, f"channel={channel} must be divisible by heads={heads}"
        self.channel = channel
        self.heads = heads
        self.c_per_head = channel // heads
        self.blocks = nn.ModuleList([
            SEBlock(self.c_per_head, reduction=reduction) for _ in range(heads)
        ])

    def forward(self, x):
        chunks = torch.chunk(x, self.heads, dim=1)
        outs = [blk(ch) for blk, ch in zip(self.blocks, chunks)]
        return torch.cat(outs, dim=1)


class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        repeated_hidden = hidden.unsqueeze(1).repeat(1, max_len, 2)
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector


class AddMeanFusion(nn.Module):
    def __init__(self, tcn_dim, gru_dim, embed_dim, mode="add"):
        super().__init__()
        assert mode in ["add", "mean"]
        self.mode = mode
        self.fc_tcn = nn.Linear(tcn_dim, embed_dim)
        self.fc_gru = nn.Linear(gru_dim, embed_dim)

    def forward(self, tcn_feat, gru_feat):
        tcn_proj = self.fc_tcn(tcn_feat)
        gru_proj = self.fc_gru(gru_feat)
        if self.mode == "add":
            return tcn_proj + gru_proj
        return 0.5 * (tcn_proj + gru_proj)


class GatedFusion(nn.Module):
    def __init__(self, tcn_dim, gru_dim, embed_dim):
        super().__init__()
        self.fc_tcn = nn.Linear(tcn_dim, embed_dim)
        self.fc_gru = nn.Linear(gru_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, tcn_feat, gru_feat):
        tcn_proj = self.fc_tcn(tcn_feat)
        gru_proj = self.fc_gru(gru_feat)
        z = self.gate(torch.cat([tcn_proj, gru_proj], dim=1))
        return z * tcn_proj + (1.0 - z) * gru_proj


class TSEBG(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels,
        kernel_size,
        hidden_layer_sizes,
        attention_dim,
        dropout,
        embed_dim,
        tcn_mha_heads=1,
        mha_dropout=0.1,
        attn_out_dropout=0.2,
        use_tcn_mha=True,
        fusion_type="add",
        se_type="se",
        mhse_heads=4,
        se_reduction=16
    ):
        super().__init__()
        self.use_tcn_mha = bool(use_tcn_mha)

        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )

            if se_type.lower() == "se":
                layers.append(SEBlock(out_channels, reduction=se_reduction))
            elif se_type.lower() == "mhse":
                layers.append(MHSEBlock(out_channels, heads=mhse_heads, reduction=se_reduction))
            else:
                raise ValueError(f"Unknown se_type={se_type}")

        self.TCNnetwork = nn.Sequential(*layers)
        tcn_out_dim = num_channels[-1]

        if self.use_tcn_mha:
            assert tcn_out_dim % tcn_mha_heads == 0
            self.tcn_mha = nn.MultiheadAttention(
                embed_dim=tcn_out_dim,
                num_heads=tcn_mha_heads,
                dropout=mha_dropout,
                batch_first=False
            )
            self.tcn_mha_ln = nn.LayerNorm(tcn_out_dim)
            self.attn_out_dropout = nn.Dropout(attn_out_dropout)
        else:
            self.tcn_mha = None
            self.tcn_mha_ln = None
            self.attn_out_dropout = None

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.bigru_layers = nn.ModuleList()
        self.bigru_layers.append(
            nn.GRU(input_dim, hidden_layer_sizes[0], batch_first=True, bidirectional=True)
        )
        for i in range(1, len(hidden_layer_sizes)):
            self.bigru_layers.append(
                nn.GRU(
                    hidden_layer_sizes[i - 1] * 2,
                    hidden_layer_sizes[i],
                    batch_first=True,
                    bidirectional=True
                )
            )

        self.globalAttention = GlobalAttention(attention_dim * 2)

        if fusion_type == "add":
            self.fusion = AddMeanFusion(
                tcn_dim=tcn_out_dim,
                gru_dim=hidden_layer_sizes[-1] * 2,
                embed_dim=embed_dim,
                mode="add"
            )
        elif fusion_type == "mean":
            self.fusion = AddMeanFusion(
                tcn_dim=tcn_out_dim,
                gru_dim=hidden_layer_sizes[-1] * 2,
                embed_dim=embed_dim,
                mode="mean"
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                tcn_dim=tcn_out_dim,
                gru_dim=hidden_layer_sizes[-1] * 2,
                embed_dim=embed_dim
            )
        else:
            raise ValueError(f"Unknown fusion_type={fusion_type}")

        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, input_seq):
        bsz = input_seq.size(0)

        tcn_input = input_seq.permute(0, 2, 1)
        tcn_feat = self.TCNnetwork(tcn_input)

        if self.use_tcn_mha:
            tcn_seq = tcn_feat.permute(2, 0, 1)
            mha_out, _ = self.tcn_mha(tcn_seq, tcn_seq, tcn_seq)
            mha_out = self.attn_out_dropout(mha_out)
            mha_out = self.tcn_mha_ln(mha_out + tcn_seq)
            tcn_pool_in = mha_out.permute(1, 2, 0)
        else:
            tcn_pool_in = tcn_feat

        tcn_features = self.avgpool(tcn_pool_in).reshape(bsz, -1)

        bigru_out = input_seq
        for bigru in self.bigru_layers:
            bigru_out, hidden = bigru(bigru_out)

        final_hidden = hidden[-2] + hidden[-1]
        gatt_features = self.globalAttention(final_hidden, bigru_out)

        fused = self.fusion(tcn_features, gatt_features)
        return self.fc(fused)

# =========================================================
# 数据与工具函数
# =========================================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def check_columns(df):
    required = [TIME_COL] + FEATURE_COLS + [TARGET_COL]
    return [c for c in required if c not in df.columns]


def fit_minmax(df, cols):
    stats = {}
    for c in cols:
        cmin = float(df[c].min())
        cmax = float(df[c].max())
        stats[c] = (cmin, cmax)
    return stats


def transform_minmax(df, cols, stats):
    out = df.copy()
    for c in cols:
        cmin, cmax = stats[c]
        if abs(cmax - cmin) < 1e-12:
            out[c] = 0.0
        else:
            out[c] = (out[c] - cmin) / (cmax - cmin)
    return out


def inverse_minmax(arr, cmin, cmax):
    return arr * (cmax - cmin) + cmin


@st.cache_resource
def load_model():
    model = TSEBG(
        input_dim=14,
        output_dim=1,
        num_channels=[32, 64, 64],
        kernel_size=3,
        hidden_layer_sizes=[64],
        attention_dim=64,
        dropout=0.2,
        embed_dim=64,
        tcn_mha_heads=4,
        mha_dropout=0.1,
        attn_out_dropout=0.1,
        use_tcn_mha=False,
        fusion_type="gated",
        se_type="mhse",
        mhse_heads=2,
        se_reduction=8
    )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_single_step(model, window_np):
    with torch.no_grad():
        x = torch.tensor(window_np[np.newaxis, :, :], dtype=torch.float32)
        pred = model(x).squeeze(-1).cpu().numpy()[0]
    return float(pred)


def recompute_engineered_features(history_df):
    df = history_df.copy().reset_index(drop=True)

    df["hour"] = pd.to_datetime(df[TIME_COL]).dt.hour
    df["weekday"] = pd.to_datetime(df[TIME_COL]).dt.dayofweek
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["accident_ma_3"] = df["accident_count"].rolling(window=3, min_periods=1).mean()
    df["accident_ma_6"] = df["accident_count"].rolling(window=6, min_periods=1).mean()
    df["accident_lag_24"] = df["accident_count"].shift(24).bfill()
    df["accident_lag_168"] = df["accident_count"].shift(168).bfill()

    return df


def build_future_prediction(base_df, base_time, horizon, model):
    used_df = base_df[base_df[TIME_COL] <= base_time].copy().reset_index(drop=True)

    if len(used_df) < SEQ_LEN:
        raise ValueError(f"基准时点前数据不足，至少需要 {SEQ_LEN} 条历史记录。")

    scale_cols = FEATURE_COLS + [TARGET_COL]
    stats = fit_minmax(used_df, scale_cols)
    used_scaled = transform_minmax(used_df, scale_cols, stats)

    history_scaled = used_scaled.copy().reset_index(drop=True)
    future_records = []

    y_min, y_max = stats[TARGET_COL]
    count_min, count_max = stats["accident_count"]

    for _ in range(horizon):
        window = history_scaled[FEATURE_COLS].iloc[-SEQ_LEN:].values.astype(np.float32)
        pred_scaled = predict_single_step(model, window)
        pred_real = inverse_minmax(np.array([pred_scaled]), y_min, y_max)[0]

        last_time = pd.to_datetime(history_scaled.iloc[-1][TIME_COL])
        next_time = last_time + pd.Timedelta(hours=1)

        next_count_real = max(0.0, np.expm1(pred_real))
        if abs(count_max - count_min) < 1e-12:
            next_count_scaled = 0.0
        else:
            next_count_scaled = (next_count_real - count_min) / (count_max - count_min)

        new_row = history_scaled.iloc[-1].copy()
        new_row[TIME_COL] = next_time
        new_row[TARGET_COL] = pred_scaled
        new_row["accident_count"] = next_count_scaled

        history_scaled = pd.concat([history_scaled, pd.DataFrame([new_row])], ignore_index=True)

        history_real = history_scaled.copy()
        history_real["accident_count"] = inverse_minmax(history_real["accident_count"].values, count_min, count_max)
        history_real[TARGET_COL] = inverse_minmax(history_real[TARGET_COL].values, y_min, y_max)

        history_real = recompute_engineered_features(history_real)
        history_scaled = transform_minmax(history_real, scale_cols, stats)

        future_records.append({
            "time": next_time,
            "risk_index": float(pred_real),
            "accident_count_est": float(next_count_real)
        })

    future_df = pd.DataFrame(future_records)
    return used_df, future_df


def risk_level(value, low_th, high_th):
    if value < low_th:
        return "低风险", "🟢", "当前事故风险处于较低水平，建议保持常规巡查。"
    elif value < high_th:
        return "中风险", "🟡", "当前事故风险高于一般水平，建议加强重点时段巡查与疏导。"
    else:
        return "高风险", "🔴", "当前事故风险较高，建议加强警力布控并关注重点路段。"


def level_from_value(value, low_th, high_th):
    if value < low_th:
        return "低风险"
    elif value < high_th:
        return "中风险"
    return "高风险"


def show_risk_light(level):
    if level == "低风险":
        cls = "risk-box-green"
        text = "🟢 当前交通风险等级：低风险"
    elif level == "中风险":
        cls = "risk-box-yellow"
        text = "🟡 当前交通风险等级：中风险"
    else:
        cls = "risk-box-red"
        text = "🔴 当前交通风险等级：高风险"

    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def plot_input_window(input_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=input_df[TIME_COL],
        y=input_df["accident_count"],
        mode="lines+markers",
        name="事故数量"
    ))
    fig.update_layout(
        title="基准时点前48小时事故数量变化",
        xaxis_title="时间",
        yaxis_title="事故数量",
        height=360
    )
    return fig


def plot_feature_window(input_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=input_df[TIME_COL],
        y=input_df["injury_ratio"],
        mode="lines",
        name="受伤占比"
    ))
    fig.add_trace(go.Scatter(
        x=input_df[TIME_COL],
        y=input_df["abnormal_factor_ratio"],
        mode="lines",
        name="异常因素占比"
    ))
    fig.update_layout(
        title="基准时点前48小时关键输入特征变化",
        xaxis_title="时间",
        yaxis_title="特征值",
        height=360,
        legend=dict(orientation="h", y=1.08, x=0)
    )
    return fig


def plot_history_future(input_df, future_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=input_df[TIME_COL],
        y=input_df[TARGET_COL],
        mode="lines",
        name="历史风险指数"
    ))
    fig.add_trace(go.Scatter(
        x=future_df["time"],
        y=future_df["risk_index"],
        mode="lines+markers",
        name="未来预测风险",
        line=dict(dash="dash")
    ))
    fig.update_layout(
        title="历史风险与未来风险预测衔接图",
        xaxis_title="时间",
        yaxis_title="风险指数",
        height=400,
        legend=dict(orientation="h", y=1.08, x=0)
    )
    return fig


def plot_future_bar(future_df, low_th, high_th):
    colors = []
    for v in future_df["risk_index"]:
        if v < low_th:
            colors.append("#34c759")
        elif v < high_th:
            colors.append("#f5c542")
        else:
            colors.append("#ef5350")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=future_df["time"],
        y=future_df["risk_index"],
        marker_color=colors,
        name="风险指数"
    ))
    fig.update_layout(
        title="未来风险分时预测",
        xaxis_title="时间",
        yaxis_title="风险指数",
        height=400
    )
    return fig


def plot_risk_pie(future_df, low_th, high_th):
    levels = [level_from_value(v, low_th, high_th) for v in future_df["risk_index"]]
    s = pd.Series(levels).value_counts()
    order = ["低风险", "中风险", "高风险"]
    pie_df = pd.DataFrame({
        "风险等级": order,
        "数量": [int(s.get(x, 0)) for x in order]
    })

    fig = px.pie(
        pie_df,
        names="风险等级",
        values="数量",
        title="未来预测风险等级分布",
        hole=0.38
    )
    fig.update_layout(height=360)
    return fig


def make_high_risk_table(future_df, low_th, high_th):
    show_df = future_df.copy()
    show_df["风险等级"] = show_df["risk_index"].apply(lambda x: level_from_value(x, low_th, high_th))
    show_df["预测事故数"] = show_df["accident_count_est"].round(2)
    show_df["风险指数"] = show_df["risk_index"].round(4)
    show_df["时间"] = pd.to_datetime(show_df["time"]).dt.strftime("%Y-%m-%d %H:%M")
    high_df = show_df[show_df["风险等级"].isin(["高风险", "中风险"])][["时间", "风险指数", "预测事故数", "风险等级"]].copy()
    return high_df


def make_timeline_text(future_df, low_th, high_th):
    out = []
    for _, row in future_df.iterrows():
        lv = level_from_value(row["risk_index"], low_th, high_th)
        ts = pd.to_datetime(row["time"]).strftime("%m-%d %H:%M")
        if lv == "高风险":
            out.append(f"🔴 {ts}  高风险")
        elif lv == "中风险":
            out.append(f"🟡 {ts}  中风险")
    return out[:10]

# =========================================================
# 功能模块卡片
# =========================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="func-card">
        <div class="func-icon">🕒</div>
        <div class="func-title">历史回放预测</div>
        <div class="func-desc">可选择任意历史时点作为预测基准</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="func-card">
        <div class="func-icon">📈</div>
        <div class="func-title">未来风险趋势</div>
        <div class="func-desc">支持未来6/12/24小时动态预测</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="func-card">
        <div class="func-icon">🚨</div>
        <div class="func-title">风险等级预警</div>
        <div class="func-desc">自动识别高风险与中风险时段</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="func-card">
        <div class="func-icon">🛣️</div>
        <div class="func-title">辅助交通管理</div>
        <div class="func-desc">为巡查、疏导与预警提供参考</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

# =========================================================
# 系统说明
# =========================================================
st.markdown("""
<div class="soft-card">
    <div class="sub-title">系统功能说明</div>
    <div class="small-note">
        本系统支持选择任意历史时点作为预测基准，自动提取该时点前48小时交通事故历史数据，
        结合 TSEBG 深度学习模型，对未来若干小时事故风险进行动态预测，
        并通过风险等级划分、趋势图、时间轴和高风险时段识别实现可视化展示。
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 数据读取
# =========================================================
if not os.path.exists(DATA_PATH):
    st.error(f"未找到数据文件：{DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)
missing_cols = check_columns(df)
if missing_cols:
    st.error(f"数据缺少必要字段：{missing_cols}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"未找到模型文件：{MODEL_PATH}")
    st.stop()

# =========================================================
# 数据概览
# =========================================================
s1, s2, s3, s4 = st.columns(4)
s1.metric("数据总条数", len(df))
s2.metric("输入特征维度", len(FEATURE_COLS))
s3.metric("数据起始时间", df[TIME_COL].min().strftime("%Y-%m-%d"))
s4.metric("数据结束时间", df[TIME_COL].max().strftime("%Y-%m-%d"))
st.markdown("### 📊 输入特征说明")

feature_desc = {
    "accident_count": "事故数量（原始值）",
    "accident_log1p": "事故数量对数变换（预测目标）",
    "injury_ratio": "受伤占比",
    "death_ratio": "死亡占比",
    "abnormal_factor_ratio": "异常因素占比",
    "hour_sin": "小时周期编码（sin）",
    "hour_cos": "小时周期编码（cos）",
    "weekday_sin": "星期周期编码（sin）",
    "weekday_cos": "星期周期编码（cos）",
    "is_weekend": "是否周末",
    "accident_ma_3": "3小时滑动平均",
    "accident_ma_6": "6小时滑动平均",
    "accident_lag_24": "24小时前事故数",
    "accident_lag_168": "7天前事故数",
}

df_feature = pd.DataFrame({
    "特征名称": list(feature_desc.keys()),
    "含义说明": list(feature_desc.values())
})

st.table(df_feature)


st.markdown("---")

# =========================================================
# 预测设置
# =========================================================
st.markdown("""
<div class="soft-card">
    <div class="sub-title">预测设置</div>
    <div class="small-note">请选择历史基准时点与未来预测时长，系统将模拟该时点下的动态风险预测场景。</div>
</div>
""", unsafe_allow_html=True)

valid_times = df[TIME_COL].dropna().sort_values().reset_index(drop=True)
min_time = valid_times.iloc[SEQ_LEN]
max_time = valid_times.iloc[-25]

p1, p2, p3 = st.columns(3)

with p1:
    selected_date = st.date_input(
        "选择预测基准日期",
        value=max_time.date(),
        min_value=min_time.date(),
        max_value=max_time.date()
    )

with p2:
    hour_options = list(range(24))
    selected_hour = st.selectbox("选择预测基准小时", hour_options, index=int(max_time.hour))

with p3:
    horizon = st.selectbox("选择未来预测时长", [6, 12, 24], index=2)

selected_base_time = pd.Timestamp(
    year=selected_date.year,
    month=selected_date.month,
    day=selected_date.day,
    hour=int(selected_hour)
)

available_times = set(valid_times.dt.floor("h").tolist())
if selected_base_time not in available_times:
    nearest_time = valid_times[valid_times <= selected_base_time].max()
    if pd.isna(nearest_time):
        st.error("当前选择的基准时点无可用历史数据。")
        st.stop()
    selected_base_time = nearest_time
    st.warning(f"所选时点无精确数据，已自动调整为最近可用时点：{selected_base_time}")

run_btn = st.button("🚀 开始动态预测", use_container_width=True)

# =========================================================
# 执行预测
# =========================================================
if run_btn:
    try:
        model = load_model()
        used_df, future_df = build_future_prediction(df, selected_base_time, horizon, model)

        input_window_df = used_df.tail(SEQ_LEN).copy()

        low_th = float(np.quantile(used_df[TARGET_COL], 0.33))
        high_th = float(np.quantile(used_df[TARGET_COL], 0.66))

        current_risk = float(future_df["risk_index"].iloc[0])
        avg_future_risk = float(future_df["risk_index"].mean())
        peak_risk = float(future_df["risk_index"].max())
        peak_time = pd.to_datetime(future_df.loc[future_df["risk_index"].idxmax(), "time"])

        risk_text, risk_icon, suggestion = risk_level(current_risk, low_th, high_th)
        high_risk_hours = int((future_df["risk_index"] >= high_th).sum())

        st.session_state["used_df"] = used_df
        st.session_state["input_window_df"] = input_window_df
        st.session_state["future_df"] = future_df
        st.session_state["low_th"] = low_th
        st.session_state["high_th"] = high_th
        st.session_state["current_risk"] = current_risk
        st.session_state["avg_future_risk"] = avg_future_risk
        st.session_state["peak_risk"] = peak_risk
        st.session_state["peak_time"] = peak_time
        st.session_state["risk_text"] = risk_text
        st.session_state["risk_icon"] = risk_icon
        st.session_state["suggestion"] = suggestion
        st.session_state["high_risk_hours"] = high_risk_hours
        st.session_state["base_time"] = selected_base_time
        st.session_state["horizon"] = horizon

        st.success("预测完成！")

    except Exception as e:
        st.error(f"预测失败：{e}")

# =========================================================
# 结果展示
# =========================================================
if "future_df" in st.session_state:
    used_df = st.session_state["used_df"]
    input_window_df = st.session_state["input_window_df"]
    future_df = st.session_state["future_df"]
    low_th = st.session_state["low_th"]
    high_th = st.session_state["high_th"]

    st.markdown("---")

    st.markdown("""
    <div class="soft-card">
        <div class="sub-title">本次预测任务说明</div>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        f"本次预测以 **{st.session_state['base_time']}** 作为基准时点，"
        f"使用此前 **48小时历史数据** 作为模型输入，预测未来 **{st.session_state['horizon']} 小时** 的事故风险变化。"
    )

    st.markdown("---")

    st.markdown("""
    <div class="soft-card">
        <div class="sub-title">核心预测结果</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("当前风险等级", st.session_state["risk_text"])
    k2.metric("未来首时刻风险指数", f"{st.session_state['current_risk']:.4f}")
    k3.metric("未来平均风险指数", f"{st.session_state['avg_future_risk']:.4f}")
    k4.metric("高风险时段数", f"{st.session_state['high_risk_hours']} 小时")

    k5, k6 = st.columns(2)
    with k5:
        st.metric("峰值风险指数", f"{st.session_state['peak_risk']:.4f}")
    with k6:
        st.metric("峰值出现时间", st.session_state["peak_time"].strftime("%Y-%m-%d %H:%M"))

    show_risk_light(st.session_state["risk_text"])

    st.markdown("### 预警说明")
    st.warning(
        f"{st.session_state['risk_icon']} 当前预警等级为：{st.session_state['risk_text']}。"
        f"{st.session_state['suggestion']}"
    )

    st.markdown("---")

    st.markdown("""
    <div class="soft-card">
        <div class="sub-title">输入窗口可视化</div>
        <div class="small-note">展示当前基准时点前 48 小时的事故历史变化与关键输入特征，用于体现模型预测依据。</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_input_window(input_window_df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_feature_window(input_window_df), use_container_width=True)

    st.markdown("---")

    st.markdown("""
    <div class="soft-card">
        <div class="sub-title">未来风险预测结果</div>
        <div class="small-note">从历史窗口出发，对未来交通事故风险进行滚动预测，并识别高风险时段。</div>
    </div>
    """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_history_future(input_window_df, future_df), use_container_width=True)
    with c4:
        st.plotly_chart(plot_future_bar(future_df, low_th, high_th), use_container_width=True)

    c5, c6 = st.columns([1, 1])

    with c5:
        st.plotly_chart(plot_risk_pie(future_df, low_th, high_th), use_container_width=True)

    with c6:
        st.markdown("""
        <div class="soft-card">
            <div class="sub-title">结果解读</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("**1. 系统预测内容**")
        st.write("本系统预测的是未来小时级交通事故风险指数，而不是事故类型分类。")

        st.write("**2. 风险指数含义**")
        st.write("风险指数越高，表示该时段发生交通事故的可能性越大。")

        st.write("**3. 动态预测机制**")
        st.write("系统支持选择任意历史时点作为基准，模拟真实业务中的时点预测场景。")

        st.write("**4. 管理建议**")
        st.write(st.session_state["suggestion"])

        timeline_items = make_timeline_text(future_df, low_th, high_th)
        st.write("**5. 风险时间轴**")
        if timeline_items:
            for item in timeline_items:
                st.write(item)
        else:
            st.write("未来预测区间内未识别出中高风险时段。")

    st.markdown("---")

    st.markdown("""
    <div class="soft-card">
        <div class="sub-title">高风险 / 中风险时段识别</div>
    </div>
    """, unsafe_allow_html=True)

    high_table = make_high_risk_table(future_df, low_th, high_th)
    if len(high_table) == 0:
        st.success("未来预测区间内未识别出中高风险时段。")
    else:
        st.dataframe(high_table, use_container_width=True)

    st.markdown("---")

    st.markdown("""
    <div class="soft-card">
        <div class="sub-title">未来预测明细</div>
    </div>
    """, unsafe_allow_html=True)

    detail_df = future_df.copy()
    detail_df["风险等级"] = detail_df["risk_index"].apply(lambda x: level_from_value(x, low_th, high_th))
    detail_df["time"] = pd.to_datetime(detail_df["time"]).dt.strftime("%Y-%m-%d %H:%M")
    detail_df["risk_index"] = detail_df["risk_index"].round(4)
    detail_df["accident_count_est"] = detail_df["accident_count_est"].round(2)
    detail_df = detail_df.rename(columns={
        "time": "时间",
        "risk_index": "预测风险指数",
        "accident_count_est": "预测事故数",
    })

    st.dataframe(detail_df, use_container_width=True)

    csv_bytes = detail_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="下载本次预测结果 CSV",
        data=csv_bytes,
        file_name="dynamic_prediction_result.csv",
        mime="text/csv",
        use_container_width=True
    )