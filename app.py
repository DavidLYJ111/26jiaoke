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
    max-width: 1380px;
}

.main {
    background: linear-gradient(180deg, #f4f8fc 0%, #eef4fa 100%);
}

.banner {
    width: 100%;
    padding: 30px 34px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 12px 32px rgba(15,32,39,0.20);
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
    padding: 18px 20px;
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(30,60,90,0.08);
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
    margin: 6px 0 10px 0;
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
    background: linear-gradient(90deg, #d48a00, #f5b942);
    color: white;
    border-radius: 14px;
    padding: 18px;
    font-weight: 700;
    font-size: 22px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(212,138,0,0.22);
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

.alert-danger {
    background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 55%, #ef4444 100%);
    color: #fff;
    border-radius: 18px;
    padding: 18px 20px;
    border-left: 8px solid #fecaca;
    box-shadow: 0 10px 24px rgba(185,28,28,0.26);
    margin: 10px 0 16px 0;
}

.alert-danger-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 6px;
}

.kpi-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 8px 18px rgba(30,60,90,0.08);
    border: 1px solid rgba(20,60,120,0.06);
    margin-bottom: 12px;
}

.kpi-label {
    font-size: 13px;
    color: #607080;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: #17324d;
}

.kpi-sub {
    font-size: 12px;
    color: #7a8b99;
    margin-top: 6px;
}

.card-title-inline {
    font-size: 18px;
    font-weight: 800;
    color: #17324d;
    margin-bottom: 8px;
}

.badge-green, .badge-yellow, .badge-red {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    color: white;
}

.badge-green { background: #1f9d55; }
.badge-yellow { background: #d48a00; }
.badge-red { background: #d32f2f; }

.section-gap {
    margin-top: 8px;
    margin-bottom: 10px;
}

hr {
    margin: 0.8rem 0 1rem 0;
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
    <p>支持历史时点选择、未来风险预测、风险等级预警、风险解释与智能决策建议</p>
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
        return "中风险", "🟠", "当前事故风险高于一般水平，建议加强重点时段巡查与疏导。"
    else:
        return "高风险", "🔴", "当前事故风险较高，建议加强警力布控并关注重点路段。"


def level_from_value(value, low_th, high_th):
    if value < low_th:
        return "低风险"
    elif value < high_th:
        return "中风险"
    return "高风险"


def get_level_badge(level):
    if level == "低风险":
        return '<span class="badge-green">低风险</span>'
    if level == "中风险":
        return '<span class="badge-yellow">中风险</span>'
    return '<span class="badge-red">高风险</span>'


def show_risk_light(level):
    if level == "低风险":
        cls = "risk-box-green"
        text = "🟢 当前交通风险等级：低风险"
    elif level == "中风险":
        cls = "risk-box-yellow"
        text = "🟠 当前交通风险等级：中风险"
    else:
        cls = "risk-box-red"
        text = "🔴 当前交通风险等级：高风险"

    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def infer_time_period(ts):
    hour = int(pd.to_datetime(ts).hour)
    if 7 <= hour <= 9:
        return "早高峰"
    if 17 <= hour <= 19:
        return "晚高峰"
    if 22 <= hour or hour <= 5:
        return "夜间"
    return "平峰"


def calc_trend_label(future_df):
    if len(future_df) < 2:
        return "平稳", 0.0
    first_val = float(future_df["risk_index"].iloc[0])
    last_val = float(future_df["risk_index"].iloc[-1])
    delta = last_val - first_val
    if delta > 0.08:
        return "上升", delta
    if delta < -0.08:
        return "下降", delta
    return "平稳", delta


def get_time_period_measures(time_period):
    period_map = {
        "早高峰": [
            "早高峰优先保障通勤主干道、学校周边和轨交换乘节点通行效率。",
            "在上游路口提前分流，降低排队回溢引发的并道冲突风险。"
        ],
        "晚高峰": [
            "晚高峰加强商圈、环线匝道和过江通道的定点疏导。",
            "结合返程潮汐流向，动态优化关键路口放行相位与配时。"
        ],
        "夜间": [
            "夜间重点关注照明不足、施工路段与大型车辆混行区域。",
            "增加酒驾、疲劳驾驶高发时段巡查频次，提前压降突发风险。"
        ],
        "平峰": [
            "平峰阶段以精细化巡检为主，及时处置违停和短时拥堵点。",
            "维持路网稳定运行，为下一高峰时段预留调控余量。"
        ]
    }
    return period_map.get(time_period, period_map["平峰"])


def generate_decision_suggestions(level, time_period, trend_label, high_risk_hours):
    strategy_map = {
        "低风险": {
            "title": f"🚦 {time_period}低风险交通运行建议",
            "measures": [
                "保持常态化路面巡查，重点关注学校、医院和主干道出入口。",
                "维持现有信号配时与诱导策略，持续监测风险波动。"
            ]
        },
        "中风险": {
            "title": f"⚠️ {time_period}中风险交通管控建议",
            "measures": [
                "加强重点路口与事故易发路段的视频巡检和现场疏导。",
                "适度优化信号配时，提升瓶颈路段通行效率，减少排队冲突。"
            ]
        },
        "高风险": {
            "title": f"🚨 {time_period}高风险应急处置建议",
            "measures": [
                "立即对高风险路段进行重点布控，必要时增派交警与清障力量。",
                "结合拥堵与事故热点位置，启动重点路口渠化与分流预案。"
            ]
        }
    }

    trend_advice = {
        "上升": "当前风险呈上升趋势，建议将巡查与疏导资源向未来 2~4 小时倾斜配置。",
        "下降": "当前风险整体趋于回落，可维持重点关注并逐步恢复常态化管理。",
        "平稳": "当前风险波动相对平稳，建议持续监测，避免局部风险突然抬升。"
    }

    base_plan = strategy_map[level]
    measures = []
    measures.extend(base_plan["measures"])
    measures.extend(get_time_period_measures(time_period)[:1])
    measures.append(trend_advice[trend_label])

    if high_risk_hours >= 3 and level != "低风险":
        measures.append("未来连续中高风险时段较多，建议将警力部署从单点响应升级为分区联动响应。")

    # 统一输出 3~5 条
    measures = measures[:5]
    if len(measures) < 3:
        measures.append("保持跨部门信息共享，按小时复核风险变化并动态调整策略。")

    return {
        "title": base_plan["title"],
        "measures": measures
    }


def generate_risk_explanation(input_window_df, future_df, low_th, high_th):
    latest = input_window_df.iloc[-1]
    current_risk = float(future_df["risk_index"].iloc[0])
    risk_lv = "低风险" if current_risk < low_th else ("中风险" if current_risk < high_th else "高风险")

    accident_count = float(latest.get("accident_count", 0.0))
    injury_ratio = float(latest.get("injury_ratio", 0.0))
    death_ratio = float(latest.get("death_ratio", 0.0))
    abnormal_ratio = float(latest.get("abnormal_factor_ratio", 0.0))
    ma3 = float(latest.get("accident_ma_3", accident_count))
    lag24 = float(latest.get("accident_lag_24", accident_count))

    reasons = []
    if accident_count >= ma3:
        reasons.append("最近1小时事故数高于或接近短时平均水平")
    else:
        reasons.append("最近1小时事故数低于短时平均水平")

    if injury_ratio >= input_window_df["injury_ratio"].median():
        reasons.append("受伤占比偏高，说明事故后果严重程度有所抬升")
    else:
        reasons.append("受伤占比整体可控，说明事故严重程度没有明显放大")

    if abnormal_ratio >= input_window_df["abnormal_factor_ratio"].median():
        reasons.append("异常因素占比偏高，表明当前交通环境扰动较明显")
    else:
        reasons.append("异常因素占比不高，说明外部扰动相对有限")

    if accident_count >= lag24:
        reasons.append("与24小时前相比，当前事故活跃度没有下降")
    else:
        reasons.append("与24小时前相比，当前事故活跃度有所缓和")

    if death_ratio > 0:
        reasons.append("当前样本中存在死亡占比，系统会进一步提高风险敏感度")

    intro = f"当前系统判定为{risk_lv}，主要依据不是单一指标，而是事故数量、伤亡比例、异常因素占比以及历史时序变化的综合结果。"
    body = "；".join(reasons[:4]) + "。"
    closing = "这意味着系统不仅在看‘事故有没有发生’，也在看‘事故是否更严重、是否持续走高’。"
    return intro + body + closing


def generate_emergency_actions(time_period):
    return [
        f"针对{time_period}重点路口，立即安排现场值守与快速处置力量。",
        "通过导航平台、广播和诱导屏同步发布高风险提示与绕行信息。",
        "检查事故快撤、清障和医疗联动机制，确保突发情况 15 分钟内响应。",
        "对视距不足、混行严重或拥堵加剧路段采取临时限速与交通分流。"
    ]


def build_summary_metrics(future_df, low_th, high_th):
    current_risk = float(future_df["risk_index"].iloc[0])
    avg_future_risk = float(future_df["risk_index"].mean())
    peak_risk = float(future_df["risk_index"].max())
    peak_time = pd.to_datetime(future_df.loc[future_df["risk_index"].idxmax(), "time"])
    high_risk_hours = int((future_df["risk_index"] >= high_th).sum())
    risk_text, risk_icon, suggestion = risk_level(current_risk, low_th, high_th)
    trend_label, trend_delta = calc_trend_label(future_df)
    time_period = infer_time_period(future_df["time"].iloc[0])

    return {
        "current_risk": current_risk,
        "avg_future_risk": avg_future_risk,
        "peak_risk": peak_risk,
        "peak_time": peak_time,
        "high_risk_hours": high_risk_hours,
        "risk_text": risk_text,
        "risk_icon": risk_icon,
        "suggestion": suggestion,
        "trend_label": trend_label,
        "trend_delta": trend_delta,
        "time_period": time_period
    }


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
    history_risk = 1 / (1 + np.exp(-input_df[TARGET_COL]))

    future_df["risk_index"] = future_df["risk_index"].clip(0, 1)
    future_risk = future_df["risk_index"]

    fig.add_trace(go.Scatter(
        x=input_df[TIME_COL],
        y=history_risk,
        mode="lines",
        name="历史风险指数"
    ))

    fig.add_trace(go.Scatter(
        x=future_df["time"],
        y=future_risk,
        mode="lines+markers",
        name="未来预测风险",
        line=dict(dash="dash")
    ))

    x_val = input_df[TIME_COL].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[x_val, x_val],
        y=[
            min(input_df[TARGET_COL].min(), future_df["risk_index"].min()),
            max(input_df[TARGET_COL].max(), future_df["risk_index"].max())
        ],
        mode="lines",
        name="预测起点",
        line=dict(dash="dot", color="gray")
    ))

    fig.update_layout(
        title="历史风险与未来预测衔接图（风险指数）",
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
            colors.append("#f5b942")
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
        hole=0.42,
        color="风险等级",
        color_discrete_map={
            "低风险": "#34c759",
            "中风险": "#f5b942",
            "高风险": "#ef5350"
        }
    )
    fig.update_layout(height=360)
    return fig


def plot_risk_trend_line(future_df, low_th, high_th):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_df["time"],
        y=future_df["risk_index"],
        mode="lines+markers",
        name="未来风险趋势"
    ))
    fig.add_hline(y=low_th, line_dash="dot", line_color="#34c759", annotation_text="低风险阈值")
    fig.add_hline(y=high_th, line_dash="dot", line_color="#ef5350", annotation_text="高风险阈值")
    fig.update_layout(
        title="未来24小时风险趋势图",
        xaxis_title="时间",
        yaxis_title="风险指数",
        height=380
    )
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
            out.append(f"🟠 {ts}  中风险")
    return out[:10]


def risk_color_rgb(value, low_th, high_th):
    if value < low_th:
        return [0, 200, 0]
    if value < high_th:
        return [255, 165, 0]
    return [255, 0, 0]


def build_risk_map_df(future_df, low_th, high_th, center_lat=40.7, center_lon=-74.0, seed=42):
    map_df = future_df.copy()
    np.random.seed(seed)
    map_df["lat"] = center_lat + np.random.normal(0, 0.05, len(map_df))
    map_df["lon"] = center_lon + np.random.normal(0, 0.05, len(map_df))
    map_df["color"] = map_df["risk_index"].apply(lambda v: risk_color_rgb(v, low_th, high_th))
    return map_df


def render_risk_map(future_df, low_th, high_th):
    import pydeck as pdk

    center_lat = 40.7
    center_lon = -74.0
    map_df = build_risk_map_df(future_df, low_th, high_th, center_lat=center_lat, center_lon=center_lon)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=200,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10,
        pitch=40,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "风险指数: {risk_index}"}
    )
    st.pydeck_chart(deck)


def render_card_title(title, note=None):
    st.markdown(f"""
    <div class="soft-card">
        <div class="sub-title">{title}</div>
        {f'<div class="small-note">{note}</div>' if note else ''}
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(label, value, sub_text=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_strategy_card(plan):
    measures_html = "".join([f"<li>{m}</li>" for m in plan["measures"]])
    st.markdown(f"""
    <div class="soft-card">
        <div class="card-title-inline">{plan['title']}</div>
        <ul style="margin-top: 8px; line-height: 1.8; color: #304455;">
            {measures_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)


def render_emergency_block(emergency_actions):
    action_html = "".join([f"<li>{a}</li>" for a in emergency_actions])
    st.markdown(f"""
    <div class="alert-danger">
        <div class="alert-danger-title">🚨 高风险预警：建议立即启动应急响应</div>
        <div style="font-size:14px; opacity:0.95; margin-bottom:6px;">当前预测结果显示未来时段存在明显高风险波动，请优先保障重点路口、重点路段与应急联动资源。</div>
        <ul style="margin: 8px 0 0 18px; line-height: 1.8;">
            {action_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

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
        <div class="func-icon">⚠️</div>
        <div class="func-title">智能决策建议</div>
        <div class="func-desc">按风险等级、时段与趋势自动生成建议</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="func-card">
        <div class="func-icon">📊</div>
        <div class="func-title">风险解释说明</div>
        <div class="func-desc">对高低风险成因进行自然语言解释</div>
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
        并通过风险等级划分、趋势图、预警提示、风险解释与智能决策建议实现可视化展示。
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
render_card_title("预测控制区", "请选择历史基准时点与未来预测时长，系统将模拟该时点下的动态风险预测场景。")

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
        # ✅ 把未来风险全部映射到 0~1（关键！！）
        future_df["risk_index"] = 1 / (1 + np.exp(-future_df["risk_index"]))
        input_window_df = used_df.tail(SEQ_LEN).copy()

        low_th = float(np.quantile(used_df[TARGET_COL], 0.33))
        high_th = float(np.quantile(used_df[TARGET_COL], 0.66))

        summary = build_summary_metrics(future_df, low_th, high_th)
        # 先把未来风险映射到 0~1
        future_df["risk_index"] = 1 / (1 + np.exp(-future_df["risk_index"]))

        # 统一使用固定阈值
        low_th = 0.6
        high_th = 0.75

        # 基于映射后的风险重新生成 summary
        summary = build_summary_metrics(future_df, low_th, high_th)

        # 再强制统一当前风险等级
        if summary["current_risk"] < low_th:
            summary["risk_text"] = "低风险"
        elif summary["current_risk"] < high_th:
            summary["risk_text"] = "中风险"
        else:
            summary["risk_text"] = "高风险"

        decision_plan = generate_decision_suggestions(
            level=summary["risk_text"],
            time_period=summary["time_period"],
            trend_label=summary["trend_label"],
            high_risk_hours=summary["high_risk_hours"]
        )
        explanation_text = generate_risk_explanation(input_window_df, future_df, low_th, high_th)
        emergency_actions = generate_emergency_actions(summary["time_period"])

        st.session_state["used_df"] = used_df
        st.session_state["input_window_df"] = input_window_df
        st.session_state["future_df"] = future_df
        st.session_state["low_th"] = low_th
        st.session_state["high_th"] = high_th
        st.session_state["summary"] = summary
        st.session_state["decision_plan"] = decision_plan
        st.session_state["explanation_text"] = explanation_text
        st.session_state["emergency_actions"] = emergency_actions
        st.session_state["base_time"] = selected_base_time
        st.session_state["horizon"] = horizon

        st.success("预测完成！系统已生成风险趋势、智能决策建议与风险解释说明。")
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
    summary = st.session_state["summary"]
    decision_plan = st.session_state["decision_plan"]
    explanation_text = st.session_state["explanation_text"]
    emergency_actions = st.session_state["emergency_actions"]

    st.markdown("---")
    render_card_title(
        "本次预测任务说明",
        f"以 {st.session_state['base_time']} 作为基准时点，使用此前 48 小时历史数据作为模型输入，预测未来 {st.session_state['horizon']} 小时的事故风险变化。"
    )

    # 高风险预警强化
    if summary["risk_text"] == "高风险":
        render_emergency_block(emergency_actions)
        st.error("⚠️ 当前预测结果为高风险，请优先执行应急布控、现场疏导与信息发布。")
    elif summary["risk_text"] == "中风险":
        st.warning("⚠️ 当前预测结果为中风险，建议提前加强路面巡查与信号优化。")
    else:
        st.success("✅ 当前预测结果为低风险，道路运行总体平稳。")

    # 核心预测结果卡片
    render_card_title("核心预测结果", "统一采用低风险=绿色、中风险=橙色、高风险=红色的视觉表达。")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_kpi_card("当前风险等级", summary["risk_text"], f"时段类型：{summary['time_period']}")
    with r2:
        render_kpi_card("未来首时刻风险指数", f"{summary['current_risk']:.4f}", f"趋势判断：{summary['trend_label']}")
    with r3:
        render_kpi_card("未来平均风险指数", f"{summary['avg_future_risk']:.4f}", f"高风险时段：{summary['high_risk_hours']} 小时")
    with r4:
        render_kpi_card("峰值风险指数", f"{summary['peak_risk']:.4f}", summary["peak_time"].strftime("峰值时间：%m-%d %H:%M"))

    show_risk_light(summary["risk_text"])

    # 预警说明 + 风险解释
    a1, a2 = st.columns([1.05, 1.15])
    with a1:
        render_card_title("预警说明", "结合风险等级、未来变化趋势与高风险时段数量进行综合预警。")
        trend_text = f"未来风险整体呈{summary['trend_label']}趋势。"
        st.warning(f"{summary['risk_icon']} 当前预警等级为：{summary['risk_text']}。{summary['suggestion']}{trend_text}")
        st.markdown(f"当前页面状态：{get_level_badge(summary['risk_text'])}", unsafe_allow_html=True)

    with a2:
        render_card_title("风险解释模块", "面向非技术人员，用自然语言解释为什么当前风险高或低。")
        st.info(explanation_text)

    # 趋势图卡片
    render_card_title("风险趋势图", "保留原有趋势展示能力，并补充阈值线与更清晰的颜色表达。")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_history_future(input_window_df, future_df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_risk_trend_line(future_df, low_th, high_th), use_container_width=True)

    # 输入依据 + 未来分布
    render_card_title("模型输入依据与预测分布", "展示预测依据和未来等级分布，增强可解释性与展示完整度。")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_input_window(input_window_df), use_container_width=True)
        st.plotly_chart(plot_feature_window(input_window_df), use_container_width=True)
    with c4:
        st.plotly_chart(plot_future_bar(future_df, low_th, high_th), use_container_width=True)
        st.plotly_chart(plot_risk_pie(future_df, low_th, high_th), use_container_width=True)

    # 智能决策建议 + 时间轴
    d1, d2 = st.columns([1.2, 0.9])
    with d1:
        render_card_title("智能交通决策建议", "依据风险等级、时段特征和变化趋势自动生成 3~5 条策略。")
        render_strategy_card(decision_plan)
    with d2:
        render_card_title("风险时间轴", "用于快速定位未来中高风险出现时段。")
        timeline_items = make_timeline_text(future_df, low_th, high_th)
        if timeline_items:
            for item in timeline_items:
                st.write(item)
        else:
            st.success("未来预测区间内未识别出中高风险时段。")

    # 中高风险识别
    render_card_title("中高风险时段识别", "自动筛选未来预测中达到中风险和高风险等级的时段。")
    high_table = make_high_risk_table(future_df, low_th, high_th)
    if len(high_table) == 0:
        st.success("未来预测区间内未识别出中高风险时段。")
    else:
        st.dataframe(high_table, use_container_width=True)

    # =========================================================
    # 风险排行榜（新增模块）
    # =========================================================
    render_card_title("风险排行榜（TOP风险时段）", "按风险指数排序，快速识别未来最危险的时间段。")

    # 取TOP5高风险
    top_risk_df = future_df.sort_values("risk_index", ascending=False).head(5).copy()

    # 格式化展示
    top_risk_df["时间"] = pd.to_datetime(top_risk_df["time"]).dt.strftime("%m-%d %H:%M")
    top_risk_df["风险指数"] = top_risk_df["risk_index"].round(4)
    top_risk_df["预测事故数"] = top_risk_df["accident_count_est"].round(2)
    top_risk_df["风险等级"] = top_risk_df["risk_index"].apply(lambda x: level_from_value(x, low_th, high_th))

    # 只保留展示列
    top_risk_df = top_risk_df[["时间", "风险指数", "预测事故数", "风险等级"]]

    # 展示
    st.dataframe(top_risk_df, use_container_width=True)

    # 高亮提示
    st.info("📊 上表展示未来预测区间内风险最高的5个时段，可用于优先部署警力与交通管控。")

    # =========================================================
    # 风险地图（新增模块）
    # =========================================================
    render_card_title("城市交通风险地图（模拟）", "基于未来预测风险构建空间分布，用于直观展示风险热点区域。")

    import pydeck as pdk

    # 构造模拟经纬度（以城市中心为基准）
    center_lat = 40.7
    center_lon = -74.0

    map_df = future_df.copy()

    # 随机生成空间点（模拟城市分布）
    np.random.seed(42)
    map_df["lat"] = center_lat + np.random.normal(0, 0.05, len(map_df))
    map_df["lon"] = center_lon + np.random.normal(0, 0.05, len(map_df))


    # 风险颜色映射
    def get_color(val):
        if val < low_th:
            return [0, 200, 0]  # 绿色
        elif val < high_th:
            return [255, 165, 0]  # 橙色
        else:
            return [255, 0, 0]  # 红色


    map_df["color"] = map_df["risk_index"].apply(get_color)

    # 构建地图
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=200,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10,
        pitch=40,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "风险指数: {risk_index}"}
    )

    st.pydeck_chart(deck)

    # 明细结果
    render_card_title("未来预测明细", "支持比赛答辩时展示完整数据输出。")
    detail_df = future_df.copy()
    detail_df["风险等级"] = detail_df["risk_index"].apply(lambda x: level_from_value(x, low_th, high_th))
    detail_df["趋势判断"] = summary["trend_label"]
    detail_df["time"] = pd.to_datetime(detail_df["time"]).dt.strftime("%Y-%m-%d %H:%M")
    detail_df["risk_index"] = detail_df["risk_index"].round(4)
    detail_df["accident_count_est"] = detail_df["accident_count_est"].round(2)
    detail_df = detail_df.rename(columns={
        "time": "时间",
        "risk_index": "预测风险指数",
        "accident_count_est": "预测事故数"
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
