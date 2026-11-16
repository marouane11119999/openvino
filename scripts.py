import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DEVICE_ORDER = ["CPU", "GPU", "NPU"]
MODE_ORDER = ["latency", "throughput"]


# ============================================================
# Data preparation
# ============================================================

def prepare_benchmark_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the OpenVINO benchmark dataframe.

    Expected column:
        benchmark_config
            e.g. CPU_latency, GPU_latency, NPU_throughput
    """

    data = df.copy()

    # Split benchmark_config into device + benchmark mode
    split = data["benchmark_config"].str.split("_", n=1, expand=True)

    data["device"] = split[0]
    data["mode"] = split[1]

    # Keep a predictable ordering
    data["device"] = pd.Categorical(
        data["device"],
        categories=DEVICE_ORDER,
        ordered=True,
    )

    data["mode"] = pd.Categorical(
        data["mode"],
        categories=MODE_ORDER,
        ordered=True,
    )

    return data.sort_values(["device", "mode"])


# ============================================================
# Generic chart formatting
# ============================================================

def format_figure(
    fig: go.Figure,
    title: str,
    y_title: str,
    x_title: str = "Device",
) -> go.Figure:
    """
    Apply common styling to all benchmark plots.
    """

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        font=dict(size=13),
        legend_title="Benchmark mode",
        hovermode="x unified",
        margin=dict(l=70, r=30, t=80, b=70),
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=DEVICE_ORDER,
    )

    return fig


# ============================================================
# 1. Latency
# ============================================================

def plot_latency(df: pd.DataFrame) -> go.Figure:
    """
    Plot average latency for CPU, GPU and NPU.

    Uses:
        avg latency
    """

    fig = px.bar(
        df,
        x="device",
        y="avg latency",
        color="mode",
        barmode="group",
        category_orders={
            "device": DEVICE_ORDER,
            "mode": MODE_ORDER,
        },
        labels={
            "avg latency": "Average latency (ms)",
            "device": "Device",
            "mode": "Benchmark mode",
        },
        text="avg latency",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    return format_figure(
        fig,
        "ResNet50 — Average Latency",
        "Average latency (ms)",
    )


# ============================================================
# 2. Throughput
# ============================================================

def plot_throughput(df: pd.DataFrame) -> go.Figure:
    """
    Plot throughput for CPU, GPU and NPU.
    """

    fig = px.bar(
        df,
        x="device",
        y="throughput",
        color="mode",
        barmode="group",
        category_orders={
            "device": DEVICE_ORDER,
            "mode": MODE_ORDER,
        },
        labels={
            "throughput": "Throughput (FPS)",
            "device": "Device",
            "mode": "Benchmark mode",
        },
        text="throughput",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    return format_figure(
        fig,
        "ResNet50 — Throughput",
        "Throughput (FPS)",
    )


# ============================================================
# 3. Compile time
# ============================================================

def plot_compile_time(df: pd.DataFrame) -> go.Figure:
    """
    Plot model compilation time.
    """

    fig = px.bar(
        df,
        x="device",
        y="compile model time (ms)",
        color="mode",
        barmode="group",
        category_orders={
            "device": DEVICE_ORDER,
            "mode": MODE_ORDER,
        },
        labels={
            "compile model time (ms)": "Compile time (ms)",
            "device": "Device",
            "mode": "Benchmark mode",
        },
        text="compile model time (ms)",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    return format_figure(
        fig,
        "ResNet50 — Model Compilation Time",
        "Compile time (ms)",
    )


# ============================================================
# 4. First inference time
# ============================================================

def plot_first_inference(df: pd.DataFrame) -> go.Figure:
    """
    Plot first inference time.
    """

    fig = px.bar(
        df,
        x="device",
        y="first inference time (ms)",
        color="mode",
        barmode="group",
        category_orders={
            "device": DEVICE_ORDER,
            "mode": MODE_ORDER,
        },
        labels={
            "first inference time (ms)": "First inference time (ms)",
            "device": "Device",
            "mode": "Benchmark mode",
        },
        text="first inference time (ms)",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    return format_figure(
        fig,
        "ResNet50 — First Inference Time",
        "First inference time (ms)",
    )


def plot_latency_stability(df: pd.DataFrame) -> go.Figure:
    """
    Plot latency variance (Min, Avg, Max) to visualize stability.
    """
    # Filter for latency mode to evaluate true steady-state stability
    latency_df = df[df["mode"] == "latency"].copy()

    fig = go.Figure()

    for device in DEVICE_ORDER:
        dev_data = latency_df[latency_df["device"] == device]
        if dev_data.empty:
            continue
            
        avg = dev_data["avg latency"].values[0]
        min_val = dev_data["min latency"].values[0]
        max_val = dev_data["max latency"].values[0]

        fig.add_trace(go.Scatter(
            x=[device],
            y=[avg],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[max_val - avg],
                arrayminus=[avg - min_val],
                thickness=2,
                width=10
            ),
            mode='markers',
            marker=dict(size=12),
            name=f"{device} (Avg ± Min/Max)"
        ))

    fig.update_layout(
        title={"text": "ResNet50 — Latency Jitter & Range (Min / Avg / Max)", "x": 0.5},
        xaxis_title="Device",
        yaxis_title="Latency (ms)",
        template="plotly_white",
    )
    return fig

def plot_efficiency_tradeoff(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot comparing Throughput (FPS) vs Average Latency (ms).
    Top-left quadrant = Optimal (High FPS, Low Latency).
    """
    fig = px.scatter(
        df,
        x="avg latency",
        y="throughput",
        color="device",
        symbol="mode",
        size_max=15,
        hover_data=["benchmark_config"],
        labels={
            "avg latency": "Average Latency (ms) [Lower is better]",
            "throughput": "Throughput (FPS) [Higher is better]",
            "device": "Device",
            "mode": "Hint Mode"
        },
        category_orders={"device": DEVICE_ORDER}
    )

    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))

    return format_figure(
            fig,
            title="ResNet50 — Throughput vs. Latency Trade-off",
            y_title="Throughput (FPS)",
            x_title="Average Latency (ms) [Lower is better]", 
                        
        )


# ============================================================
# Generate all plots
# ============================================================

def create_benchmark_plots(df: pd.DataFrame) -> dict[str, go.Figure]:
    """
    Create all ResNet50 benchmark plots.

    Returns
    -------
    dict
        Dictionary containing the four Plotly figures.
    """

    data = prepare_benchmark_data(df)

    return {
        # Core Performance
        "latency": plot_latency(data),
        "throughput": plot_throughput(data),
        
        # Overhead & Startup
        "compile_time": plot_compile_time(data),
        "first_inference": plot_first_inference(data),
        
        # Stability & Trade-offs
        "latency_stability": plot_latency_stability(data),
        "tradeoff_scatter": plot_efficiency_tradeoff(data),
    }


def preprocess(path: Path) -> pd.DataFrame:
    dataframes = {}
    for path in Path(path).iterdir():
        if path.is_dir():
            parent_dir = path.parent
            parts = path.name.split('_')
            
            device, hint = parts[1], parts[2]
            report_name = "benchmark_report.csv"
            report = pd.read_csv(str(path) + "/" + report_name,skiprows=22,sep=";")  
            dataframes.update({device + "_" + hint: report})
    processed_dfs = []
    for key, df in dataframes.items():
        df_transposed = df.T
        df_transposed.columns = df_transposed.iloc[0,:]
        df_transposed = df_transposed[1:]
        df_transposed['benchmark_config'] = key 
        processed_dfs.append(df_transposed) 
        
    final_df = pd.concat(processed_dfs, ignore_index=False)
    return final_df


def prepare_benchmark(df):
    df = df.copy()

    # Keep only actual iterations, excluding warm-up and summaries
    df = df[df["iteration"].astype(str).str.isdigit()].copy()
    df["iteration"] = df["iteration"].astype(int)

    # Normalize common CSV-export formatting
    df["device"] = df["device"].astype(str).str.strip("* ")
    df["model"] = df["model"].astype(str).str.strip("* ")
    df["framework"] = df["framework"].astype(str).str.strip("* ")

    # Convert metrics to numeric
    numeric_cols = [
        "pretrain_time(s)",
        "input_size",
        "output_size",
        "generation_time(s)",
        "latency(ms)",
        "1st_latency(ms)",
        "2nd_avg_latency(ms)",
        "1st_infer_latency(ms)",
        "2nd_infer_avg_latency(ms)",
        "initial_rss_mem(MiB)",
        "max_rss_mem(MiB)",
        "max_increase_rss_mem(MiB)",
        "max_sys_mem(MiB)",
        "max_increase_sys_mem(MiB)",
        "tokenization_time",
        "detokenization_time",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Useful derived metrics
    if "output_size" in df and "generation_time(s)" in df:
        df["tokens_per_sec"] = (
            df["output_size"] / df["generation_time(s)"]
        )

    return df





def clean_benchmark(df):
    df = df.copy()

    # Clean column names such as **max_rss_mem(MiB)
    df.columns = [
        str(c).strip().strip("*").strip()
        for c in df.columns
    ]

    for col in ["model", "framework", "device", "iteration"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.strip("*").str.strip()
            )

    # Keep measured iterations; exclude warm-up and summary rows
    df = df[df["iteration"].str.fullmatch(r"\d+")].copy()
    df["iteration"] = df["iteration"].astype(int)
    df = df[df["iteration"] > 0].copy()

    metrics = [
        "pretrain_time(s)", "input_size", "output_size",
        "generation_time(s)", "latency(ms)", "1st_latency(ms)",
        "2nd_avg_latency(ms)", "max_rss_mem(MiB)",
        "max_increase_rss_mem(MiB)", "max_sys_mem(MiB)",
        "max_increase_sys_mem(MiB)",
    ]

    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"output_size", "generation_time(s)"} <= set(df.columns):
        df["tokens_per_sec"] = (
            df["output_size"] / df["generation_time(s)"]
        )

    return df



import matplotlib.pyplot as plt
import seaborn as sns

def plot_generation_throughput(data):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data,
        x="model",
        y="tokens_per_sec",
        hue="device",
        errorbar=("ci", 95),
    )

    plt.ylabel("Generation throughput (tokens/sec)")
    plt.xlabel("Model")
    plt.title("LLM generation throughput by device")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()



def plot_startup_cost(dfs):
    startup_frames = []

    for name, df in dfs.items():
        d = df.copy()

        # Clean column names
        d.columns = [
            str(c).strip().strip("*").strip()
            for c in d.columns
        ]

        # Clean relevant columns
        for col in ["device", "model", "iteration", "pretrain_time(s)"]:
            if col in d.columns:
                d[col] = (
                    d[col]
                    .astype(str)
                    .str.strip()
                    .str.strip("*")
                    .str.strip()
                )

        # Convert startup time to numeric
        d["pretrain_time(s)"] = pd.to_numeric(
            d["pretrain_time(s)"], errors="coerce"
        )

        # Keep numeric iteration rows, including warm-up row 0
        d = d[
            d["iteration"].str.fullmatch(r"\d+", na=False)
        ].copy()

        d = d[d["iteration"].astype(int) == 0]

        # Remove rows without startup time
        d = d.dropna(subset=["pretrain_time(s)"])

        startup_frames.append(d)

    # Combine all DataFrames
    if not startup_frames:
        print("No startup data found.")
        return

    startup = pd.concat(startup_frames, ignore_index=True)
    
    if startup.empty:
        print("No valid startup rows found.")
        return

    # Plot startup cost
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=startup,
        x="model",
        y="pretrain_time(s)",
        hue="device",
        errorbar=None,
    )

    plt.ylabel("Load + compile time (seconds)")
    plt.xlabel("Model")
    plt.title("LLM startup cost by device")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()



def plot_latency(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data,
        x="device",
        y="latency(ms)",
        hue="device",
        legend=False,
    )
    plt.ylabel("Latency (ms/token)")
    plt.title("Token latency distribution by device")
    plt.tight_layout()
    plt.show()
    
def plot_ttft(dfs):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    frames = []

    for name, df in dfs.items():
        d = df.copy()

        # Clean column names
        d.columns = [
            str(c).strip().strip("*").strip()
            for c in d.columns
        ]

        required = [
            "model", "device", "iteration",
            "1st_latency(ms)", "input_size"
        ]

        if not all(col in d.columns for col in required):
            continue

        # Clean string columns
        for col in ["model", "device", "iteration"]:
            d[col] = (
                d[col].astype(str)
                .str.strip()
                .str.strip("*")
                .str.strip()
            )

        # Convert to numeric
        d["1st_latency(ms)"] = pd.to_numeric(
            d["1st_latency(ms)"], errors="coerce"
        )
        d["input_size"] = pd.to_numeric(
            d["input_size"], errors="coerce"
        )

        # Exclude warm-up iteration
        d = d[
            d["iteration"].str.fullmatch(r"\d+", na=False)
        ].copy()

        d = d[d["iteration"].astype(int) > 0]

        # Calculate TTFT in milliseconds
        d["TTFT(ms)"] = (
            d["1st_latency(ms)"] * d["input_size"]
        )

        d = d.dropna(subset=["TTFT(ms)"])
        frames.append(d)

    if not frames:
        print("No valid TTFT data found.")
        return

    data = pd.concat(frames, ignore_index=True)

    # Average TTFT across measured iterations
    summary = (
        data.groupby(["model", "device"], as_index=False)
        ["TTFT(ms)"].mean()
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=summary,
        x="model",
        y="TTFT(ms)",
        hue="device",
        errorbar=None,
    )

    plt.ylabel("TTFT (ms)")
    plt.xlabel("Model")
    plt.title("Time to First Token (TTFT) by Model and Device")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()



def plot_tpot(dfs):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    frames = []

    for name, df in dfs.items():
        d = df.copy()

        # Clean column names
        d.columns = [
            str(c).strip().strip("*").strip()
            for c in d.columns
        ]

        required = [
            "model", "device", "iteration",
            "2nd_avg_latency(ms)"
        ]

        if not all(col in d.columns for col in required):
            continue

        # Clean string columns
        for col in ["model", "device", "iteration"]:
            d[col] = (
                d[col].astype(str)
                .str.strip()
                .str.strip("*")
                .str.strip()
            )

        # Convert TPOT to numeric
        d["2nd_avg_latency(ms)"] = pd.to_numeric(
            d["2nd_avg_latency(ms)"], errors="coerce"
        )

        # Exclude warm-up iteration
        d = d[
            d["iteration"].str.fullmatch(r"\d+", na=False)
        ].copy()

        d = d[d["iteration"].astype(int) > 0]

        d = d.dropna(subset=["2nd_avg_latency(ms)"])
        frames.append(d)

    if not frames:
        print("No valid TPOT data found.")
        return

    data = pd.concat(frames, ignore_index=True)

    # Average TPOT across measured iterations
    summary = (
        data.groupby(["model", "device"], as_index=False)
        ["2nd_avg_latency(ms)"].mean()
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=summary,
        x="model",
        y="2nd_avg_latency(ms)",
        hue="device",
        errorbar=None,
    )

    plt.ylabel("TPOT (ms/token)")
    plt.xlabel("Model")
    plt.title("Time Per Output Token (TPOT) by Model and Device")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()