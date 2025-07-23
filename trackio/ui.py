import os
import shutil
from typing import Any

import gradio as gr
import huggingface_hub as hf
import pandas as pd

HfApi = hf.HfApi()

try:
    from trackio.sqlite_storage import SQLiteStorage
    from trackio.utils import (
        COLOR_PALETTE,
        RESERVED_KEYS,
        TRACKIO_LOGO_PATH,
        downsample,
        get_color_mapping,
    )
except:  # noqa: E722
    from sqlite_storage import SQLiteStorage
    from utils import (
        COLOR_PALETTE,
        RESERVED_KEYS,
        TRACKIO_LOGO_PATH,
        downsample,
        get_color_mapping,
    )

css = ""


def get_projects(request: gr.Request):
    dataset_id = os.environ.get("TRACKIO_DATASET_ID")
    projects = SQLiteStorage.get_projects()
    if project := request.query_params.get("project"):
        interactive = False
    else:
        interactive = True
        project = projects[0] if projects else None
    return gr.Dropdown(
        label="Project",
        choices=projects,
        value=project,
        allow_custom_value=True,
        interactive=interactive,
        info=f"&#x21bb; Synced to <a href='https://huggingface.co/datasets/{dataset_id}' target='_blank'>{dataset_id}</a> every 5 min"
        if dataset_id
        else None,
    )


def get_runs(project) -> list[str]:
    if not project:
        return []
    return SQLiteStorage.get_runs(project)


def get_available_metrics(project: str, runs: list[str]) -> list[str]:
    """Get all available metrics across all runs for x-axis selection."""
    if not project or not runs:
        return ["step", "time"]

    all_metrics = set()
    for run in runs:
        metrics = SQLiteStorage.get_metrics(project, run)
        if metrics:
            df = pd.DataFrame(metrics)
            numeric_cols = df.select_dtypes(include="number").columns
            numeric_cols = [c for c in numeric_cols if c not in RESERVED_KEYS]
            all_metrics.update(numeric_cols)

    # Always include step and time as options
    all_metrics.add("step")
    all_metrics.add("time")

    # Sort metrics by prefix
    sorted_metrics = sort_metrics_by_prefix(list(all_metrics))

    # Put step and time at the beginning
    result = ["step", "time"]
    for metric in sorted_metrics:
        if metric not in result:
            result.append(metric)

    return result


def load_run_data(
    project: str | None,
    run: str | None,
    smoothing_method: str,
    smoothing_value: float,
    x_axis: str,
) -> pd.DataFrame | None:
    if not project or not run:
        return None
    metrics = SQLiteStorage.get_metrics(project, run)
    if not metrics:
        return None
    df = pd.DataFrame(metrics)

    if "step" not in df.columns:
        df["step"] = range(len(df))

    if x_axis == "time" and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        first_timestamp = df["timestamp"].min()
        df["time"] = (df["timestamp"] - first_timestamp).dt.total_seconds()
        x_column = "time"
    elif x_axis == "step":
        x_column = "step"
    else:
        x_column = x_axis

    if smoothing_method != "None":
        numeric_cols = df.select_dtypes(include="number").columns
        numeric_cols = [c for c in numeric_cols if c not in RESERVED_KEYS]

        df_original = df.copy()
        df_original["run"] = f"{run}_original"
        df_original["data_type"] = "original"

        df_smoothed = df.copy()
        if smoothing_method == "SMA":
            window_size = max(1, int(smoothing_value))
            df_smoothed[numeric_cols] = (
                df_smoothed[numeric_cols]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
        elif smoothing_method == "EMA":
            alpha = max(0.0, min(1.0, float(smoothing_value)))
            df_smoothed[numeric_cols] = (
                df_smoothed[numeric_cols].ewm(alpha=alpha, adjust=False).mean()
            )
        else:
            df_smoothed = df_original.copy()
        df_smoothed["run"] = f"{run}_smoothed"
        df_smoothed["data_type"] = "smoothed"

        combined_df = pd.concat([df_original, df_smoothed], ignore_index=True)
        combined_df["x_axis"] = x_column
        return combined_df
    df["run"] = run
    df["data_type"] = "original"
    df["x_axis"] = x_column
    return df


def update_runs(project, filter_text, visible_runs, user_interacted_with_runs=False):
    if project is None:
        runs = []
        num_runs = 0
    else:
        runs = get_runs(project)
        num_runs = len(runs)
        if filter_text:
            runs = [r for r in runs if filter_text in r]

    if not user_interacted_with_runs:
        visible_runs = runs
    else:
        visible_runs = [r for r in visible_runs if r in runs]

    with gr.Column() as run_list:
        for i, run in enumerate(runs):
            current_color = SQLiteStorage.get_run_color(project, run)
            if current_color is None:
                current_color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
                SQLiteStorage.set_run_color(project, run, current_color)

            is_visible = run in visible_runs
            cb = gr.Checkbox(value=is_visible, label=run, key=f"vis-{run}")
            cb.change(
                lambda checked, lst, run=run: update_visible_runs(checked, run, lst),
                inputs=[cb, visible_runs],
                outputs=visible_runs,
            )
            cb.change(lambda: True, inputs=None, outputs=user_interacted_with_run_cb)

            dd = gr.Dropdown(
                label="Color",
                choices=COLOR_PALETTE,
                value=current_color,
                interactive=True,
                key=f"color-{run}",
            )
            dd.change(
                lambda color, project=project, run=run: SQLiteStorage.set_run_color(
                    project, run, color
                ),
                inputs=dd,
                outputs=None,
            )

    return run_list, gr.Textbox(label=f"Runs ({num_runs})"), visible_runs


def filter_runs(project, filter_text, visible_runs):
    return update_runs(
        project,
        filter_text,
        visible_runs,
        user_interacted_with_runs=True,
    )


def update_x_axis_choices(project, runs):
    """Update x-axis dropdown choices based on available metrics."""
    available_metrics = get_available_metrics(project, runs)
    return gr.Dropdown(
        label="X-axis",
        choices=available_metrics,
        value="step",
    )


def toggle_timer(cb_value):
    if cb_value:
        return gr.Timer(active=True)
    else:
        return gr.Timer(active=False)


def toggle_smoothing_slider(method: str):
    if method == "SMA":
        return gr.update(
            interactive=True,
            label="SMA window",
            minimum=1,
            maximum=100,
            step=1,
            value=10,
        )
    if method == "EMA":
        return gr.update(
            interactive=True,
            label="EMA alpha",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.5,
        )
    return gr.update(interactive=False, label="Smoothing parameter")


def update_visible_runs(checked: bool, run: str, current: list[str]):
    updated = current.copy()
    if checked and run not in updated:
        updated.append(run)
    elif not checked and run in updated:
        updated.remove(run)
    return updated


def check_auth(hf_token: str | None) -> None:
    if os.getenv("SYSTEM") == "spaces":  # if we are running in Spaces
        # check auth token passed in
        if hf_token is None:
            raise PermissionError(
                "Expected a HF_TOKEN to be provided when logging to a Space"
            )
        who = HfApi.whoami(hf_token)
        access_token = who["auth"]["accessToken"]
        owner_name = os.getenv("SPACE_AUTHOR_NAME")
        repo_name = os.getenv("SPACE_REPO_NAME")
        # make sure the token user is either the author of the space,
        # or is a member of an org that is the author.
        orgs = [o["name"] for o in who["orgs"]]
        if owner_name != who["name"] and owner_name not in orgs:
            raise PermissionError(
                "Expected the provided hf_token to be the user owner of the space, or be a member of the org owner of the space"
            )
        # reject fine-grained tokens without specific repo access
        if access_token["role"] == "fineGrained":
            matched = False
            for item in access_token["fineGrained"]["scoped"]:
                if (
                    item["entity"]["type"] == "space"
                    and item["entity"]["name"] == f"{owner_name}/{repo_name}"
                    and "repo.write" in item["permissions"]
                ):
                    matched = True
                    break
                if (
                    item["entity"]["type"] == "user"
                    and item["entity"]["name"] == owner_name
                    and "repo.write" in item["permissions"]
                ):
                    matched = True
                    break
            if not matched:
                raise PermissionError(
                    "Expected the provided hf_token with fine grained permissions to provide write access to the space"
                )
        # reject read-only tokens
        elif access_token["role"] != "write":
            raise PermissionError(
                "Expected the provided hf_token to provide write permissions"
            )


def upload_db_to_space(
    project: str, uploaded_db: gr.FileData, hf_token: str | None
) -> None:
    check_auth(hf_token)
    db_project_path = SQLiteStorage.get_project_db_path(project)
    if os.path.exists(db_project_path):
        raise gr.Error(
            f"Trackio database file already exists for project {project}, cannot overwrite."
        )
    os.makedirs(os.path.dirname(db_project_path), exist_ok=True)
    shutil.copy(uploaded_db["path"], db_project_path)


def log(
    project: str,
    run: str,
    metrics: dict[str, Any],
    hf_token: str | None,
) -> None:
    check_auth(hf_token)
    SQLiteStorage.log(project=project, run=run, metrics=metrics)


def sort_metrics_by_prefix(metrics: list[str]) -> list[str]:
    """
    Sort metrics by grouping prefixes together.
    Metrics without prefixes come first, then grouped by prefix.

    Example:
    Input: ["train/loss", "loss", "train/acc", "val/loss"]
    Output: ["loss", "train/acc", "train/loss", "val/loss"]
    """
    no_prefix = []
    with_prefix = []

    for metric in metrics:
        if "/" in metric:
            with_prefix.append(metric)
        else:
            no_prefix.append(metric)

    no_prefix.sort()

    prefix_groups = {}
    for metric in with_prefix:
        prefix = metric.split("/")[0]
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(metric)

    sorted_with_prefix = []
    for prefix in sorted(prefix_groups.keys()):
        sorted_with_prefix.extend(sorted(prefix_groups[prefix]))

    return no_prefix + sorted_with_prefix


def configure(request: gr.Request):
    sidebar_param = request.query_params.get("sidebar")
    match sidebar_param:
        case "collapsed":
            sidebar = gr.Sidebar(open=False, visible=True)
        case "hidden":
            sidebar = gr.Sidebar(open=False, visible=False)
        case _:
            sidebar = gr.Sidebar(open=True, visible=True)

    if metrics := request.query_params.get("metrics"):
        return metrics.split(","), sidebar
    else:
        return [], sidebar


with gr.Blocks(theme="citrus", title="Trackio Dashboard", css=css) as demo:
    with gr.Sidebar(open=False) as sidebar:
        gr.Markdown(
            f"<div style='display: flex; align-items: center; gap: 8px;'><img src='/gradio_api/file={TRACKIO_LOGO_PATH}' width='32' height='32'><span style='font-size: 2em; font-weight: bold;'>Trackio</span></div>"
        )
        project_dd = gr.Dropdown(label="Project", allow_custom_value=True)
        run_tb = gr.Textbox(label="Runs", placeholder="Type to filter...")
        run_list = gr.Column()
        gr.HTML("<hr>")
        realtime_cb = gr.Checkbox(label="Refresh metrics realtime", value=True)
        smoothing_method_dd = gr.Dropdown(
            label="Smoothing",
            choices=["None", "SMA", "EMA"],
            value="None",
        )
        smoothing_slider = gr.Slider(
            label="Smoothing parameter",
            minimum=1,
            maximum=100,
            step=1,
            value=10,
            interactive=False,
        )
        x_axis_dd = gr.Dropdown(
            label="X-axis",
            choices=["step", "time"],
            value="step",
        )

    timer = gr.Timer(value=1)
    metrics_subset = gr.State([])
    user_interacted_with_run_cb = gr.State(False)
    visible_runs = gr.State([])

    gr.on([demo.load], fn=configure, outputs=[metrics_subset, sidebar])
    gr.on(
        [demo.load],
        fn=get_projects,
        outputs=project_dd,
        show_progress="hidden",
    )
    gr.on(
        [timer.tick],
        fn=update_runs,
        inputs=[project_dd, run_tb, visible_runs, user_interacted_with_run_cb],
        outputs=[run_list, run_tb, visible_runs],
        show_progress="hidden",
    )
    gr.on(
        [demo.load, project_dd.change],
        fn=update_runs,
        inputs=[project_dd, run_tb, visible_runs, user_interacted_with_run_cb],
        outputs=[run_list, run_tb, visible_runs],
        show_progress="hidden",
    )
    gr.on(
        [demo.load, project_dd.change, visible_runs.change],
        fn=update_x_axis_choices,
        inputs=[project_dd, visible_runs],
        outputs=x_axis_dd,
        show_progress="hidden",
    )

    realtime_cb.change(
        fn=toggle_timer,
        inputs=realtime_cb,
        outputs=timer,
        api_name="toggle_timer",
    )
    smoothing_method_dd.change(
        fn=toggle_smoothing_slider,
        inputs=smoothing_method_dd,
        outputs=smoothing_slider,
    )
    run_tb.input(
        fn=filter_runs,
        inputs=[project_dd, run_tb, visible_runs],
        outputs=[run_list, visible_runs],
    )

    gr.api(
        fn=upload_db_to_space,
        api_name="upload_db_to_space",
    )
    gr.api(
        fn=log,
        api_name="log",
    )

    x_lim = gr.State(None)
    last_steps = gr.State({})

    def update_x_lim(select_data: gr.SelectData):
        return select_data.index

    def update_last_steps(project, runs):
        """Update the last step from all runs to detect when new data is available."""
        if not project or not runs:
            return {}

        return SQLiteStorage.get_max_steps_for_runs(project, runs)

    timer.tick(
        fn=update_last_steps,
        inputs=[project_dd, visible_runs],
        outputs=last_steps,
        show_progress="hidden",
    )

    @gr.render(
        triggers=[
            demo.load,
            visible_runs.change,
            last_steps.change,
            smoothing_method_dd.change,
            smoothing_slider.change,
            x_lim.change,
            x_axis_dd.change,
        ],
        inputs=[
            project_dd,
            visible_runs,
            smoothing_method_dd,
            smoothing_slider,
            metrics_subset,
            x_lim,
            x_axis_dd,
        ],
        show_progress="hidden",
    )
    def update_dashboard(
        project,
        runs,
        smoothing_method,
        smoothing_value,
        metrics_subset,
        x_lim_value,
        x_axis,
    ):
        dfs = []
        original_runs = runs.copy()

        for run in runs:
            df = load_run_data(project, run, smoothing_method, smoothing_value, x_axis)
            if df is not None:
                dfs.append(df)

        if dfs:
            master_df = pd.concat(dfs, ignore_index=True)
        else:
            master_df = pd.DataFrame()

        if master_df.empty:
            return

        x_column = "step"
        if dfs and not dfs[0].empty and "x_axis" in dfs[0].columns:
            x_column = dfs[0]["x_axis"].iloc[0]

        numeric_cols = master_df.select_dtypes(include="number").columns
        numeric_cols = [c for c in numeric_cols if c not in RESERVED_KEYS]
        if metrics_subset:
            numeric_cols = [c for c in numeric_cols if c in metrics_subset]

        numeric_cols = sort_metrics_by_prefix(list(numeric_cols))
        color_map = get_color_mapping(
            project, original_runs, smoothing_method != "None"
        )

        with gr.Row(key="row"):
            for metric_idx, metric_name in enumerate(numeric_cols):
                metric_df = master_df.dropna(subset=[metric_name])
                color = "run" if "run" in metric_df.columns else None
                if not metric_df.empty:
                    plot = gr.LinePlot(
                        downsample(
                            metric_df, x_column, metric_name, color, x_lim_value
                        ),
                        x=x_column,
                        y=metric_name,
                        color=color,
                        color_map=color_map,
                        title=metric_name,
                        key=f"plot-{metric_idx}",
                        preserved_by_key=None,
                        x_lim=x_lim_value,
                        show_fullscreen_button=True,
                        min_width=400,
                    )
                plot.select(update_x_lim, outputs=x_lim, key=f"select-{metric_idx}")
                plot.double_click(
                    lambda: None, outputs=x_lim, key=f"double-{metric_idx}"
                )

    gr.HTML(
        """
        <a class='weatherwidget-io' href='https://forecast7.com/ru/60d0830d12/saint-petersburg/' data-label_1='Санкт-Петербург' data-label_2='ПОГОДА' data-theme='original'>Санкт-Петербург ПОГОДА</a>
        <script>
        !function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0];if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src='https://weatherwidget.io/js/widget.min.js';fjs.parentNode.insertBefore(js,fjs);}}(document,'script','weatherwidget-io-js');
        </script>
        """
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=[TRACKIO_LOGO_PATH], show_api=False, show_error=True)
