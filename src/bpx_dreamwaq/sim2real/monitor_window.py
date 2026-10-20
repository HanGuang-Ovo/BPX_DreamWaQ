"""只读监控窗口：Tk 布局、参数表、反馈刷新、骨架与曲线绘制。

SDK 采样和策略预览由 MonitorWorker 执行；窗口不发送运动命令。
"""

from dataclasses import replace
import queue
import time

import numpy as np

from bpx_dreamwaq.runtime.joints import SDK_JOINT_NAMES
from .monitor_worker import MonitorWorker
from .monitor_geometry import leg_points


class MonitorWindow:
    """通过 Tk 定时刷新有界消息队列，绘制反馈；连接生命周期交给工作线程。"""

    BG = "#101823"
    PANEL = "#172332"
    TEXT = "#e6edf5"
    MUTED = "#99adc2"
    CYAN = "#51d5c5"
    ORANGE = "#ffbd69"

    def __init__(
        self,
        root,
        config,
        *,
        demo=False,
        recording=None,
        no_policy=False,
        output=None,
        gamepad_device=None,
        gamepad_index=None,
    ):
        import tkinter as tk
        from tkinter import font as tkfont, ttk

        self.tk, self.ttk = tk, ttk
        self.root, self.config = root, config
        # 固定为常见的 96 DPI 字体比例，避免多显示器虚拟桌面的异常
        # DPI 让同一几何尺寸中的数字被裁切（Conda Tk 可能使用位图字体）。
        root.tk.call("tk", "scaling", 4 / 3)
        families = set(tkfont.families(root))
        family = (
            "Noto Sans CJK SC" if "Noto Sans CJK SC" in families else "song ti" if "song ti" in families else "sans"
        )
        initial_size = int(config.runtime.get("monitor", {}).get("font_size", 32))
        if not 12 <= initial_size <= 48:
            raise ValueError("monitor.font_size 要求 12～48")
        self.font_size = tk.IntVar(value=initial_size)
        self.body_font = tkfont.Font(root, family=family, size=initial_size)
        self.small_font = tkfont.Font(root, family=family, size=initial_size - 2)
        self.table_font = tkfont.Font(root, family="courier 10 pitch", size=initial_size, weight="bold")
        self.title_font = tkfont.Font(root, family=family, size=initial_size + 12, weight="bold")
        self.parameter_tables = []
        self.demo, self.recording, self.no_policy, self.output = demo, recording, no_policy, output
        self.gamepad_device, self.gamepad_index = gamepad_device, gamepad_index
        self.messages = queue.Queue(maxsize=8)
        self.worker = None
        self.snapshot = None
        self.closing = False
        self.active = False
        self.history = []
        self.mode = tk.StringVar(value="")
        self.policy_selection = tk.StringVar(value="统一策略（只读状态机）：等待反馈和电脑手柄")
        self.robot_state = tk.StringVar(value="机器人反馈：运动状态 —    步态 —    控制来源 —")
        self.remote_input = tk.StringVar(value="电脑遥控器输入：vx — m/s    vy — m/s    wz — rad/s")
        self.show_target = tk.BooleanVar(value=True)
        self.ip = tk.StringVar(value=config.network["robot_ip"])
        self.status = tk.StringVar(value="未连接 · 插好网线并完成上电后，点击「连接只读反馈」")
        self.identity = tk.StringVar(value="型号 —   SN —   固件 —   电量 —")
        self.freshness = tk.StringVar(value="joint — ms     IMU — ms")
        self.policy_status = tk.StringVar(value="模型尚未加载")
        self.attitude_text = tk.StringVar(value="等待有效四元数 xyzw…")
        self.warning = tk.StringVar(value="读取电脑手柄并自动选择预览模型；不发送速度、关节或起身命令。")
        root.title("BPX · SDK 反馈与策略对齐")
        width = min(3400, root.winfo_screenwidth() - 120)
        height = min(2000, root.winfo_screenheight() - 120)
        self.window_width = max(1180, width)
        root.geometry(f"{max(1180, width)}x{max(780, height)}")
        root.minsize(1180, 780)
        root.configure(bg=self.BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        self.style = style
        style.configure(".", font=self.body_font, background=self.PANEL, foreground=self.TEXT)
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
        style.configure("TButton", padding=(12, 7))
        style.map("TButton", background=[("active", "#294156")])
        style.configure("TEntry", fieldbackground="#223348", foreground=self.TEXT, font=self.body_font)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#223348")],
            foreground=[("readonly", self.TEXT)],
            selectforeground=[("readonly", self.TEXT)],
        )
        style.configure(
            "Treeview",
            background=self.PANEL,
            fieldbackground=self.PANEL,
            rowheight=42,
            font=self.table_font,
            foreground=self.TEXT,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading", font=self.body_font, background="#26364a", foreground=self.MUTED, padding=(6, 9)
        )
        style.configure("Parameters.Treeview", font=self.body_font)
        style.map("Treeview", background=[("selected", "#264d60")])
        style.configure("TNotebook", background=self.BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(22, 8), background=self.PANEL)
        style.map("TNotebook.Tab", background=[("selected", "#284052")])
        self._build()
        self.change_font_size()
        root.bind("<Configure>", self.resize_window)
        root.after(50, self.pump)
        if demo or recording:
            root.after(150, self.start)

    def label(self, parent, text=None, variable=None, **kwargs):
        options = {"text": text} if variable is None else {"textvariable": variable}
        return self.ttk.Label(parent, **options, **kwargs)

    def _build(self):
        ttk, tk = self.ttk, self.tk
        header = ttk.Frame(self.root, padding=(22, 16))
        header.pack(fill="x")
        self.label(header, "BPX / 参数对齐", font=self.title_font).pack(side="left")
        self.label(header, "READ ONLY · 不发送运动或标零命令", foreground=self.CYAN).pack(side="right")
        toolbar = ttk.Frame(self.root, padding=(22, 0, 22, 12))
        toolbar.pack(fill="x")
        self.label(toolbar, "机器人 IP").pack(side="left", padx=(0, 8))
        self.ip_entry = ttk.Entry(toolbar, textvariable=self.ip, width=17, font=self.body_font)
        self.ip_entry.pack(side="left")
        self.connect_button = ttk.Button(
            toolbar,
            text="播放模拟反馈" if self.demo else "播放录制" if self.recording else "连接只读反馈",
            command=self.start,
        )
        self.connect_button.pack(side="left", padx=10)
        self.disconnect_button = ttk.Button(toolbar, text="断开", command=self.disconnect, state="disabled")
        self.disconnect_button.pack(side="left")
        self.label(toolbar, variable=self.freshness, foreground=self.MUTED).pack(side="right")
        fontsize_control = ttk.Frame(toolbar)
        fontsize_control.pack(side="right", padx=24)
        self.label(fontsize_control, "字号").pack(side="left", padx=(0, 8))
        fontsize_box = ttk.Combobox(
            fontsize_control,
            textvariable=self.font_size,
            values=(12, 16, 20, 24, 28, 32, 36, 40, 48),
            width=3,
            state="readonly",
            font=self.body_font,
        )
        fontsize_box.pack(side="left")
        fontsize_box.bind("<<ComboboxSelected>>", self.change_font_size)
        line = ttk.Frame(self.root, padding=(22, 0, 22, 10))
        line.pack(fill="x")
        self.label(line, variable=self.status, foreground=self.CYAN).pack(anchor="w")
        self.label(line, variable=self.identity, foreground=self.MUTED).pack(anchor="w", pady=(4, 0))
        commands = ttk.Frame(self.root, padding=(22, 0, 22, 12))
        commands.pack(fill="x")
        policy_controls = ttk.Frame(commands)
        policy_controls.pack(fill="x", pady=(0, 8))
        self.selection_label = self.label(
            policy_controls,
            variable=self.policy_selection,
            foreground=self.CYAN,
            wraplength=max(400, self.window_width - 900),
        )
        self.selection_label.pack(side="left", fill="x", expand=True)
        ttk.Checkbutton(policy_controls, text="显示策略目标骨架", variable=self.show_target, command=self.redraw).pack(
            side="right"
        )
        self.control_labels = [
            self.label(commands, variable=self.robot_state, foreground=self.MUTED, wraplength=self.window_width - 80),
            self.label(commands, variable=self.remote_input, foreground=self.TEXT, wraplength=self.window_width - 80),
        ]
        for label in self.control_labels:
            label.pack(anchor="w", pady=(0, 6))
        footer = ttk.Frame(self.root, padding=(22, 10))
        footer.pack(side="bottom", fill="x")
        self.footer_labels = [
            self.label(footer, variable=self.policy_status, foreground=self.MUTED, wraplength=self.window_width - 80),
            self.label(footer, variable=self.warning, foreground=self.ORANGE, wraplength=self.window_width - 80),
        ]
        self.footer_labels[0].pack(anchor="w")
        self.footer_labels[1].pack(anchor="w", pady=(3, 0))
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=22)
        overview, parameters = ttk.Frame(notebook), ttk.Frame(notebook)
        notebook.add(overview, text="实时反馈与姿态")
        notebook.add(parameters, text="官方定义 / 当前配置")
        overview.columnconfigure(0, weight=7)
        overview.columnconfigure(1, weight=4)
        overview.rowconfigure(0, weight=1)
        left = ttk.Frame(overview, padding=(0, 12, 12, 0))
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(overview, padding=(0, 12, 0, 0))
        right.grid(row=0, column=1, sticky="nsew")
        self.label(left, "12 个关节 · 策略顺序    角度 rad / 速度 rad/s", foreground=self.MUTED).pack(
            anchor="w", pady=(0, 8)
        )
        columns = [
            ("joint", "关节 / P,S", 130),
            ("sdk", "SDK q", 68),
            ("official", "官方 q", 68),
            ("model", "当前 q", 68),
            ("dq", "当前 dq", 70),
            ("tau", "SDK 力矩*", 83),
            ("action", "action", 70),
            ("target", "策略 q", 70),
            ("target_sdk", "SDK 目标", 75),
            ("delta", "目标−反馈", 85),
        ]
        frame = ttk.Frame(left)
        frame.pack(fill="x")
        self.table = ttk.Treeview(
            frame, columns=[c[0] for c in columns], show="headings", height=12, selectmode="browse"
        )
        for key, title, width in columns:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=width, anchor="center", stretch=True)
        self.table.tag_configure("override", foreground=self.ORANGE)
        self.table.tag_configure("stale", foreground="#6f8093")
        self.table.tag_configure("outside", foreground="#ff8290")
        for i, j in enumerate(self.config.joints):
            self.table.insert(
                "", "end", iid=str(i), values=(self.joint_label(j.name) + f" / {i},{j.sdk_index}",) + ("—",) * 9
            )
        table_vertical = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        table_horizontal = ttk.Scrollbar(frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=table_vertical.set, xscrollcommand=table_horizontal.set)
        frame.columnconfigure(0, weight=1)
        self.table.grid(row=0, column=0, sticky="ew")
        table_vertical.grid(row=0, column=1, sticky="ns")
        table_horizontal.grid(row=1, column=0, sticky="ew")
        left.bind("<Configure>", self.resize_feedback_table)
        self.table.selection_set("0")
        self.table.bind("<<TreeviewSelect>>", self.selected_joint)
        self.label(
            left,
            "官方 q：官方顺序直接读取；当前 q：应用配置 sign/offset。*力矩单位以固件说明为准。",
            foreground=self.MUTED,
            font=self.small_font,
            wraplength=1200,
        ).pack(anchor="w", pady=8)
        self.label(left, "选中关节 · 青色反馈 / 橙色策略目标（未发送）", foreground=self.MUTED).pack(anchor="w")
        self.graph = tk.Canvas(left, bg=self.PANEL, height=110, highlightthickness=0)
        self.graph.pack(fill="both", expand=True, pady=(8, 0))
        self.graph.bind("<Configure>", lambda e: self.draw_graph())
        self.label(right, "姿态与关节骨架 · ZYX 四元数解算", foreground=self.MUTED).pack(anchor="w")
        self.canvas = tk.Canvas(right, bg=self.PANEL, height=370, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(8, 8))
        self.canvas.bind("<Configure>", lambda e: self.draw_robot())
        self.label(
            right,
            "灰：官方原始解释   青：当前配置   橙：策略目标",
            foreground=self.MUTED,
            font=self.small_font,
            wraplength=650,
        ).pack(anchor="w")
        self.label(right, variable=self.attitude_text, font=self.small_font, justify="left", wraplength=1000).pack(
            anchor="w", pady=8
        )
        self.label(
            right,
            "示意骨架用于核对方向；不测量地面高度或足端接触。",
            foreground=self.MUTED,
            font=self.small_font,
            wraplength=650,
        ).pack(anchor="w")
        self._parameters(parameters)

    def _parameters(self, parent):
        ttk = self.ttk
        self.label(parent, "官方关节示例参数用于对照，不是你的策略默认值或真机验收结果。", foreground=self.ORANGE).pack(
            anchor="w", padx=12, pady=12
        )
        comparison_frame = ttk.Frame(parent)
        comparison_frame.pack(fill="x", padx=12)
        comparison_frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(
            comparison_frame,
            columns=("item", "official", "configured"),
            show="headings",
            height=7,
            style="Parameters.Treeview",
        )
        for key, title, width in (
            ("item", "项目", 180),
            ("official", "官方示例 / 定义", 470),
            ("configured", "当前工程", 470),
        ):
            tree.heading(key, text=title)
            tree.column(key, width=width, anchor="w")
        rows = [
            (
                "关节示例 kp / kd",
                "100 / 2",
                (
                    f"kp={min(self.config.control['kp']):g}…{max(self.config.control['kp']):g},"
                    f" kd={min(self.config.control['kd']):g}…{max(self.config.control['kd']):g}（逐关节见下表）"
                ),
            ),
            (
                "命令发送 / 策略频率",
                "关节示例 sleep(0.01)，约 100 Hz",
                f"{self.config.control['send_rate_hz']} / {self.config.control['policy_rate_hz']} Hz",
            ),
            ("示例目标 / 策略默认姿态", "关节示例每腿 [0, 0.8, -1.8]", "训练默认每腿 [0, 0.7, -1.4] rad"),
            (
                "策略观测 / action scale",
                "SDK 无策略观测或 action scale",
                (
                    f"{self.config.policy['observation_dimension']} 维 / H={self.config.policy['history_length']} /"
                    f" scale={self.config.policy['action_scale']}"
                ),
            ),
            ("零位准备", "Motion 示例按指定趴姿调用 setZeroPositionsFlag", "本窗口不标零；按厂商流程初始化"),
            ("IMU 四元数", "xyzw / 机身姿态；首次接入核对坐标", self.config.calibration["imu"]["quaternion_frame"]),
            ("模型限位", "官方 BPX URDF 范围，非固件硬限矩", "见下表；有覆盖项时使用覆盖范围"),
        ]
        for row in rows:
            tree.insert("", "end", values=row)
        tree_scroll = ttk.Scrollbar(comparison_frame, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=tree_scroll.set)
        tree.grid(row=0, column=0, sticky="ew")
        tree_scroll.grid(row=1, column=0, sticky="ew")
        self.parameter_tables.append(tree)
        mapping_frame = ttk.Frame(parent)
        mapping_frame.pack(fill="both", expand=True, padx=12, pady=12)
        mapping = ttk.Treeview(
            mapping_frame,
            columns=("name", "sdk", "current", "sign", "offset", "limits", "kp", "kd"),
            show="headings",
            height=7,
        )
        for key, title, width in (
            ("name", "关节", 220),
            ("sdk", "官方 SDK 索引", 160),
            ("current", "配置 SDK 索引", 160),
            ("sign", "sign", 80),
            ("offset", "offset [rad]", 130),
            ("limits", "当前模型范围 [rad]", 220),
            ("kp", "配置 kp", 90),
            ("kd", "配置 kd", 90),
        ):
            mapping.heading(key, text=title)
            mapping.column(key, width=width, anchor="center")
        mapping.tag_configure("override", foreground=self.ORANGE)
        for i, j in enumerate(self.config.joints):
            index = SDK_JOINT_NAMES.index(j.name)
            mapping.insert(
                "",
                "end",
                values=(
                    j.name,
                    index,
                    j.sdk_index,
                    j.sign,
                    j.offset,
                    f"[{j.minimum}, {j.maximum}]",
                    self.config.control["kp"][i],
                    self.config.control["kd"][i],
                ),
                tags=("override",) if index != j.sdk_index or j.sign != 1 or j.offset != 0 else (),
            )
        scrollbar = ttk.Scrollbar(mapping_frame, orient="vertical", command=mapping.yview)
        horizontal = ttk.Scrollbar(mapping_frame, orient="horizontal", command=mapping.xview)
        mapping.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
        mapping_frame.columnconfigure(0, weight=1)
        mapping_frame.rowconfigure(0, weight=1)
        mapping.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.parameter_tables.append(mapping)
        self.label(
            parent, f"策略模型：{self.config.model_path()}", foreground=self.MUTED, wraplength=1300
        ).pack(anchor="w", padx=12, pady=2)
        self.label(
            parent, f"配置：{self.config.path}\n坐标覆盖：{self.config.calibration_path}", foreground=self.MUTED
        ).pack(anchor="w", padx=12, pady=8)

    def change_font_size(self, event=None):
        size = self.font_size.get()
        self.body_font.configure(size=size)
        self.small_font.configure(size=max(10, size - 2))
        self.table_font.configure(size=size)
        self.title_font.configure(size=size + 12)
        self.row_height = max(self.body_font.metrics("linespace"), self.table_font.metrics("linespace")) + 14
        self.style.configure("Treeview", rowheight=self.row_height)
        self.style.configure("Parameters.Treeview", rowheight=self.row_height)
        for column in self.table["columns"]:
            heading_width = self.body_font.measure(self.table.heading(column, "text")) + 24
            value_width = self.table_font.measure("HL Pitch / 11,11" if column == "joint" else "-000.0000") + 24
            width = max(heading_width, value_width)
            self.table.column(column, width=width, minwidth=width, stretch=False)
        for table in self.parameter_tables:
            font = self.body_font if table.cget("style") == "Parameters.Treeview" else self.table_font
            for i, column in enumerate(table["columns"]):
                texts = [str(table.item(row, "values")[i]) for row in table.get_children()]
                width = (
                    max(
                        [self.body_font.measure(table.heading(column, "text"))] + [font.measure(text) for text in texts]
                    )
                    + 28
                )
                table.column(column, width=width, minwidth=width)
        self.resize_feedback_table()
        self.redraw()

    def resize_feedback_table(self, event=None):
        if not hasattr(self, "row_height"):
            return
        height = event.height if event is not None else self.table.master.master.winfo_height()
        if height <= 1:
            return
        # 给说明与曲线预留空间；小屏幕通过滚动查看全部 12 个关节。
        rows = min(12, max(4, int((height - 260) / self.row_height)))
        if int(self.table.cget("height")) != rows:
            self.table.configure(height=rows)

    def resize_window(self, event):
        if event.widget is self.root:
            for label in self.footer_labels + self.control_labels:
                label.configure(wraplength=max(200, event.width - 60))
            self.selection_label.configure(wraplength=max(400, event.width - 900))

    @staticmethod
    def joint_label(name):
        parts = name.split("_")
        return parts[0].upper() + " " + ("Roll" if "roll" in parts else "Pitch" if "pitch" in parts else "Knee")

    def start(self):
        if self.closing or (self.worker and self.worker.is_alive()):
            return
        ip = self.ip.get().strip()
        if not ip:
            self.status.set("请填写机器人 IPv4 地址")
            return
        import ipaddress

        try:
            ipaddress.IPv4Address(ip)
        except ValueError:
            self.status.set("机器人 IP 格式无效")
            return
        runtime = {**self.config.runtime, "network": {**self.config.network, "robot_ip": ip}}
        config = replace(self.config, runtime=runtime)
        while not self.messages.empty():
            self.messages.get_nowait()
        self.snapshot = None
        self.history = []
        self.active = True
        self.worker = MonitorWorker(
            config,
            self.messages,
            demo=self.demo,
            recording=self.recording,
            no_policy=self.no_policy,
            output=self.output,
            gamepad_device=self.gamepad_device,
            gamepad_index=self.gamepad_index,
        )
        self.mode.set("")
        self.policy_selection.set("统一策略（只读状态机）：等待反馈和电脑手柄")
        self.remote_input.set("电脑遥控器输入：vx — m/s    vy — m/s    wz — rad/s")
        self.connect_button.configure(state="disabled")
        self.ip_entry.configure(state="disabled")
        self.disconnect_button.configure(state="normal")
        self.worker.start()

    def disconnect(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop_event.set()
            self.active = False
            self.status.set("正在断开只读会话…")
            self.disconnect_button.configure(state="disabled")

    def close(self):
        self.closing = True
        self.disconnect()
        self.root.after(50, self._finish_close)

    def _finish_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(50, self._finish_close)
        else:
            self.root.destroy()

    def pump(self):
        refresh_needed = False
        try:
            # 一轮只取有界数量，只绘制最新帧。不能一边慢速绘图一边
            # 无限排空 50 Hz 生产者，否则队列永不为空，Tk 事件被饿死。
            for _ in range(self.messages.maxsize):
                kind, value = self.messages.get_nowait()
                if kind == "snapshot":
                    self.snapshot = value
                    self.status.set(
                        value["source"] + " · 反馈已接收" if value["sample"].connected else "连接断开 · 策略预览暂停"
                    )
                    refresh_needed = True
                elif kind in ("status", "error"):
                    self.status.set(value)
                    if kind == "error":
                        self.active = False
                elif kind == "stopped":
                    self.active = False
                    if self.status.get().startswith("正在断开"):
                        self.status.set(value)
                    self.connect_button.configure(state="normal")
                    self.ip_entry.configure(state="normal")
                    self.disconnect_button.configure(state="disabled")
        except queue.Empty:
            pass
        if refresh_needed:
            self.update_snapshot()
        if self.snapshot:
            elapsed = max(0.0, time.monotonic() - self.snapshot["received_at"])
            sample = self.snapshot["sample"]

            def age(value):
                return "—" if value is None else f"{(value+elapsed)*1000:.0f}"

            self.freshness.set(f"joint {age(sample.joint_age_s)} ms    IMU {age(sample.imu_age_s)} ms")
        if self.snapshot and (
            not self.active or time.monotonic() - self.snapshot["received_at"] > self.config.monitor.get("display_stale_after_s", 0.10)
        ):
            self.refresh_table(stale=True)
            self.policy_status.set("策略预览暂停 · 会话结束或反馈过期；旧值不作为当前输出")
            self.policy_selection.set("统一策略（只读状态机）：暂停 · 会话结束或反馈过期")
            self.remote_input.set("电脑遥控器输入：— · 当前会话未更新")
            self.draw_robot(stale=True)
        if not self.closing:
            self.root.after(50, self.pump)

    def update_snapshot(self):
        snap = self.snapshot
        sample = snap["sample"]
        ident = sample.identity
        self.identity.set(
            f"型号 {ident.get('robot_model') or '—'}   SN {ident.get('serial_number') or '—'}   "
            f"固件 {ident.get('firmware_version') or '—'}   SDK {ident.get('sdk_version') or '—'}   "
            f"电量 {sample.telemetry.get('battery_level', '—')}%"
        )

        def age(v):
            return "—" if v is None else f"{v*1000:.0f}"

        self.freshness.set(f"joint {age(sample.joint_age_s)} ms    IMU {age(sample.imu_age_s)} ms")
        warnings = list(snap["warnings"])
        control = snap["control"]
        previous = self.mode.get()
        self.mode.set(control.get("policy") or "")
        if previous != self.mode.get():
            self.history = []
        state_name = {"stand": "Stand", "walk": "Walk", "stopping": "Walk · STOPPING 减速过渡"}.get(control["mode"])
        self.policy_selection.set("统一策略（只读状态机）：" + (state_name or "暂停 / 等待"))
        self.robot_state.set(
            f"机器人反馈：{control['motion']}    步态 {control['gait']}    控制来源 {control['control_source']}"
        )
        command = control["remote_command"]
        values = [f"{v:+.3f}" for v in command] if command is not None else ["—"] * 3
        self.remote_input.set(
            f"电脑遥控器输入：vx {values[0]} m/s    vy {values[1]} m/s    wz {values[2]} rad/s"
            f"  |  {control['input_source']}"
        )
        mode = self.mode.get()
        result = snap["policies"].get(mode) if self.active else None
        if result:
            self.policy_status.set(
                f"{mode.title()} · {self.config.policy['observation_dimension']} 维观测 → 12 维 action  |  "
                f"q目标 = q默认 + {self.config.policy['action_scale']} × action  |  未混合、未限幅、未发送"
            )
            if result["outside_limits"]:
                warnings.append("策略目标超模型范围：" + ", ".join(result["outside_limits"]))
        else:
            self.policy_status.set(
                (mode.title() + " · " if mode else "") + snap["policy_errors"].get(mode, control["reason"])
            )
        for name, error in snap["policy_errors"].items():
            if name != mode:
                warnings.append(name + ": " + error)
        if not control.get("eligible"):
            warnings.append(control["reason"])
        self.warning.set(
            "；".join(warnings)
            if warnings
            else "只读状态机与部署共用切换逻辑；手柄与策略输出未发送，不代表固件内部正在使用本机模型。"
        )
        att = snap["attitude"]
        if att:
            fmt = lambda xs: "  ".join(f"{x:+.2f}" for x in xs)
            self.attitude_text.set(
                f"四元数 xyzw：\n{'  '.join(f'{v:+.4f}' for v in att['quat_xyzw'])}\n模长 |q|={att['norm']:.4f}\n"
                f"SDK R/P/Y [°]：{fmt(att['sdk_rpy_deg'])}\n当前 R/P/Y [°]：{fmt(att['body_rpy_deg'])}\n"
                f"当前重力方向：{fmt(att['gravity_body'])}\n"
                f"机体角速度 [rad/s]：{fmt(att.get('omega_body', [0,0,0])) if 'omega_body' in att else '—'}"
            )
        else:
            self.attitude_text.set("四元数缺失或无效 · 无姿态示意")
        self.refresh_table()
        index = self.selected_index()
        row = snap["rows"][index]
        self.history.append((time.monotonic(), row["q_model"], result["target_model"][index] if result else None))
        cutoff = time.monotonic() - 10
        self.history = [v for v in self.history if v[0] >= cutoff]
        self.redraw()

    def refresh_table(self, stale=False):
        if not self.snapshot:
            return
        sample = self.snapshot["sample"]
        stale = (
            stale
            or not self.active
            or not sample.connected
            or sample.joint_age_s is None
            or sample.joint_age_s > self.config.monitor.get("display_stale_after_s", 0.10)
        )
        result = None if stale else self.snapshot["policies"].get(self.mode.get())
        fmt = lambda x: "—" if x is None else f"{x:+.4f}"
        for i, row in enumerate(self.snapshot["rows"]):
            action = result["action"][i] if result else None
            target = result["target_model"][i] if result else None
            sdk_target = result["target_sdk"][row["sdk_index"]] if result else None
            delta = target - row["q_model"] if target is not None and row["q_model"] is not None else None
            overrides = row["sdk_index"] != row["official_index"] or row["sign"] != 1 or row["offset"] != 0
            outside = result and row["name"] in result["outside_limits"]
            tag = "stale" if stale else "outside" if outside else "override" if overrides else ""
            self.table.item(
                str(i),
                values=(
                    self.joint_label(row["name"]) + f" / {i},{row['sdk_index']}",
                    *[fmt(row[k]) for k in ("q_sdk", "q_official", "q_model", "dq_model", "torque_sdk")],
                    fmt(action),
                    fmt(target),
                    fmt(sdk_target),
                    fmt(delta),
                ),
                tags=(tag,) if tag else (),
            )

    def selected_index(self):
        selection = self.table.selection()
        return int(selection[0]) if selection else 0

    def selected_joint(self, event=None):
        self.history = []
        self.draw_graph()

    def redraw(self):
        self.draw_robot()
        self.draw_graph()

    def draw_robot(self, stale=False):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 1), max(c.winfo_height(), 1)
        if not self.snapshot or not self.snapshot["attitude"]:
            c.create_text(w / 2, h / 2, text="等待四元数与关节反馈", fill=self.MUTED, font=self.body_font)
            return
        snap = self.snapshot
        sample = snap["sample"]
        stale = (
            stale
            or not self.active
            or not sample.connected
            or sample.imu_age_s is None
            or sample.imu_age_s > self.config.monitor.get("display_stale_after_s", 0.10)
        )
        scale = min(w / 1.25, h / 0.98)
        right = np.array([0.707, -0.707, 0.0])
        up = np.array([0.35, 0.35, 0.869])

        def project(p):
            p = np.asarray(p) + [0, 0, 0.14]
            return (w * 0.5 + np.dot(p, right) * scale, h * 0.47 - np.dot(p, up) * scale)

        for v in np.linspace(-0.5, 0.5, 9):
            for a, b in (([-0.5, v, -0.44], [0.5, v, -0.44]), ([v, -0.5, -0.44], [v, 0.5, -0.44])):
                c.create_line(*project(a), *project(b), fill="#263549")
        origin = project([-0.43, -0.38, -0.44])
        for direction, name, color in (
            ([0.15, 0, 0], "X 前", "#ee7b7b"),
            ([0, 0.15, 0], "Y 左", "#7ee4a6"),
            ([0, 0, 0.15], "Z 上", "#82afff"),
        ):
            end = project(np.array([-0.43, -0.38, -0.44]) + direction)
            c.create_line(*origin, *end, fill=color, width=2, arrow="last")
            c.create_text(end[0], end[1] - 16, text=name, fill=color, font=self.small_font)

        def robot(rotation, q, color, fill=False, dashed=False):
            r = np.asarray(rotation)
            body = np.array([[0.2386, 0.10, 0], [0.2386, -0.10, 0], [-0.2386, -0.10, 0], [-0.2386, 0.10, 0]]) @ r.T
            coords = [x for p in body for x in project(p)]
            c.create_polygon(
                *coords, fill="#244753" if fill else "", outline=color, width=2, dash=(4, 3) if dashed else ()
            )
            c.create_line(*project([0, 0, 0]), *project(r @ [0.30, 0, 0]), fill=color, width=3, arrow="last")
            if q is not None and all(v is not None for v in q):
                for name, leg in zip(("FL", "FR", "HL", "HR"), leg_points(q)):
                    points = leg @ r.T
                    c.create_line(
                        *[x for p in points for x in project(p)], fill=color, width=2, dash=(4, 3) if dashed else ()
                    )
                    if fill:
                        for p in points[1:]:
                            x, y = project(p)
                            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")
                        x, y = project(points[-1])
                        c.create_text(x, y + 18, text=name, fill=color, font=self.small_font)

        att = snap["attitude"]
        robot(att["sdk_rotation"], [row["q_official"] for row in snap["rows"]], "#8190a3", dashed=True)
        robot(
            att["body_rotation"], [row["q_model"] for row in snap["rows"]], "#6d8092" if stale else self.CYAN, fill=True
        )
        result = snap["policies"].get(self.mode.get())
        if result and self.show_target.get() and not stale:
            robot(att["body_rotation"], result["target_model"], self.ORANGE, dashed=True)
        c.create_text(
            12,
            15,
            text="姿态参考示意" + (" · 旧帧" if stale else ""),
            anchor="w",
            fill=self.MUTED,
            font=self.small_font,
        )

    def draw_graph(self):
        c = self.graph
        c.delete("all")
        w, h = max(c.winfo_width(), 1), max(c.winfo_height(), 1)
        if len(self.history) < 2:
            c.create_text(
                w / 2, h / 2, text="选中关节，观察最近 10 秒反馈与预览目标", fill=self.MUTED, font=self.small_font
            )
            return
        valid = [v for _, a, b in self.history for v in (a, b) if v is not None]
        if not valid:
            return
        lo, hi = min(valid) - 0.05, max(valid) + 0.05
        start = max(self.history[-1][0] - 10, self.history[0][0])
        span = max(self.history[-1][0] - start, 0.1)
        margin = max(self.small_font.measure(f"{v:.2f}") for v in (lo, hi)) + 18

        def point(t, v):
            return (margin + (w - margin - 15) * (t - start) / span, h - 25 - (h - 45) * (v - lo) / (hi - lo))

        for v in np.linspace(lo, hi, 4):
            x, y = point(start, v)
            c.create_line(margin - 5, y, w - 10, y, fill="#2a3b50")
            c.create_text(margin - 10, y, text=f"{v:.2f}", anchor="e", fill=self.MUTED, font=self.small_font)
        for col, color in ((1, self.CYAN), (2, self.ORANGE)):
            segments = []
            for row in self.history:
                if row[col] is None:
                    if len(segments) >= 4:
                        c.create_line(*segments, fill=color, width=2)
                    segments = []
                else:
                    segments.extend(point(row[0], row[col]))
            if len(segments) >= 4:
                c.create_line(*segments, fill=color, width=2)
