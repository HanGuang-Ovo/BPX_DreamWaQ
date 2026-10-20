"""Tkinter 实时关节力矩窗口；独立进程绘图，仿真端非阻塞发送物理步数据。"""

from __future__ import annotations

from collections import deque
import csv
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full

import numpy as np


class TorqueHistory:
    """按仿真时间保存最近 30 秒，容量另外受物理采样频率约束。"""

    def __init__(self, joint_names: tuple[str, ...], timestep: float):
        self.joint_names = joint_names
        self.rows = deque(maxlen=math.ceil(30.0 / timestep) + 2)

    def append(self, sample) -> None:
        timestamp, torques, mode, dropped = sample
        if len(torques) != len(self.joint_names) or not np.all(np.isfinite((timestamp, *torques))):
            return
        if self.rows and timestamp < self.rows[-1][0]:
            self.rows.clear()
        self.rows.append((timestamp, tuple(torques), mode, dropped))
        while self.rows and timestamp - self.rows[0][0] > 30.0:
            self.rows.popleft()

    def visible(self, seconds: float):
        if not self.rows:
            return []
        start = self.rows[-1][0] - seconds
        return [row for row in self.rows if row[0] >= start]

    def export(self, path: str | Path, rows) -> None:
        with Path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('time', 'behavior_mode', 'dropped_samples', *self.joint_names))
            for timestamp, torques, mode, dropped in rows:
                writer.writerow((timestamp, mode, dropped, *torques))


def envelope_indices(values: np.ndarray, width: int) -> np.ndarray:
    """按像素桶保留局部最大/最小值，避免抽点时漏掉交替正负力矩尖峰。"""
    if len(values) <= 2 * width:
        return np.arange(len(values))
    selected = [0]
    edges = np.linspace(0, len(values), max(2, width) + 1, dtype=int)
    for start, end in zip(edges, edges[1:]):
        chunk = values[start:end]
        selected.extend(sorted((start + int(np.argmin(chunk)), start + int(np.argmax(chunk)))))
    selected.append(len(values) - 1)
    return np.unique(selected)


class TorqueWindow:
    """Tk widgets 只在绘图进程的主线程访问。"""

    title = "Joint torque monitor"
    heading = "JOINT TORQUES"
    subtitle = "Actuator joint torque [N·m]  •  physics-step samples  •  dashed lines: configured limit"
    export_name = "bpx_joint_torques.csv"
    ui_font_size = 10
    title_font_size = 16

    def __init__(self, root, samples, stop, joint_names, limit, timestep, seconds):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.samples, self.stop = root, samples, stop
        self.history = TorqueHistory(joint_names, timestep)
        self.limit = limit
        self.paused_rows = None
        self.seconds = tk.StringVar(value=f'{seconds:g}')
        self.status = tk.StringVar(value='Waiting for simulation samples...')
        root.title(f'BPX | {self.title}')
        root.geometry('1360x850')
        root.minsize(920, 600)
        root.configure(bg='#101827')
        toolbar = tk.Frame(root, bg='#101827', padx=16, pady=12)
        toolbar.pack(fill='x')
        tk.Label(toolbar, text=f'BPX  /  {self.heading}', fg='#f0f5ff', bg='#101827',
                 font=('DejaVu Sans', self.title_font_size, 'bold')).pack(side='left')
        ui_font = ('DejaVu Sans', self.ui_font_size)
        style = ttk.Style(root)
        style.configure('Monitor.TButton', font=ui_font)
        root.option_add('*TCombobox*Listbox.font', ui_font)
        tk.Label(toolbar, text='Window (s)', fg='#aabbd2', bg='#101827',
                 font=ui_font).pack(side='left', padx=(30, 8))
        selector = ttk.Combobox(toolbar, textvariable=self.seconds, values=(5, 10, 20, 30),
                                width=5, state='readonly', font=ui_font)
        selector.pack(side='left', padx=(0, 16))
        self.pause_button = ttk.Button(toolbar, text='Pause display', command=self.toggle_pause,
                                       style='Monitor.TButton')
        self.pause_button.pack(side='left', padx=4)
        ttk.Button(toolbar, text='Save visible CSV', command=self.save_csv,
                   style='Monitor.TButton').pack(side='left', padx=4)
        tk.Label(root, text=self.subtitle,
                 fg='#aabbd2', bg='#101827', anchor='w', justify='left',
                 font=ui_font, padx=16).pack(fill='x')
        self.canvas = tk.Canvas(root, bg='#101827', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=8, pady=8)
        tk.Label(root, textvariable=self.status, fg='#aabbd2', bg='#101827',
                 anchor='w', font=ui_font, padx=16, pady=8).pack(fill='x')
        self.joint_names = joint_names
        # Display by leg and joint type, independent of policy/MuJoCo index order.
        self.panels = self.make_panels(joint_names)
        self.root.after(0, self.tick)

    def make_panels(self, joint_names):
        return [(leg, kind, joint_names.index(f'{leg}_{kind}_joint'))
                       for kind in ('hip_roll', 'hip_pitch', 'knee')
                       for leg in ('fl', 'fr', 'hl', 'hr')]

    def toggle_pause(self):
        if self.paused_rows is None:
            self.paused_rows = self.history.visible(float(self.seconds.get()))
            self.pause_button.configure(text='Resume display')
        else:
            self.paused_rows = None
            self.pause_button.configure(text='Pause display')

    def save_csv(self):
        from tkinter import filedialog, messagebox
        rows = self.paused_rows if self.paused_rows is not None else self.history.visible(float(self.seconds.get()))
        if not rows:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title=f'Save visible {self.title} samples',
                                          defaultextension='.csv', filetypes=[('CSV', '*.csv')],
                                          initialfile=self.export_name)
        if path:
            try:
                self.history.export(path, rows)
            except OSError as exc:
                messagebox.showerror('Could not save CSV', str(exc), parent=self.root)

    def tick(self):
        if self.stop.is_set():
            self.root.destroy()
            return
        # Bounded drain: UI events remain responsive even with --no-realtime.
        for _ in range(1024):
            try:
                self.history.append(self.samples.get_nowait())
            except Empty:
                break
        self.draw()
        self.root.after(50, self.tick)

    def draw(self):
        canvas = self.canvas
        width, height = max(canvas.winfo_width(), 900), max(canvas.winfo_height(), 480)
        canvas.delete('all')
        seconds = float(self.seconds.get())
        rows = self.paused_rows if self.paused_rows is not None else self.history.visible(seconds)
        end = rows[-1][0] if rows else 0.0
        start, end = max(0.0, end - seconds), max(seconds, end)
        times = np.array([row[0] for row in rows])
        torques = np.array([row[1] for row in rows])
        colors = ('#49dcb1', '#5eb2ff', '#f8c369', '#c39bff')
        for panel, (leg, kind, index) in enumerate(self.panels):
            row, col = divmod(panel, 4)
            left, top = col * width / 4 + 6, row * height / 3 + 5
            right, bottom = (col + 1) * width / 4 - 6, (row + 1) * height / 3 - 5
            canvas.create_rectangle(left, top, right, bottom, fill='#182439', outline='#28364d')
            color = colors[col]
            values = torques[:, index] if rows else np.array([])
            current = values[-1] if len(values) else 0.0
            peak = float(np.max(np.abs(values))) if len(values) else 0.0
            value_color = '#ff7878' if abs(current) >= .9 * self.limit else color
            canvas.create_text(left + 12, top + 16, text=f'{leg.upper()}  {kind.replace("_", " ")}',
                               fill='#ecf3ff', anchor='w', font=('DejaVu Sans', 10, 'bold'))
            canvas.create_text(right - 10, top + 16, text=f'{current:+.2f}', fill=value_color,
                               anchor='e', font=('DejaVu Sans Mono', 12, 'bold'))
            canvas.create_text(left + 12, top + 35, text=f'peak |τ|  {peak:.2f} N·m',
                               fill='#96a9c4', anchor='w', font=('DejaVu Sans', 8))
            x0, x1, y0, y1 = left + 43, right - 12, top + 55, bottom - 26
            extent = max(self.limit * 1.15, peak * 1.1, 1.0)
            for value in (-self.limit, 0, self.limit):
                y = (y0 + y1) / 2 - value / extent * (y1 - y0) / 2
                canvas.create_line(x0, y, x1, y, fill='#54617a' if value else '#34455f',
                                   dash=(4, 4) if value else ())
                canvas.create_text(x0 - 6, y, text=f'{value:g}', anchor='e', fill='#96a9c4',
                                   font=('DejaVu Sans', 8))
            for fraction in (0, .5, 1):
                x = x0 + fraction * (x1 - x0)
                canvas.create_text(x, y1 + 14, text=f'{start + fraction * (end - start):.1f}s',
                                   fill='#96a9c4', font=('DejaVu Sans', 8))
            if len(values) > 1:
                keep = envelope_indices(values, max(2, int(x1 - x0)))
                xs = x0 + (times[keep] - start) / (end - start) * (x1 - x0)
                ys = (y0 + y1) / 2 - values[keep] / extent * (y1 - y0) / 2
                canvas.create_line(*np.column_stack((xs, ys)).ravel().tolist(), fill=color, width=1.5)
        if rows:
            t, _, mode, dropped = rows[-1]
            state = 'DISPLAY PAUSED — simulation continues' if self.paused_rows is not None else 'LIVE'
            self.status.set(f'{state}   |   t = {t:.3f}s   |   mode = {mode.upper()}   |   '
                            f'limit = ±{self.limit:g} N·m   |   dropped display samples = {dropped}')


def _run_window(samples, stop, ready, joint_names, limit, timestep, seconds, window_type):
    root = None
    try:
        import tkinter as tk
        root = tk.Tk()
        window_type(root, samples, stop, joint_names, limit, timestep, seconds)
        ready.put(None)
        root.mainloop()
    except Exception as exc:
        ready.put(f'{type(exc).__name__}: {exc}')
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


class TorquePlot:
    """Context manager. Closing the chart leaves the simulation running."""

    window_type = TorqueWindow
    label = "力矩"

    def __init__(self, joint_names, limit, timestep, seconds=10.0):
        if not math.isfinite(seconds) or not 1 <= seconds <= 30:
            raise ValueError(f'{self.label} window 必须在 1 到 30 秒之间')
        self.joint_names = tuple(joint_names)
        self.limit, self.timestep, self.seconds = limit, timestep, seconds
        self.process = None
        self.dropped = 0

    def __enter__(self):
        # Spawn avoids inheriting MuJoCo/OpenGL/ONNX threads into the GUI process.
        context = mp.get_context('spawn')
        self.samples = context.Queue(maxsize=1024)
        self.ready = context.Queue(maxsize=2)
        self.stop = context.Event()
        self.process = context.Process(target=_run_window, args=(self.samples, self.stop, self.ready,
                                       self.joint_names, self.limit, self.timestep, self.seconds, self.window_type), daemon=True)
        try:
            self.process.start()
            error = self.ready.get(timeout=8)
            if error:
                raise RuntimeError(error)
        except Exception as exc:
            self.close()
            raise RuntimeError(f'无法打开{self.label}窗口，请检查 Tkinter 和桌面显示环境：{exc}') from exc
        return self

    def publish(self, timestamp, torques, mode):
        if self.process is None or not self.process.is_alive():
            return
        sample = (float(timestamp), tuple(float(value) for value in torques), mode, self.dropped)
        try:
            self.samples.put_nowait(sample)
        except Full:
            # Never stall the control loop if rendering cannot keep up.
            self.dropped += 1

    def close(self):
        self.stop.set()
        if self.process is not None and self.process.pid is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        for queue in (self.samples, self.ready):
            queue.cancel_join_thread()
            queue.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
