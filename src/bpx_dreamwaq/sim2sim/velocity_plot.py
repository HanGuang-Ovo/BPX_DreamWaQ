"""实时 command / 机体系实际速度；复用力矩窗口的非阻塞传输和历史管理。"""

import numpy as np

from .torque_plot import TorquePlot, TorqueWindow, envelope_indices


class VelocityWindow(TorqueWindow):
    title = 'Velocity tracking'
    heading = 'VELOCITY TRACKING'
    subtitle = ('Body frame  •  Blue dashed: command (after supervisor)  •  Green solid: actual\n'
                'Error = actual - command  •  RMSE over visible samples')
    ui_font_size = 15
    title_font_size = 20
    export_name = 'bpx_velocity_tracking.csv'

    def make_panels(self, joint_names):
        return [('vx', 'm/s'), ('vy', 'm/s'), ('wz', 'rad/s')]

    def draw(self):
        canvas = self.canvas
        width, height = max(canvas.winfo_width(), 900), max(canvas.winfo_height(), 480)
        canvas.delete('all')
        seconds = float(self.seconds.get())
        rows = self.paused_rows if self.paused_rows is not None else self.history.visible(seconds)
        end = rows[-1][0] if rows else 0.0
        start, end = max(0.0, end - seconds), max(seconds, end)
        times = np.array([row[0] for row in rows])
        values = np.array([row[1] for row in rows])
        for index, (name, unit) in enumerate(self.panels):
            left, right = 6, width - 6
            top, bottom = index * height / 3 + 5, (index + 1) * height / 3 - 5
            canvas.create_rectangle(left, top, right, bottom, fill='#182439', outline='#28364d')
            command = values[:, index] if rows else np.array([])
            actual = values[:, index + 3] if rows else np.array([])
            error = actual - command
            current_command, current_actual = (command[-1], actual[-1]) if rows else (0., 0.)
            rmse = float(np.sqrt(np.mean(error ** 2))) if rows else 0.
            canvas.create_text(left + 12, top + 22, anchor='w', fill='#ecf3ff',
                               font=('DejaVu Sans', 18, 'bold'),
                               text=f'{name} [{unit}]    command {current_command:+.3f}    '
                                    f'actual {current_actual:+.3f}')
            canvas.create_text(left + 12, top + 50, anchor='w', fill='#aabbd2',
                               font=('DejaVu Sans', 15),
                               text=f'error {current_actual - current_command:+.3f}    RMSE {rmse:.3f}')
            x0, x1, y0, y1 = left + 90, right - 50, top + 80, bottom - 34
            extent = max(.2, float(np.max(np.abs(values[:, [index, index + 3]]))) * 1.15) if rows else .2
            for value in (-extent, 0., extent):
                y = (y0 + y1) / 2 - value / extent * (y1 - y0) / 2
                canvas.create_line(x0, y, x1, y, fill='#34455f')
                canvas.create_text(x0 - 8, y, anchor='e', text=f'{value:.2f}',
                                   fill='#96a9c4', font=('DejaVu Sans', 15))
            for fraction in (0., .25, .5, .75, 1.):
                x = x0 + fraction * (x1 - x0)
                canvas.create_text(x, y1 + 20, text=f'{start + fraction * (end - start):.1f}s',
                                   fill='#96a9c4', font=('DejaVu Sans', 15))
            for series, color, dash in ((command, '#5eb2ff', (6, 4)), (actual, '#49dcb1', ())):
                if len(series) < 2:
                    continue
                keep = envelope_indices(series, max(2, int(x1 - x0)))
                xs = x0 + (times[keep] - start) / (end - start) * (x1 - x0)
                ys = (y0 + y1) / 2 - series[keep] / extent * (y1 - y0) / 2
                canvas.create_line(*np.column_stack((xs, ys)).ravel().tolist(), fill=color, dash=dash, width=2)
        if rows:
            timestamp, _, mode, dropped = rows[-1]
            state = 'DISPLAY PAUSED — simulation continues' if self.paused_rows is not None else 'LIVE'
            self.status.set(f'{state}   |   t = {timestamp:.3f}s   |   mode = {mode.upper()}   |   '
                            f'dropped display samples = {dropped}')


class VelocityPlot(TorquePlot):
    window_type = VelocityWindow
    label = '速度'

    def __init__(self, timestep, seconds=10.0):
        super().__init__(('command_vx', 'command_vy', 'command_wz', 'body_vx', 'body_vy', 'body_wz'),
                         0., timestep, seconds)

    def publish(self, timestamp, command, actual, mode):
        super().publish(timestamp, (*command, *actual), mode)
