"""有图形桌面时验证真实 Tk 窗口；仅使用 DEMO，禁止创建 SDK 连接。"""

import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from test_runtime import config
from bpx_dreamwaq.sim2real.monitor_window import MonitorWindow


class MonitorGuiTest(unittest.TestCase):
    def setUp(self):
        try:
            import tkinter as tk
        except ImportError as exc:
            self.skipTest(f"需要 Tkinter：{exc}")
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"需要可用的图形桌面：{exc}")
        self.root.withdraw()
        self.callbacks = []
        self.root.report_callback_exception = lambda *error: self.callbacks.append(error)

    def run_window(self, *, no_policy):
        captured = {}
        with patch("bpx_dreamwaq.sim2real.sdk_robot.SDKSource.__init__",
                   side_effect=AssertionError("DEMO 不应创建 SDK")) as sdk:
            window = MonitorWindow(self.root, config(), demo=True, no_policy=no_policy)

            def finish():
                try:
                    captured["report"] = window.preflight_report
                    captured["snapshot"] = window.snapshot
                    captured["model"] = window.check_table.item("model", "values")
                    captured["tabs"] = [window.notebook.tab(tab, "text") for tab in window.notebook.tabs()]
                    captured["rows"] = len(window.table.get_children())
                    directory = os.environ.get("BPX_GUI_CAPTURE_DIR")
                    if directory and not no_policy:
                        path = Path(directory)
                        path.mkdir(parents=True, exist_ok=True)
                        window.font_size.set(20)
                        window.change_font_size()
                        self.root.geometry("2400x1500+0+0")
                        self.root.deiconify()
                        self.root.update_idletasks()
                        for index, tab in enumerate(window.notebook.tabs()):
                            window.notebook.select(tab)
                            self.root.update_idletasks()
                            subprocess.run(["xwd", "-silent", "-id", str(self.root.winfo_id()),
                                            "-out", str(path / f"monitor-{index}.xwd")],
                                           check=True, timeout=5, capture_output=True)
                finally:
                    window.close()

            self.root.after(1800, finish)
            try:
                self.root.mainloop()
            finally:
                # 断言失败也应等待仅有的两个本地工作线程退出。
                for worker in (window.worker, window.preflight_worker):
                    if worker:
                        if hasattr(worker, "stop_event"):
                            worker.stop_event.set()
                        worker.join(timeout=3)
                        self.assertFalse(worker.is_alive())
            sdk.assert_not_called()
        self.assertEqual(self.callbacks, [])
        self.assertEqual(captured["tabs"], ["启动前检查", "实时反馈与姿态", "官方定义 / 当前配置"])
        self.assertEqual(captured["rows"], 12)
        self.assertEqual(captured["snapshot"]["source"], "DEMO · 模拟反馈")
        return captured

    def test_real_widgets_without_policy_load(self):
        with patch("bpx_dreamwaq.sim2real.preflight.OnnxPolicy", side_effect=AssertionError("no-policy")):
            captured = self.run_window(no_policy=True)
        self.assertEqual(captured["model"][1], "已关闭")
        self.assertFalse(captured["snapshot"]["policies"])

    def test_real_widgets_with_actual_onnx_preview(self):
        if not config().model_path().is_file() or importlib.util.find_spec("onnxruntime") is None:
            self.root.destroy()
            self.skipTest("需要本地 ONNX 模型及推理依赖")
        captured = self.run_window(no_policy=False)
        self.assertTrue(captured["report"]["model_checked"])
        self.assertEqual(captured["model"][1], "通过")
        self.assertEqual(set(captured["snapshot"]["policies"]), {"policy"})


if __name__ == "__main__":
    unittest.main()
