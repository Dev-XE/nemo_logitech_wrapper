#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import threading

try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import Gst, GLib
except Exception as e:
    raise RuntimeError(
        "Missing GStreamer Python bindings. Install: sudo apt install python3-gi gir1.2-gstreamer-1.0"
    ) from e


class GstTxNode(Node):
    def __init__(self):
        super().__init__("gst_tx_node")

        # Params matching your pipeline
        self.declare_parameter("device", "/dev/video2")  # Logitech C270 HD WEBCAM
        self.declare_parameter("host", "192.168.2.1")
        self.declare_parameter("port", 5001)  # Logitech — separate port from Groov-e (5000)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("bitrate_kbps", 5000)
        self.declare_parameter("speed_preset", "veryfast")  # ultrafast/superfast/veryfast/faster/...
        self.declare_parameter("key_int_max", 60)
        self.declare_parameter("use_videoconvert", True)

        self.loop = None
        self.pipeline = None
        self._gst_thread = None

        Gst.init(None)

        pipe = self._build_pipeline()
        self.get_logger().info(f"Starting TX pipeline:\n{pipe}")

        self.pipeline = Gst.parse_launch(pipe)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.loop = GLib.MainLoop()

        # Start pipeline + GLib loop in background thread
        self._gst_thread = threading.Thread(target=self._run, daemon=True)
        self._gst_thread.start()

        # Ensure we stop on shutdown
        rclpy.get_default_context().on_shutdown(self._shutdown_hook)

    def _build_pipeline(self) -> str:
        device = self.get_parameter("device").get_parameter_value().string_value
        host = self.get_parameter("host").get_parameter_value().string_value
        port = self.get_parameter("port").get_parameter_value().integer_value
        width = self.get_parameter("width").get_parameter_value().integer_value
        height = self.get_parameter("height").get_parameter_value().integer_value
        fps = self.get_parameter("fps").get_parameter_value().integer_value
        bitrate = self.get_parameter("bitrate_kbps").get_parameter_value().integer_value
        speed = self.get_parameter("speed_preset").get_parameter_value().string_value
        keyint = self.get_parameter("key_int_max").get_parameter_value().integer_value
        use_vc = self.get_parameter("use_videoconvert").get_parameter_value().bool_value

        convert = "videoconvert !" if use_vc else ""

        # Your exact TX pipeline, just parameterized
        return (
            f'v4l2src device={device} ! '
            f'video/x-raw,width={width},height={height},framerate={fps}/1 ! '
            f'{convert} '
            f'x264enc tune=zerolatency speed-preset={speed} bitrate={bitrate} '
            f'key-int-max={keyint} bframes=0 ! '
            f'rtph264pay config-interval=1 pt=96 ! '
            f'udpsink host={host} port={port}'
        )

    def _run(self):
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error("Failed to set TX pipeline to PLAYING")
            return

        try:
            self.loop.run()
        except Exception as e:
            self.get_logger().error(f"GLib loop exception: {e}")

    def _on_bus_message(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            self.get_logger().error(f"Gst ERROR: {err} | debug: {dbg}")
        elif t == Gst.MessageType.EOS:
            self.get_logger().warn("Gst EOS (end of stream)")
        elif t == Gst.MessageType.WARNING:
            warn, dbg = msg.parse_warning()
            self.get_logger().warn(f"Gst WARNING: {warn} | debug: {dbg}")

    def _shutdown_hook(self):
        self.get_logger().info("Stopping TX pipeline...")
        try:
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
            if self.loop is not None and self.loop.is_running():
                self.loop.quit()
        except Exception:
            pass


def main():
    rclpy.init()
    node = GstTxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()