import os
import socketserver
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from src.common.param import args, data_args, model_args
from src.model_wrapper.edge_vlm import EdgeVLMWrapper
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.edge_vlm_rpc import recv_message, send_message


class EdgeVLMRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        payload = recv_message(self.request)
        try:
            episodes = [payload["episode"]]
            target_positions = [payload["target_position"]]
            assist_notices = [payload.get("assist_notice")]
            predict_done = self.server.dino_monitor.get_dino_results(
                episodes[0],
                payload.get("object_info"),
            )
            latest_position = np.asarray(episodes[0][-1]["sensors"]["state"]["position"][0:3], dtype=np.float64)
            target_position = np.asarray(target_positions[0], dtype=np.float64)
            distance_to_target_m = float(np.linalg.norm(latest_position - target_position))
            coarse_local = None
            llm_latency_ms = 0.0
            if not (predict_done and distance_to_target_m <= 20):
                coarse_local, llm_latency_ms = self.server.model_wrapper.run_coarse(
                    episodes,
                    target_positions,
                    assist_notices,
                )
                coarse_local = coarse_local[0].tolist()
            response = {
                "ok": True,
                "request_id": int(payload["request_id"]),
                "coarse_local": coarse_local,
                "llm_latency_ms": float(llm_latency_ms),
                "predict_done": bool(predict_done),
                "dino_distance_to_target_m": distance_to_target_m,
            }
        except BaseException as exc:
            response = {
                "ok": False,
                "request_id": payload.get("request_id"),
                "error": repr(exc),
            }
        send_message(self.request, response)


class ThreadedEdgeVLMServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def main():
    model_wrapper = EdgeVLMWrapper(model_args=model_args, data_args=data_args)
    model_wrapper.eval()
    dino_monitor = DinoMonitor.get_instance()

    with ThreadedEdgeVLMServer((args.edge_vlm_bind_host, int(args.edge_vlm_port)), EdgeVLMRequestHandler) as server:
        server.model_wrapper = model_wrapper
        server.dino_monitor = dino_monitor
        print(f"Edge VLM server listening on {args.edge_vlm_bind_host}:{args.edge_vlm_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
