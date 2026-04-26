import zmq
import json
import base64
import numpy as np
import cv2
import asyncio
import websockets

# ── ZeroMQ subscriber ─────────────────────────────────────────
context = zmq.Context()
sub = context.socket(zmq.SUB)
# Keep only the most recent frame so the viewer never lags behind the
# publisher — older frames are discarded instead of queueing up.
sub.setsockopt(zmq.CONFLATE, 1)
sub.setsockopt(zmq.RCVHWM, 1)
sub.connect("tcp://192.168.0.139:5555")   # ← replace with your Pi's IP
#sub.connect("tcp://192.168.0.158:5555")   # ← replace with your Pi's IP
sub.setsockopt_string(zmq.SUBSCRIBE, "")

# ── Foxglove WebSocket clients ────────────────────────────────
connected_clients = set()

async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"Foxglove client connected: {websocket.remote_address}")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)

async def broadcast_loop():
    loop = asyncio.get_event_loop()
    while True:
        # Non-blocking ZMQ receive via executor
        raw = await loop.run_in_executor(None, sub.recv_string)
        data = json.loads(raw)

        if connected_clients:
            # Send full payload to all Foxglove clients
            msg = json.dumps({
                "timestamp":  data["timestamp"],
                "image_b64":  data["image_b64"],
                "detections": data["detections"]
            })
            websockets.broadcast(connected_clients, msg)

        # Optional: show locally too
        img_bytes = base64.b64decode(data["image_b64"])
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break

async def main():
    async with websockets.serve(ws_handler, "localhost", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await broadcast_loop()

asyncio.run(main())