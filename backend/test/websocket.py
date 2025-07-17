import asyncio
import websockets

async def test_ws():
    uri = "ws://localhost:8000/train/ws/logs/ppo-job-4214d0af-290e-4c18-b4a2-68fa29b8caf3"  # replace with real job_id
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                msg = await websocket.recv()
                print(msg)
            except websockets.exceptions.ConnectionClosed:
                break

asyncio.run(test_ws())
