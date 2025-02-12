import modal
import time
import sys
import re
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Create an app with the Modal framework
app = modal.App("my-app")
app.image = modal.Image.debian_slim().pip_install("fastapi", "websockets")

VOLUME_NAME = "Hunyuan-outputs"

outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")  # remote path for saving video outputs
# List to store all active WebSocket connections
active_clients = []
# This is the global buffer where we store the file's content
file_content = ""

LOG_FILE_PATH = f"{OUTPUTS_PATH}/"+'output.log'

# Function to check if the file is at 100% progress
def is_file_progress_complete(file_content: str) -> bool:
    # Example: assuming the log file contains a line like "Progress: 100%"
    if "Loading checkpoint shards: 100%|#" in file_content:
        progress_bar_pattern = r'Loading checkpoint shards: \d+%\|[^\|]+\|.*?s/it'
        matches = list(re.finditer(progress_bar_pattern, file_content))
        if matches:
            # Get the last match
            last_match = matches[-1]
            # Get the end position of the last match (where the progress bar ends)
            end_position = last_match.end()
            print(f"End position of the loading checkpoint progress bar: {end_position}")
        else:
            print("Pattern not found.")
            if "100%|#" in file_content:
                 return True
        if "100%|#" in file_content[end_position+1:] :
            return True
    return False

# Function to send the log file content to the WebSocket
def send_file_content():
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'r') as file:
            file_content = file.read()
    else:
        file_content = "Not found"
    return file_content

# Initial file content send
file_content = send_file_content()
    
# Function to capture terminal logs and stream them to WebSocket
@app.function(volumes={
        OUTPUTS_PATH: outputs,  # videos will be saved to a distributed volume
    })

@modal.asgi_app()
def log_endpoint():
        # Create FastAPI app
        # from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        import asyncio
        import os
        fastapi_app = FastAPI()

        # WebSocket handler to send logs
        @fastapi_app.websocket("/ws")
        async def websocket_handler(websocket: WebSocket) -> None:
            #  with torch.no_grad():
            await websocket.accept()
            active_clients.append(websocket)
            print("Websocket connected!")  # Verify connection
            await websocket.send_text("Websocket connected!")
            await websocket.send_text(file_content)

            # Track last activity timestamp
            last_activity = asyncio.get_event_loop().time()

            # Ping/Pong mechanism (ping every 30 seconds)
            async def ping_pong():
                nonlocal last_activity
                while True:
                    await asyncio.sleep(30)  # Ping every 30 seconds
                    if asyncio.get_event_loop().time() - last_activity > 60:
                        # If no activity for more than 60 seconds, disconnect the client
                        print("Client not responding, closing WebSocket.")
                        await websocket.close()
                        active_clients.remove(websocket)
                        break
                    # Send a ping message and wait for pong response
                    try:
                        await websocket.send_text("ping")
                        response = await asyncio.wait_for(websocket.receive_text(), timeout=10)  # Wait for pong response
                        print("Message received from client", response)
                        if response == "pong":
                            print("Response is pong")
                            last_activity = asyncio.get_event_loop().time()  # Update last activity time
                        else:
                            print(f"Unexpected response: {response}")
                    except asyncio.TimeoutError:
                        print("No pong received. Closing WebSocket.")
                        await websocket.close()
                        active_clients.remove(websocket)
                        break

            # Function to poll the file for changes
            async def poll_file_for_changes():
                last_modified_time = None

                while True:
                    outputs.reload()
                    try:
                        # Get the last modified time of the file
                        current_modified_time = os.path.getmtime(LOG_FILE_PATH)
                        
                        if last_modified_time is None:
                            last_modified_time = current_modified_time
                        elif current_modified_time > last_modified_time:
                            # If the file has been modified
                            print(f"File modified: {LOG_FILE_PATH}")
                            
                            # Send the updated content to all connected WebSocket clients
                            for client in active_clients:
                                # Read the updated content of the file
                                file_content = send_file_content()
                                print(file_content)
                                asyncio.create_task(client.send_text(file_content))

                                # Check if the file content indicates 100% progress
                                if is_file_progress_complete(file_content):
                                    print("File reached 100% progress. Disconnecting WebSocket.")
                                    await client.send_text(file_content)
                                    # print(file_content)
                                    await client.send_text("File reached 100% progress. Disconnecting WebSocket.")
                                    await client.close()
                                    active_clients.remove(client)
                                    break
                                if not active_clients:
                                    await websocket.close()
                                    break        

                            # Update the last modified time
                            last_modified_time = current_modified_time
                        
                    except FileNotFoundError:
                        print(f"File not found: {LOG_FILE_PATH}")
                        for client in active_clients:
                            asyncio.create_task(client.send_text("Log file not generated yet"))
                    except IOError as e:
                        print(f"Error reading the file: {e}")
                        for client in active_clients:
                            asyncio.create_task(client.send_text("Error reading the file"))

                    # Sleep for a while before checking the file again
                    await asyncio.sleep(2)

            try:
                # Start the ping/pong mechanism to check if the client is alive
                asyncio.create_task(ping_pong())
                # Start the polling function to check for file changes
                asyncio.create_task(poll_file_for_changes())
                while True:
                    # Keep the connection alive, and send new content if file is updated
                    await asyncio.sleep(5)  # Adjust sleep time based on your needs

            except WebSocketDisconnect:
                active_clients.remove(websocket)
                print("Disconnecting WebSocket.")
                await websocket.close()

        return fastapi_app