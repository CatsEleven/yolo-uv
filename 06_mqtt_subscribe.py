import paho.mqtt.client as mqtt
import json
import base64
from datetime import datetime
import os

# --- MQTT Settings ---
broker_address = "broker.hivemq.com"
port = 1883
topic = "yolo/data"

# --- Directory for saving images ---
save_dir = "mqtt"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    print(f"Received message from topic: {msg.topic}")
    try:
        # Decode payload
        data = json.loads(msg.payload.decode())

        # Print data except for the image
        image_data = data.pop("image", None)
        print("--- Received Data ---")
        print(json.dumps(data, indent=4))
        print("---------------------\n")


        # Decode and save the image
        if image_data:
            image_bytes = base64.b64decode(image_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = os.path.join(save_dir, f"{timestamp}.png")
            with open(file_path, "wb") as image_file:
                image_file.write(image_bytes)
            print(f"Image saved to: {file_path}")

    except json.JSONDecodeError:
        print("Error decoding JSON")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "python_subscriber")
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(broker_address, port, 60)
    # Run a blocking loop to keep listening for messages
    client.loop_forever()

except KeyboardInterrupt:
    print("Subscriber stopped.")
    client.disconnect()
except Exception as e:
    print(f"An error occurred: {e}")
    client.disconnect()
