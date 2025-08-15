import paho.mqtt.client as mqtt
import json
import base64
from datetime import datetime

# --- MQTT Settings ---
broker_address = "broker.hivemq.com"
port = 1883
topic = "yolo/data"

# --- Data to be sent ---
# Image data
image_path = "yolo/source/side.png"
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Other data (using dummy values for now)
data = {
    "vehicle_speed": 10,
    "recognition_result": "person",
    "relative_distance": 20.5,
    "warning_level": 1,
    "image": encoded_image,
    "timestamp": datetime.now().isoformat()
}

# Convert dictionary to JSON
payload = json.dumps(data)

# --- MQTT Publish ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_publish(client, userdata, mid):
    print(f"Message Published to topic {topic}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "python_publisher")
client.on_connect = on_connect
client.on_publish = on_publish

try:
    client.connect(broker_address, port, 60)
    client.loop_start()
    result = client.publish(topic, payload)
    result.wait_for_publish() # Wait for publish to complete
    client.loop_stop()
    client.disconnect()
    print("Disconnected from MQTT Broker.")

except Exception as e:
    print(f"An error occurred: {e}")
