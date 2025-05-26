import json
import os

from PyDirectus.pydirectus import DirectusClient

class DirectusSender:
    def __init__(self):
        self.hostname = os.environ["DIRECTUS_HOSTNAME"]
        self.static_token = os.environ["DIRECTUS_STATIC_TOKEN"]

    def send(self):
        directus = DirectusClient(hostname = self.hostname, static_token = self.static_token)
        try:
            new_item_data = {"id": "value1", "name": "松村孝宏", "phone_number": "08012345678"}
            created_item = directus.create_item("license_number", data=new_item_data)
            return {
                "statusCode": 200,
                "body": json.dumps(created_item)
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)})
            }