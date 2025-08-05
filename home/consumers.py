
import json
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from model.words import SignLanguageProcessor, load_models_once

load_models_once()

class VideoConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = SignLanguageProcessor()
        self.task = None # This variable is less relevant in a single-threaded model, but kept for consistency

    async def connect(self):
        await self.accept()
        print("WebSocket connected from:", self.scope['client'])

    async def disconnect(self, close_code):
        print("WebSocket disconnected from:", self.scope['client'])
 
        if self.task:
  
            self.task.cancel()
        print("Consumer shut down gracefully.")
        pass

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        frame_base64 = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_base64)

        sentence = await sync_to_async(self.processor.process_frame)(frame_bytes)

        await self.send_result_to_client(sentence)



    async def send_result_to_client(self, sentence):
        if sentence != ' ' and sentence != None:
            await self.send(text_data=json.dumps({
                'sentence': sentence
            }))
            