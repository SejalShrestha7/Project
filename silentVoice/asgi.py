

import os
import django
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from home.routing import websocket_urlpatterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'silentVoice.settings')
django.setup()
print("ASGI py loaded")
application = ProtocolTypeRouter({
    # The ProtocolTypeRouter is the top-level application.
    # It must be configured exactly like this.
    
    # Standard HTTP requests will be handled by the Django ASGI application
    "http": get_asgi_application(),
    
    # WebSocket requests will be handled by this URLRouter
    "websocket": AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})