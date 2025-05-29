# home/views.py
from django.shortcuts import render
import logging

logger = logging.getLogger(__name__)

def landing_page(request):
    logger.info("Landing page view called!")  # Ini akan muncul di terminal
    return render(request, 'home/landing.html')
