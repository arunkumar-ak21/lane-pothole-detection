import requests
import time

class TelegramNotifier:
    def __init__(self, bot_token, chat_id, min_interval_sec=5):
        self.bot_token = "8541464745:AAEOegMT8BloGTQeJF4fM9UAxLtA1Au0NqE"
        self.chat_id = 1872508623
        self.base = f"https://api.telegram.org/bot{bot_token}"
        self.min_interval = min_interval_sec
        self.last_sent = 0

    def send(self, message):
        now = time.time()
        if now - self.last_sent < self.min_interval:
            return False

        url = f"{self.base}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message
        }
        try:
            requests.post(url, data=data, timeout=2)
            self.last_sent = now
            return True
        except Exception as e:
            print("Telegram error:", e)
            return False
