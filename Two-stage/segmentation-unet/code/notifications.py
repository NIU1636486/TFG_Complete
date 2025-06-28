import requests


def send_notification(content, title="TFG - UNet"):
    requests.post("https://ntfy.sh/tfg-pol", data = content,
                headers={"title": title,})
                        