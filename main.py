import os
import threading
import logging
import webview

from api import app


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"), 
        logging.StreamHandler()
    ]
)

# Set up file handler to capture all debug info and errors
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
file_handler.setFormatter(file_formatter)


class JSAPI:
    def select_folder(self):
        window = webview.active_window()
        if not window:
            return None
        result = window.create_file_dialog(webview.FOLDER_DIALOG)
        return result[0] if result else None

    def select_file(self):
        window = webview.active_window()
        if not window:
            return None
        result = window.create_file_dialog(webview.OPEN_DIALOG)
        return result[0] if result else None


def run_fastapi():
    port = int(os.getenv("PORT", "9001"))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", reload=False)


def run_webview():
    api = JSAPI()
    webview.create_window("OperaFOR", url=f"http://localhost:{os.getenv('PORT', '9001')}", js_api=api)
    webview.start()


if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    run_webview()
