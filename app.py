"""Entrypoint for the Hugging Face Space (Gradio SDK).

The actual app is the FastAPI backend + static frontend from server/main.py.
No manual uvicorn.run() here: the Spaces runtime finds the module-level
`demo` object and serves it directly. Calling uvicorn.run() ourselves as
well was binding port 7860 twice in the same process, which is what was
crashing the container.
"""
import gradio as gr

from server.main import app as fastapi_app

status_page = gr.Interface(
    fn=lambda: "Titanic Survival Prediction API is running. Visit / for the app.",
    inputs=None,
    outputs="text",
)

demo = gr.mount_gradio_app(fastapi_app, status_page, path="/status")