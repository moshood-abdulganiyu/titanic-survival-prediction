"""Entrypoint for the Hugging Face Space (Gradio SDK).

The actual app is the FastAPI backend + static frontend from server/main.py.
Gradio SDK spaces need something named `demo` for the platform to detect,
so this file mounts the real app under Gradio rather than building a
Gradio UI we don't need.
"""
import gradio as gr

from server.main import app as fastapi_app

# A single-element status page. Not the real UI, just satisfies the
# Gradio SDK's requirement for a `demo` object with a launch(). The
# actual site is served by the FastAPI mount below at "/".
status_page = gr.Interface(
    fn=lambda: "Titanic Survival Prediction API is running. Visit / for the app.",
    inputs=None,
    outputs="text",
)

demo = gr.mount_gradio_app(fastapi_app, status_page, path="/status")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(demo, host="0.0.0.0", port=7860)