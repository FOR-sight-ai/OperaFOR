# OperaFOR

OperaFOR is a minimalist graphical interface for LLM and RAG agents, designed to provide fast and simple access to powerful search tools via the MCP protocol. Its streamlined architecture—just a single HTML file and a single Python file—enables rapid iteration and creativity.

## ✨ Main Features

- **Minimalist interface**: just one HTML page, one Python file.
- **Access to advanced search tools** via MCP (Model Context Protocol).
- **Local execution**: runs with pywebview and can be packaged with pyinstaller.
- **A task-oriented experience**: the interface emphasizes tasks, making it clear that answers are not instantaneous (unlike a chatbot).
- **Supports RAG (Retrieval-Augmented Generation)** and LLM agents.
- **Organize your work**: manage sandboxes/conversations for different sessions.
- **Integrated Git system**: sandboxes are versioned conversations, allowing rollback and version control of your research sessions.

## 🚀 Development Installation


Install the required dependencies:

```bash
uv pip install .
```
    
To start the application in development mode :

```bash
uv run main.py
```

The interface will automatically open in a pywebview window.

## 📦 Windows Executable

A standalone Windows executable is available on the [GitHub Releases page](https://github.com/FOR-sight-ai/OperaFOR/releases) — no Python installation required!


## 🛠️ Usage

- Configure your API key and LLM settings in the configuration panel.
- Create sandboxes (similar to conversations) to organize your tasks or research.
- Each sandbox is versioned using an integrated Git system, enabling you to rollback, track changes, and manage versions of your conversations.
- Launch requests, follow their progress, and view results in a task-oriented timeline interface.


## 📁 Project Structure

- `main.py`: FastAPI server, sandboxes management, MCP integration, pywebview launcher.
- `index.html`: unique, responsive, task-oriented user interface.

## 📜 License

This project is open source, under the MIT license.
