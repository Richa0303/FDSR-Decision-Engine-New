# Copilot Instructions for FDSR-Decision-Engine

## Overview
This project is a human-centered framework for transdisciplinary decision-making, featuring a Flask backend, a CLI chatbot, and a web frontend. It supports fuzzy search and semantic recommendations for decision methods, with data-driven workflows and feedback logging.

## Architecture
- **backend/**: Flask app (`app.py`) serves API endpoints, renders the main UI, and logs feedback. Relies on `semantic_sim.py` for method similarity and `database.py` for MySQL integration.
- **chatbot/**: CLI tool (`chatbot.py`) for exploring decision methods interactively. Uses local JSON data and supports keyword/phase search.
- **frontend/**: Static HTML (`index.html`) for the main UI, styled with custom CSS and Vis.js for network visualization.
- **data.json**: Central data file for decision methods, used by both backend and chatbot.

## Key Workflows
- **Run backend server**: Launch Flask app from `backend/app.py`. (No build step required.)
- **Chatbot CLI**: Run `chatbot/chatbot.py` for interactive exploration.
- **Database**: MySQL required for `database.py` (see connection details in file). Not used by default Flask endpoints.
- **Feedback Logging**: POST to `/feedback` endpoint; logs to `feedback_log.jsonl`.
- **Semantic Recommendations**: Uses method embeddings in `data.json` and `semantic_sim.py` for similarity search.

## Patterns & Conventions
- **Data Access**: Always read/write methods from `data.json`. Use helper functions (`load_methods`) in both backend and chatbot.
- **Fuzzy Search**: Use `difflib.get_close_matches` for method lookup in backend.
- **Phase Membership**: Methods are grouped by `phase_membership` (see chatbot and backend logic).
- **Embeddings**: Each method in `data.json` has an `embedding` vector for semantic similarity.
- **Frontend**: UI is rendered via Flask (`render_template`), with network visualization using Vis.js.
- **Feedback**: All feedback is appended as JSON lines to `feedback_log.jsonl`.

## Integration Points
- **Flask <-> Frontend**: Main page served at `/`, data at `/data.json`, feedback at `/feedback`.
- **Backend <-> Database**: Only used for advanced method insertion/search (see `database.py`).
- **Semantic Search**: `semantic_sim.py` provides similarity functions for recommendations.

## Examples
- To add a new decision method, update `data.json` and optionally use `database.py` for MySQL.
- To extend fuzzy search, modify `find_method_by_title_fuzzy` in `backend/app.py`.
- To visualize new data, update `frontend/index.html` and ensure Flask serves it correctly.

## References
- `backend/app.py`, `backend/semantic_sim.py`, `backend/database.py`, `chatbot/chatbot.py`, `frontend/index.html`, `data.json`

---
For questions or unclear conventions, review the referenced files or ask for clarification.
