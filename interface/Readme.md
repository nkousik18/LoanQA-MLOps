## Environment Setup

### Prerequisites


From the root directory, start the backend:

```bash
flask run
```

Then open the interface in a browser at:

```
http://127.0.0.1:8000
```

## Directory Structure

```
interface/
├── templates/
│   └── index.html              # Main interface page
│
├── static/
│   ├── css/
│   │   ├── style.css           # Layout, buttons, typography
│   │   └── modal.css           # Styling for modals & response box
│   │
│   └── js/
│       ├── api_calls.js        # Handles API requests to Flask endpoints
│       ├── upload_text_viewer.js # Manages uploads & renders extracted text
│       ├── selection_handler.js  # Enables text selection and action toolbar
│       ├── modal_handler.js      # Controls persistent response box behavior
│       └── helpers.js (optional) # Placeholder for future utilities
```


## Implementation Details

### File Upload & OCR Integration

* The user uploads `.pdf` files through the upload form.
* The file is sent to the Flask backend at:

  ```
  POST /api/upload
  ```
* The backend performs OCR (via PaddleOCR / PP-OCRv5 pipeline) and saves the extracted text under:

  ```
  data/clean_texts/
  ```
* The server returns a JSON response:

  ```json
  { "content_path": "data/clean_texts/filename_timestamp.txt" }
  ```

### Extracted Text Viewer

* The file path returned from the upload is then passed to:

  ```
  GET /api/read_text?path=<encoded_path>
  ```
* The **`upload_text_viewer.js`** script retrieves the full text and injects it into the `<div id="document-content">` element dynamically.
* Long text files are wrapped with `white-space: pre-wrap` and scrollable for easy viewing.
* Color scheme optimized for dark background readability (`#e0f7fa` on `#111`).


### Text Selection & Toolbar Actions

* Users can **select text segments** in the document viewer.
* When text is selected, a floating toolbar appears with three buttons:

  * **Summary**
  * **Translate**
  * **Explain**
* These buttons are dynamically controlled by `selection_handler.js`.


### API Calls to Backend LLM Routes

* All toolbar actions trigger respective Flask API endpoints via `api_calls.js`:

  * `POST /api/summary`
  * `POST /api/translate`
  * `POST /api/explain`

* Example request payloads:

  ```json
  { "text": "Selected text for summarization" }
  ```

  ```json
  { "text": "Selected text for translation", "lang": "fr" }
  ```

* The backend runs prompt routing logic (`prompt_router.py` → Ollama / Phi-3 model) and returns structured responses:

  ```json
  { "summary": "...processed output..." }
  ```


### Chatbot Integration

* A floating chatbot button is provided:

  ```html
  <button id="chatbot-btn" onclick="window.location.href='/api/chat'"> Chatbot</button>
  ```
* Navigates to the conversational chat interface served by Flask at `/api/chat`.
* Chatbot logic connects to the same vectorstore used in the document index (Chroma backend).


## Key JavaScript Modules

### `upload_text_viewer.js`

Handles:

* File submission and upload to `/api/upload`
* Fetching and rendering the processed `.txt` output
* Converts raw newlines into readable paragraphs
* Displays loading and completion messages

### `api_calls.js`

Handles:

* Asynchronous API calls for `/summary`, `/translate`, `/explain`
* Dynamic rendering of LLM responses inside the response box
* Centralized error handling and fallback alerts
* Uses modern `fetch` with JSON payloads

### `selection_handler.js`

Handles:

* Text highlighting detection
* Toolbar visibility logic
* Debounce behavior to avoid flickering when user clicks outside selection

### `modal_handler.js`

Handles:

* Persistent response box logic
* Manual close button
* Optional animation and scroll control
* Cross-browser styling consistency



## Styling Notes

### `style.css`

* Defines layout, background, buttons, fonts, and scroll behavior.
* Implements gradient buttons (`linear-gradient(135deg, #00b0ff, #0078d4)`).
* Responsive up to 950px width for centered document viewer.

### `modal.css`

* Custom styling for persistent response box:

  ```css
  .response-box {
      position: fixed;
      bottom: 40px;
      right: 40px;
      width: 420px;
      background: #111;
      color: #e0f7fa;
      border: 1px solid #00b0ff;
      border-radius: 10px;
      box-shadow: 0 0 15px rgba(0, 176, 255, 0.3);
      z-index: 1000;
      display: none;
      flex-direction: column;
      overflow: hidden;
  }
  ```
* `.response-header` keeps the close “×” button aligned right beside the title.
* `.response-content` is scrollable and auto-resizes with long LLM outputs.


## Browser Compatibility

All major functionality (upload, selection, API calls, persistent responses) verified on **Safari** and **Chrome (macOS)**.


## Testing and Debugging

Run the backend with debug logs:

```bash
flask run --debug
```

Verify logs for:

* `/api/upload` → OCR extraction success
* `/api/read_text` → file streaming success
* `/api/summary`, `/api/translate`, `/api/explain` → model inference logs
