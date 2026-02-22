# Changelog

All notable changes to this project will be documented in this file.

---
## [1.2.1] - 2026-02-22
### Fixed
- **Critical: Streaming UI not updating messages** - Each 100ms interval tick created a NEW message bubble instead of updating a single one, causing fragmented display and stuck typing indicator
- `update_streaming` callback rewritten with chunk accumulation: all chunks now stored in `streaming_state['chunks']` across ticks and rebuilt as a single message on each tick
- `chat_history` is no longer polluted with partial entries on every tick; only updated once on streaming completion
- Genie loading GIFs now rendered inline during streaming and converted to final status on completion
- Inter-chunk timeout increased from 5s to 15s (was too aggressive for normal LLM response times)
- Removed dead `_process_chunks_to_display()` and `_convert_genie_gifs()` helper methods (logic now inline in rewritten callback)
- Updated tests: replaced old helper method tests with `TestStreamingAccumulation` suite (10 tests covering chunk accumulation, display filtering, genie dedup, history management)

---
## [1.2.0] - 2026-02-22
### Added
- CLAUDE.md documentation for AI-assisted development
- Comprehensive test suite (135 tests) covering all modules:
  - `test_general_functions.py` - clean_text, tool creation, LLM calls, token counting, Pydantic models
  - `test_genie_functions.py` - Genie Space API, conversation management, input validation
  - `test_agents.py` - ReAct and Simple agent initialization, tool calling, prediction
  - `test_chatbot.py` - DatabricksChatbot layout, streaming, chunk processing, CSS, thread management
  - `test_app.py` - SSE endpoint, logger setup
- `conftest.py` with shared pytest fixtures (mocked WorkspaceClient, OpenAI, Flask context)
- `pytest.ini` configuration
- Server-Sent Events (SSE) streaming endpoint (`/api/stream/<username>`) for real-time streaming support
- `ToolException` class in `genie_functions.py` (previously referenced but never defined)
- Global `_global_lock` for thread-safe access to per-user dictionaries
- Error boundary in `update_streaming` callback with graceful fallback on exceptions
- Worker thread idle timeout (5 min) to clean up stale threads
- Guaranteed `__STREAMING_COMPLETE__` sentinel even on worker thread exceptions

### Updated
- Default LLM model from `databricks-claude-3-7-sonnet` to `databricks-claude-sonnet-4-6` (Claude Sonnet 4.6)
- All model references across config.yml, databricks.yml, DatabricksChatbot.py, general_functions.py, common_functions.py, Deployment notebook
- Streaming interval reduced from 200ms to 100ms for snappier UI updates
- Thread and queue cleanup now uses global lock for thread safety
- `_start_user_worker` initializes chunk_queues under global lock
- `clear_chat` callback uses global lock for cleanup
- Genie chunk deduplication fixed to handle ZWJ unicode escaping in Dash `str()` representation
- README.md updated: removed deprecated notice, updated model references to Claude Sonnet 4.6, added v1.2
- Deployment notebook updated for Claude Sonnet 4.6

### Fixed
- Race condition: Queue/thread dictionary access without synchronization
- Race condition: Non-atomic streaming state updates in callbacks
- Race condition: Worker thread accessing replaced queue after `_create_layout` resets
- Missing `ToolException` class (NameError on Genie validation errors)
- Worker thread silently dying on unhandled exceptions without sending completion sentinel
- Genie chunk deduplication failing due to ZWJ unicode escaping differences

---
## [1.0.0] - 2025-07-23
### Added
- CHANGELOG.md file
- Behalf-on-user authentication support to Apps on Genie spaces
- Dynamic LLM configurations for Apps deployment (Multi-LLM support)
- Created 'assets' folder and added logo pictures & Genie loading gif
- Pydantic validation for the Genie spaces tool-calling component
- Helper functions to automate Genie Space integration using OBO tokens
- Support for continuous conversation with Genie, enabling the agent to guide Genie during interactions
- OBO-token support enabled by default (the Agent automatically uses the end-user’s authentication token for Genie Spaces)
- Genie Spaces are updated automatically within the tool using official descriptions and end-user permissions
- Selected Genie Spaces are dynamically converted into tools for agents on the fly.
- A text cleaner is implemented for the agent's responses, refining updated ReAct answers into a more user-friendly format.
- An automated chat memory optimizer is in place:
  - Chat message history is capped at the last 25 messages unless the 100k token limit is reached.
  - The optimizer also manages token usage for chat message history.
  - Currently, Claude 3.7 Sonnet is used as a benchmark, though token counts may not be 100% accurate — this is intended to provide a general understanding of token usage.
  - Tool replies (e.g., raw Genie answers) and tool-specific tokens are not factored into the token count.
- Own autonomous ReAct agent with a hard limit of 6 nodes.
- Agent in deployment format, designed for easy transition to Mosaic AI model serving once OBO authentication is supported.
- Tracing support (currently deactivated, as the agent is not served via Model Serving).
- Streaming support for full messages (not single chunks, to ensure robustness, output validation, and format cleaning).
- Queue functionality to improve user-level session threading.
- Options box added next to the chat input area. Contains "Active Genie Spaces", "LLM", "Temperature" and "Max tokens" drop-down options - Populated automatically. Available Genie Spaces are shown based on end user's permissions.
- "Apply" button implemented to automatically update Agent parameters.
- Background streaming worker added to handle processing and checks.
- Enhanced overall functionality and error handling.
- Timeout backup process added with fine-tuned error messages.
- Genie GIFs now appear when a Genie is triggered, then switch to a static message once processing is complete.
- DAB (Databricks Asset Bundles) deployment support

### Updated
- Deployment notebook and process are now streamlined into a one-click experience
- System prompt
- Improved Genie error handling: errors are now posted to the agent immediately upon failure instead of waiting for a timeout
- Logger to it's own file
- Agent can now handle Genies Null attachment values to do failed queries
- Apps layout redesigned for better usability.
- Visualization improved and enhanced; assistant messages now support Markdown output.
- Apps now support streaming output only from agents.

### Removed
- DevOps beta ticket functionality 
- DevOps ticket functions
- Genie space IDs and descriptions are no longer required during deployment
---
## [0.9.1] - 2025-05-15
### Updated
- Genie data result processing to support max 50 rows

---
## [0.9.0] - 2025-05-12
### Added
- Flask auth
- Genie prompt to logs to improve visibility
- Apps message blocker while processing user requests
- Apps user level logging monitoring 

### Updated
- OpenAI client to be used from SDK instead of PAT token
- Print to logging, removed all tokens to use sp oAuth, changed Genie authentication to under Apps Service Principal
- Databricks SDK version, 
- Updated Genie code to run on SDK
- Systemp prompt
- Default LLM model to Claude 3.7. Sonnet
- Code documentation and documentation in overall, while cleaning code a bit
- Genie tool calling polling to every 0,7s and having 45 max retries
### Removed
- Documentation tool - it's unnecessary
---
## [BETA] - 2025-05-12
### Added
- First BETA release.
- Contained Multi-Genie agent solution with limited Azure DevOps ticket functionality
- Limited authentication and functionalities
- Deployment process