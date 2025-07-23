# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]
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
  - Currently, Claude 3.7 Sonnet is used as a benchmark, though token counts may not be 100% accurate—this is intended to provide a general understanding of token usage.
  - Tool replies (e.g., raw Genie answers) and tool-specific tokens are not factored into the token count.
- Own autonomous ReAct agent with a hard limit of 6 nodes.
- Agent in deployment format, designed for easy transition to Mosaic AI model serving once OBO authentication is supported.
- Tracing support (currently deactivated, as the agent is not served via Model Serving).
- Streaming support for full messages (not single chunks, to ensure robustness, output validation, and format cleaning).

### Updated
- Deployment notebook and process are now streamlined into a one-click experience
- System prompt
- Improved Genie error handling: errors are now posted to the agent immediately upon failure instead of waiting for a timeout
- Logger to it's own file
- Agent can now handle Genies Null attachment values to do failed queries

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
- Print to logging, removed all tokens to use sp oAuth, changed Genie authentication to under Apps Service Principal, 
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