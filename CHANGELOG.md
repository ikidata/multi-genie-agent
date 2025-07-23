# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]
### Added
- CHANGELOG.md file
- Behalf-on-user authentication support to Apps on Genie spaces
- Dynamic LLM configurations for Apps deployment (Multi-LLM support)
- Created 'assets' folder and added logo pictures & Genie loading gif

### Updated
- Deployment notebook and process are now streamlined into a one-click experience.
- Genie space IDs and descriptions are no longer required during deployment.
- System prompt

### Removed
- DevOps beta ticket functionality (code & deployment part)

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