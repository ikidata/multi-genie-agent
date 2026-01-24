# An Agentic Multi-Genie Solution on Databricks Apps

_**Deprecated - not being updated**_

This is a showcase repository for the Agentic Multi-Genie solution on Databricks Apps. This allows you to chain Genie spaces dynamically together using the Executor Agent, enabling you to effortlessly manage everything from a single interface (Databricks Apps). You can decide which Genie Spaces the agent can use and whether to use ReAct or simple agent. When ReAct option is set to "no," the solution defaults to a simple tool-calling agent, suitable for handling straightforward tasks with ease. But when it comes to complex challenges, like using multiple Genie Spaces to analyze customer behavior based on profitability, feedback and buying patterns, while factoring in average values and company policies, the ReAct Agent truly excels. It's time to unleash Genie Spaces as your personal data analysts!

The articles on the solution can be found here: 
* 📚 [Agentic Databricks Solutions - Have your own Multi-Genie assistant](https://www.ikidata.fi/post/agentic-databricks-solutions-have-your-own-multi-genie-assistant)
* 📚 [Older article explaining the first iteration and thought process](https://www.ikidata.fi/post/creating-a-multi-genie-agent-solution-with-databricks-apps-code-included)

---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites and Installation](#prerequisites-and-installation)
- [Usage](#usage)
- [Known limitations](#Limitations)
- [Versions](#Version-0.9–23.07.2025)

---

## Introduction

![usage](https://static.wixstatic.com/media/729282_614a582cba3f4b789425af0fc7176104~mv2.gif)

This agentic solution demonstrates the true power of the Multi-Genie-Agent approach. Each Genie species is totally focused on a specific topic and treated as an isolated entity. Different teams and subject matter experts can continue developing these entities without external dependencies. You can then piece the puzzle together from different Genie parts, offering an effortless, comprehensive solution. 

Remember, the goal of this solution is to demonstrate possibilities and provide new insights, rather than offering production-ready solutions. Take the insights gained and continue developing your own solutions further.

Databricks Apps provide user-level Python sessions to ensure secure usage. This also means that Genie Spaces are accessed using on-behalf-of-user (OBO) authentication. In practice, the OBO token is fetched from the Dash header and used to call the Genie REST API endpoint via the SDK. Apps support OBO tokens (the required Genie Spaces API scope is added automatically), but this also means that all Apps users will use this authentication method. All other authentication is handled via the App-specific service principal.

You can choose from ReAct and simple again from options. Simple agent is limited to one tool call but ReAct agent have autonomy all the way till six (6) nodes (feedback loop steps). It's limited to six to avoid expensive infinity loops. If the agent cannot finish analyze, you can request that it keeps working on the request. Thanks for short-term memory, it can start feedback loop again and use the history information. Short-term memory is limited to 25 messages and then messages are being deleted automatically from the latest, keeping active messages on 25. Genie spaces are being detected automatically using OBO-token. You can activate or deactive Genie spaces based on your needs. It's recommended to keep only relevant Genie spaces to help agent perform even better. And remember, Genie spaces are automatically added as a tools but requires proper descriptions (which are fectched from Genie Space metadata). Without a good description, it's impossible for the agent understand what's going on and how to use the Genie Space. Keep in mind that you can use as many Genie Spaces as you want.

Since Databricks offers good pre-built templates for creating apps, the Dash chatbot template has been used here. On top of it, a lot more sophisticated features have been built to improve user experience.

---

## Prerequisites and Installation 

The deployment notebook contains a more detailed process of the required steps to get this working. Well let's be honest, you have to choose your LLM model and then just click deploy - the rest is fully automated. Claude 3.7. Sonnet is added automatically (Pay-per-Token billing on Databricks). Databricks.yml is also added if you prefer to use Databricks Asset Bundles for deployment. 

**_Since Databricks Apps on-behalf-of-user authentication is in public preview, you have to activate it in the workspace first._**

**Implementation process**
- Ensure you have existing Genie Spaces
- Clone the repo to your Databricks workspace (instructions https://docs.databricks.com/aws/en/repos/git-operations-with-repos).
- Populate the deployment notebook (LLM model) and deploy
- Activate Public Preview in the workspace if you haven't done it yet

---

## Architecture
![architecture](https://static.wixstatic.com/media/729282_d307d048531949928dbbb7eff00b182b~mv2.gif)

---

## Usage

📄 [Instructions on how to use the solution](https://www.ikidata.fi/post/agentic-databricks-solutions-have-your-own-multi-genie-assistant)

The deployment notebook automatically deploys Databricks apps, which takes around 10 minutes. From the Databricks apps logs, you can see more detailed metadata on how they are performing. Remember to optimize the separate Genie spaces to improve results, adjust the executor agent's system prompt to fit your needs, and do the same for the tool descriptions. And once you're done, don't forget to stop the apps.

---

## Limitations
- Keep in mind that this isn't for prod use. Although it gives a clear understanding of how you can use it as a template for further development.
- The model endpoint LLM model requires tool support. Currently tested models:
  - Claude 3.7: Works really well while being thorough
  - GPT-4o: Performs well
  - GPT-4.1 series: Performs well
  - *OpenAI's reasoning models: Performed decently after disabling temperature settings, though still prefer the GPT-4.1 series.

- Recommended models: For the best user-experience, I'd say GPT-4.1 series models or GPT-4o. But Claude 3.7 continues to deliver the best results overall, easily.
- All validation has been done in Azure Databricks.

---

## Version 1.0 – 23.07.2025
This version includes improved features and refinements across key components. For a complete list of changes, please read update history from CHANGELOG.md file.
* [View changelog](./CHANGELOG.md)
* *Old version can be found here: [v.0.9](https://github.com/ikidata/multi-genie-agent/tree/version-0.9)*


<hr style="border: 1px solid #666; width: 80%;">  
<h3>  
    Agentic Automation on Databricks
</h3>  

<a href="https://raw.githubusercontent.com/ikidata/ikidata_public_pictures/refs/heads/main/logos/new_ikidata_background_logo.gif">  
  <img src="https://raw.githubusercontent.com/ikidata/ikidata_public_pictures/refs/heads/main/logos/new_ikidata_background_logo.gif" alt="Ikidata Logo">  
</a>  


<div class="follow-linkedin">  
  <a href="https://www.linkedin.com/company/ikidata/" target="_blank">  
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn Logo" style="width: 10px; vertical-align: middle;">  
      Follow Ikidata on LinkedIn  
  </a>  
</div>  
 
<div class="explore-website">  
  <a href="https://www.ikidata.fi" target="_blank">  
    <img src="https://github.com/ikidata/ikidata_public_pictures/blob/main/logos/Ikidata_aurora.png" alt="Ikidata Website Logo" style="width: 10px; vertical-align: middle;">  
      Explore our website  
  </a>  
</div>  

<div class="gain-knowledge">  
  <a href="https://www.ikidata.fi/knowledge" target="_blank">  
    📚Gain new Databricks knowledge  
  </a>  
</div>  
