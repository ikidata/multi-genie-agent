# A Multi-Genie-Agent Solution with Databricks Apps

GenAI Agent development belongs to everyone! Databricks offers an amazing platform for it, and the purpose of this repo is to make the first steps even easier. This is a showcase repository for the Multi-Genie Agent solution with Databricks Apps. This allows you to chain Genie spaces together using the Executor Agent, enabling you to effortlessly manage everything from a single interface. The Executor Agent automatically passes your queries to the appropriate Genie Space, making this scalable and easy to manage. For example, you can initially inquire about sales data from the "Sales Genie," then related data pipelines from the "Metadata Genie" and finally create a DevOps ticket – all through one interface.

The article on the solution can be found here: [Creating a Multi-Genie-Agent Solution with Databricks](https://www.ikidata.fi/post/creating-a-multi-genie-agent-solution-with-databricks-apps-code-included)

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites and Installation](#prerequisites-and-installation)
- [Usage](#usage)
- [Known limitations](#Limitations)
- [More information](#More-information)

## Introduction

![architecture](https://static.wixstatic.com/media/729282_d7ddabc2e0dd4d5589ab7978f93820de~mv2.gif)

This agentic solution demonstrates the true power of the Multi-Genie-Agent approach. Each Genie species is totally focused on a specific topic and treated as an isolated entity. Different teams and subject matter experts can continue developing these entities without external dependencies. You can then piece the puzzle together from different Genie parts, offering an effortless, comprehensive solution. 

Remember, the goal of this solution is to demonstrate possibilities and provide new insights, rather than offering production-ready solutions. Take the insights gained and continue developing your own solutions further. Because of the nature of this PoC, many essential parts are missing, such as proper validations and more advanced functionalities.

Genie subagents are limited to 30 seconds before timeout. The Databricks Apps model endpoint acts as an executor agent, passing prompts forward to Genie spaces and back. It uses a simple function call agentic approach with temporarily extended memory, wrapping Genie outputs before returning the value. All phases are visible in the logs, but the end user can only see the final result. The executor agent has the freedom to choose subagents, so tool metadata is crucial for smooth operation. Thanks to short-term memory, it's possible to ask questions from different Genie spaces and then pass them to DevOps.

Since Databricks offers good pre-built templates for creating apps, the Dash chatbot template has been used here. On top of it, more sophisticated features have been built, but the core app functionality (like callbacks) and visualization have been modified only slightly.

### Used tools in the example:
-   databricks_genie_space - Prebuilt Genie Spaces which is built on top of System Tables (outside of this repo scope)
-   customer_data_genie_space - Prebuilt Genie Spaces which is built on top fabricated customer data (outside of this repo scope)
-   create_update_devops_ticket - Create / Update Azure DevOps tickets (capabilities limited for demo purposes)

**Keep in mind that you can use more Genie spaces than just two.**

## Prerequisites and Installation 

The deployment notebook contains a more detailed process of the required steps to get this working. Automated deployment code (e.g., using Databricks Asset Bundles) is excluded due to unique authentication and modification needs. But deployment notebook contains full automation to get started with this solution really easily. 

**Implementation process**
- Create Genie spaces that you want to use.
- Clone the repo to your Databricks workspace (instructions https://docs.databricks.com/aws/en/repos/git-operations-with-repos).
- Populate the deployment notebook and follow the instructions (e.g., creating connections and deploying apps). The DevOps tool is optional — you can leave the DevOps variables empty ("") and it won't be created.
- Grant the required permissions to the Apps Service Principal (for Unity Catalog connections, used Genie spaces, and model endpoints)

## Usage

The deployment notebook automatically deploys Databricks apps, and after granting permissions, you can start using the app. From the Databricks apps logs, you can see more detailed metadata on how it is performing. Remember to optimize the separate Genie spaces to improve results, adjust the executor agent system prompt to fit your needs, and do the same for tool descriptions. And once you are ready, remember to stop Apps. 

## Limitations
- Keep in mind that this is for demo purposes only. Although it gives a clear understanding of how you can use it as a template for further development.
- For example, DevOps repo modification is limited to creating one epic with body changes only. However, you can see from the code how easy it is to start implementing more sophisticated features like commenting based on company documentation, adding and closing bug tickets automatically, etc.
    - RAG: Create an automated data processing pipeline and store it as a vector index.
    - REST API calls: Use a dynamic approach and fetch documentation using a REST API approach (requires good documentation mapping).
    - Extended prompts: Current logic. Now that LLM's context window is getting so big, RAG isn't even needed in all cases.
- Agent features are really limited, and automated quality monitoring is excluded here. Also, advanced memory and reasoning capabilities are excluded.
- The model endpoint LLM model requires tool support. Not all LLM models have been tested.
- All validation has been done in Azure Databricks.

## More information
To stay up-to-date with the latest developments: 
1) Follow Ikidata on LinkedIn: https://www.linkedin.com/company/ikidata/ 
2) Explore more on our website: https://www.ikidata.fi

![logo](https://github.com/ikidata/ikidata_public_pictures/blob/main/logos/Ikidata_aurora_small.png?raw=true)