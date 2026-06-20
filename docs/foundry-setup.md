# Azure AI Foundry setup

Notebook 3 (`03_prompt_engineering.ipynb`) and the demo app use a hosted
**GPT-4o** chat model. Follow these steps once to make it available.

## 1. Create a project and deploy the model

1. Go to the [Azure AI Foundry portal](https://ai.azure.com) and sign in.
2. Create (or open) a **project**.
3. In the left menu open **Models + endpoints → Deploy model → Deploy base model**.
4. Choose **gpt-4o** and deploy it. Note the **deployment name** you give it
   (we use `gpt-4o`).

## 2. Get the endpoint and key

1. Open your deployment (or the project **Overview**).
2. Copy the **Target URI / endpoint**. It looks like
   `https://<your-resource>.services.ai.azure.com/` (or `...openai.azure.com/`).
3. Copy one of the **API keys**.

## 3. Configure the project

1. Copy `.env.example` to `.env` in the project root.
2. Fill in the values:

   ```dotenv
   AZURE_OPENAI_ENDPOINT=https://<your-resource>.services.ai.azure.com/
   AZURE_OPENAI_API_KEY=<your-key>
   AZURE_OPENAI_API_VERSION=2024-10-21
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
   ```

   `AZURE_OPENAI_CHAT_DEPLOYMENT` must match the **deployment name** from step 1,
   not the underlying model id.

3. `.env` is git-ignored — never commit it. If a key is ever exposed, regenerate
   it in the portal.

## 4. Smoke test

Run the first cell of `notebooks/03_prompt_engineering.ipynb`. It loads the
`.env` values and sends one request; you should see the model reply `ready`.

## Cost note

Notebook 3 makes one short request per article per prompting strategy. Keep
`SAMPLE_SIZE` small (e.g. 40) while testing to control cost. `gpt-4o-mini` is a
cheaper alternative if you only need a quick run — just deploy it and set
`AZURE_OPENAI_CHAT_DEPLOYMENT` to its deployment name.
