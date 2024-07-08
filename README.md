# Benchmarking Hub 🤖

[![Open Source Love](https://firstcontributions.github.io/open-source-badges/badges/open-source-v1/open-source.svg)](https://github.com/firstcontributions/open-source-badges)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![AI](https://img.shields.io/badge/AI-enthusiast-7F52FF.svg)
![GitHub stars](https://img.shields.io/github/stars/pablosalvador10/gbb-ai-upgrade-llm?style=social)
![Issues](https://img.shields.io/github/issues/pablosalvador10/gbb-ai-upgrade-llm)
![License](https://img.shields.io/github/license/pablosalvador10/gbb-ai-upgrade-llm)

Welcome to the ultimate MaaS (LLM/SLM) benchmarking hub. This project focuses on dual intent, providing key performance metrics such as latency, throughput, and quality. It's a one-stop shop for benchmarking MaaS, helping you make smarter decisions regarding the choice of foundation model for your AI projects based on an in-depth performance analysis.
<br>

## What Makes This Project Different? 🚀

- **Light Python SDK**:Tailored for performance-centric evaluations, our SDK facilitates extensive analysis across latency, throughput, and a suite of quality metrics. Designed for bulk processing, it streamlines the assessment of multiple models simultaneously, ensuring a thorough comparison.

- **Intuitive User Interface**: Our user-centric app simplifies benchmarking processes. Engage with results for models like "GPT" and delve into intricate visualizations. These tools elucidate the dynamics between prompts and generations, latency implications, and more, offering a granular understanding of model performance.

- **Accelerated Integration**: As the landscape of LLM/SLM technologies rapidly evolves, staying ahead becomes a challenge. Our project serves as an agile launchpad for benchmarking foundational models, significantly reducing the time-to-integration for the latest advancements. Equip your enterprise with the tools to swiftly adapt and implement cutting-edge AI technologies.

- **Built from Expertise-Driven Design for Large Enterprise AI Systems**: Drawing from deep experience in building large-scale enterprise AI systems with special focuses on Azure OpenAI (AOAI) implementations. It guides you through best practices and effective troubleshooting strategies for latency, throughput, and various quality metrics to optimize performance later on.

- **BYOP (Bring Your Own Prompt) for Custom Benchmarks**: This feature enables the application of the benchmarking suite to your  data, offering valuable insights into model performance on real-world problems as opposed to theoretical scenarios. It's an essential tool for enterprises and individuals aiming to assess the effectiveness of foundational models against their specific datasets and challenges.

Check out how our v0.01 prototype is shaping up.

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.15); width: 80%; margin: auto;">
    <iframe src="https://www.loom.com/embed/9c6592b16c5b4785805ce87393601dfd?sid=bcc2e170-9295-427c-ae11-b89489f3ab6b" 
    frameborder="0" 
    webkitallowfullscreen 
    mozallowfullscreen 
    allowfullscreen 
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>
<br>

## What's Next? ✨

We're currently in full/fun development mode and adding features as we speak. Please look at the [CHANGELOG](./CHANGELOG.md) for a detailed overview. Here's what's coming next:

> [!IMPORTANT]
> For the current version, our current focus is primarily on Azure OpenAI (AOAI) deployments

- **Expanding Our MaaS Offerings**: Enhancing our Machine-as-a-Service (MaaS) portfolio, with a focus on the pay-as-you-go models available on Azure. This initiative aims to offer more flexible access to cutting-edge AI capabilities, such as SLMs, emphasizing performance optimization and scalability.

- **Adoption of PaaS Models**: We're working om expanding our scope to include Platform-as-a-Service (PaaS) models, with a special focus on SLMs (Specialized Language Models) like Phi-3 deployed on Azure infrastructure.

- **Embedded Model Benchmarking**: Enhancing our benchmarking suite to include embedding models, focusing on performance metrics in constrained environments.

- **State-of-the-Art Quality Metrics Suite**: We're developing a comprehensive suite of quality metrics, allowing you to rigorously evaluate changes and performance with your own data between different model "upgrades/versions" or models. This suite is intended to provide a deeper, more nuanced understanding of model capabilities and limitations.
<br>

## How to Get Started 🔍

First things first, let's get your development environment set up:

1. **Create a Conda Environment**: First, you need to create a Conda environment using the `environment.yaml` file provided in the repository. Open your terminal and run:

   ```bash
   conda env create -f environment.yaml
   ```
To activate your Conda environment and run your Streamlit application, follow these steps:

2. **Activate Conda Environment**: After creating your environment, activate it using the command:
   ```bash
   conda activate <your_env_name>
   ```
### Running the App 💻

To deploy your Streamlit application locally, follow these steps:

- Ensure your development environment is set up and your Conda environment is activated.

#### Step 1: Launch the Application: To start your Streamlit app, navigate to the `src/app` directory in your terminal and execute:
    
    ```bash
    streamlit run src/app/Home.py
    ```
The application should launch directly in your browser as a local host. Enjoy, but please...

**Provide Feedback - Your Insights Fuel Our Growth**

Encountered an issue or have suggestions for improvements? We want to hear from you! Please [submit an issue]() on our GitHub repository. Your feedback is vital to our development process.

### Running the SDK 💡

- Ensure your development environment is set up and your Conda environment is activated.

#### Step 1: Define Test Parameters

First, you need to define the parameters for your test:

- **Deployment Names**: An array of deployment names you wish to test.
- **Token Counts**: A list of maximum token counts to test against each deployment.

```python
deployment_names = ["YourModelName1", "YourModelName2"]
max_tokens_list = [100, 500, 700, 800, 900, 1000]
```

#### Step 2: Initialize the Testing Class
Depending on whether your test is for streaming or non-streaming deployments, initialize the appropriate class. Here's how to initialize for non-streaming:


```python
from src.performance.aoaihelpers.latencytest import AzureOpenAIBenchmarkNonStreaming

client_non_streaming = AzureOpenAIBenchmarkNonStreaming(
    api_key="YOUR_AZURE_OPENAI_API_KEY",
    azure_endpoint="YOUR_AZURE_OPENAI_ENDPOINT",
    api_version="YOUR_AZURE_OPENAI_API_VERSION"
)
```

#### Step 3: 🛠️ Execute the Tests
Run the run_latency_benchmark_bulk method with your defined parameters:

```python
await client_non_streaming.run_latency_benchmark_bulk(
    deployment_names, max_tokens_list, iterations=1, context_tokens=1000, multiregion=False
)
```

For detailed instructions on running throughput benchmarks, refer to [HOWTO-Throughput.md](notebooks/benchmarks/HOWTO-Throughput.md). For guidance on latency benchmarks, see [HOWTO-Latency.md](notebooks/benchmarks/HOWTO-Latency.md).

<!-- ## 🎒 Show and tell
> [!TIP]
> Install the [VS Code Reveal extension](https://marketplace.visualstudio.com/items?itemName=evilz.vscode-reveal), open LLM-BENCHMARK-EVALUATOR.md and click on 'slides' at the bottom to present the LLM Benchmark Evaluator without leaving VS Code.
> Or just open the [LLM-BENCHMARK-EVALUATOR.pptx](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2FYourGitHubUsername%2FLLM-Benchmark-Evaluator%2Fmain%2FLLM-BENCHMARK-EVALUATOR.pptx&wdOrigin=BROWSELINK) for a plain old PowerPoint experience. -->

### Disclaimer
> [!IMPORTANT]
> This software is provided for demonstration purposes only. It is not intended to be relied upon for any purpose. The creators of this software make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the software or the information, products, services, or related graphics contained in the software for any purpose. Any reliance you place on such information is therefore strictly at your own risk.