## Table of Contents

- [Setting Up Azure AI Services](#setting-up-azure-ai-services)
- [Configuration Environment Variables](#configuration-environment-variables)
- [Notebooks Setup](#notebooks-setup)
  - [Setting Up Conda Environment and Configuring VSCode for Jupyter Notebooks](#setting-up-conda-environment-and-configuring-vscode-for-jupyter-notebooks)

## Setting Up Azure AI Services

- **Azure Open AI Services**:  An Azure OpenAI Service resource with a model deployed using a provisioned deployment type (either `Provisioned` or `Provisioned-Managed`). For more information, see the [resource deployment guide](https://learn.microsoft.com/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).


## Configuration Environment Variables

Before running this notebook, you must configure certain environment variables. We will now use environment variables to store our configuration. This is a more secure practice as it prevents sensitive data from being accidentally committed and pushed to version control systems.

Create a `.env` file in your project root (use the provided `.env.sample` as a template) and add the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY="your-openai-api-key"  # Replace with your OpenAI API Key
AZURE_OPENAI_ENDPOINT="https://your-openai-endpoint.azure.net"  # Replace with your Azure OpenAI Endpoint
```

Please replace `your-openai-api-key` and `https://your-openai-endpoint.azure.net` with your actual OpenAI API Key and Azure OpenAI Endpoint respectively.

`OPENAI_API_KEY`: This is your OpenAI API Key. You can find it in your OpenAI account settings.
`AZURE_OPENAI_ENDPOINT`: This is the URL of your Azure OpenAI Endpoint. You can find it in the "Overview" section of your OpenAI resource in the Azure portal.

> 📌 **Note**
> Remember not to commit the .env file to your version control system. Add it to your .gitignore file to prevent it from being tracked.

## Create Conda Environment from the Repository

> Instructions for Windows users:

1. **Create the Conda Environment**:
   - In your terminal or command line, navigate to the repository directory.
   - Execute the following command to create the Conda environment using the `environment.yaml` file:
     ```bash
     conda env create -f environment.yaml
     ```
   - This command creates a Conda environment as defined in `environment.yaml`.

2. **Activating the Environment**:
   - After creation, activate the new Conda environment by using:
     ```bash
     conda activate ptu-benchmarking 
     ```

> Instructions for Linux users (or Windows users with WSL or other linux setup):

1. **Use `make` to Create the Conda Environment**:
   - In your terminal or command line, navigate to the repository directory and look at the Makefile.
   - Execute the `make` command specified below to create the Conda environment using the `environment.yaml` file:
     ```bash
     make create_conda_env
     ```

2. **Activating the Environment**:
   - After creation, activate the new Conda environment by using:
     ```bash
     conda activate ptu-benchmarking 
     ```

## Configure VSCode for Jupyter Notebooks

1. **Install Required Extensions**:
   - Download and install the `Python` and `Jupyter` extensions for VSCode. These extensions provide support for running and editing Jupyter Notebooks within VSCode.

2. **Attach Kernel to VSCode**:
   - After creating the Conda environment, it should be available in the kernel selection dropdown. This dropdown is located in the top-right corner of the VSCode interface.
   - Select your newly created environment (`ptu-benchmarking`) from the dropdown. This sets it as the kernel for running your Jupyter Notebooks.

3. **Run the Notebook**:
   - Once the kernel is attached, you can run the notebook by clicking on the "Run All" button in the top menu, or by running each cell individually.

4. **Voila! Ready to Go**:
   - Now that your environment is set up and your kernel is attached, you're ready to go! Please visit the notebooks in the repository to start exploring.

> **Note:** By following these steps, you'll establish a dedicated Conda environment for your project and configure VSCode to run Jupyter Notebooks efficiently. This environment will include all the necessary dependencies specified in your `environment.yaml` file. If you wish to add more packages or change versions, please use `pip install` in a notebook cell or in the terminal after activating the environment, and then restart the kernel. The changes should be automatically applied after the session restarts.
