def get_cosmos_db_prompt(prompt):
    return f'''
    # Cosmos DB Query Translator

    Your goal is to understand the essence of each user query, identify the relevant database fields, and construct an accurate Cosmos DB query that retrieves the requested information.

    ## Task

    Translate the user's natural language queries into Cosmos DB queries by following these steps:

    1. **Identify Key Information**: Extract the essential components and conditions from the natural language query.
    2. **Map to Database Fields**: Align the identified components with the corresponding fields in the Cosmos DB schema.
    3. **Construct the Query**: Formulate a Cosmos DB query that captures the user's request accurately.

    ## Examples

    **Example 1**

    - **User Query**: "Find all project requests from partners expected to start in the first quarter of 2024."
    SELECT * FROM c WHERE c.Partner IS NOT NULL AND c.ExpectedStartDate >= '2024-01-01' AND c.ExpectedStartDate <= '2024-03-31'

    **Example 2**

    - **User Query**: "List projects requiring more than 100 hours of work but not yet started."
    SELECT * FROM c WHERE c.ProjectedWorkHours > 100 AND c.Status = 'Not started'

    **Example 3**

    - **User Query**: "Show me projects assigned to John Doe that involve Azure AI Services."
    SELECT * FROM c WHERE c.AssignedTo = 'John Doe' AND c.AzureAIServices != ''

    **Example 4**

    - **User Query**: "Find all approved projects that have attachments."
    SELECT * FROM c WHERE c.Approved = True AND c.Attachment IS NOT NULL

    **Example 5**

    - **User Query**: "Show me projects with a projected ACR greater than 50000."
    SELECT * FROM c WHERE c.ProjectedACR > 50000

    **Example 6**

    - **User Query**: "List all projects in the 'In progress' status assigned to Jane Doe."
    SELECT * FROM c WHERE c.Status = 'In progress' AND c.AssignedTo = 'Jane Doe'

    ## Return Query

    - **User Query**: "{prompt}"

    Please generate the corresponding Cosmos DB query based on the user's request. 

    Remember, your task is to construct the query, not to execute it or return the result. 

    Regardless of the user's request, always include `c.RequestId` in your SELECT statement. For example, if the user asks for a project status, your output should be: 
    SELECT c.RequestId, c.Status FROM c WHERE c.Status = 'In progress' AND c.AssignedTo = 'Jane Doe'
    
    return only the query, no verbosity.
    '''

def get_chat_cosmos_db_prompt(prompt, json_response):
    return f'''
    # Cosmos DB Response Processor

    ## Introduction

    You are tasked with interpreting a JSON response from a Cosmos DB query related to project requests. The JSON structure reflects the schema of project requests, with fields such as `RequestTitle`, `Requester`, `ExpectedStartDate`, and others relevant to the project details.

    ## Task

    - **Input**: A JSON response from Cosmos DB and a user question related to this data.
    - **Action**: Parse the JSON to understand the data structure and content. Use this information to accurately answer the user's question. If there's not enough information to answer the question, return a message saying "We are not able to assist you at this moment. Please try with another inquiry."
    - **Output**: A clear and concise answer to the question, directly based on the data provided, or a message indicating insufficient information.

    ## JSON Response

    ```json
    "{json_response}"
    ```
    ## User Question
    "{prompt}"

    ## Instructions
    - Parse JSON: Carefully read and interpret the JSON data to understand the details of the project requests it contains.
    - Answer the Question: Based on your understanding of the JSON data, provide an answer to the user's question. Ensure that your answer is directly supported by the data in the JSON response. If there's not enough information to answer the question, return a message saying "We are not able to assist you at this moment. Please try with another inquiry.
    '''