from src.quality.eval import Eval
from openai import AzureOpenAI
import pandas as pd
import os
import asyncio


class PubMedQA(Eval):
    '''
    This is a class implementing the PubMedQA benchmark evaluation.
    Inputs:
        deployment_config:
            key: AOAI API key
            endpoint: AOAI endpoint
            model: AOAI deployment name
        sample_size: fraction of data to sample for evaluation (optional)
        log_level: logging level (optional)

    Inhereted Methods:
        _score: Compare the generated answer to the correct answer
        _load_data: Load the dataset from the benchmark into memory
        
    Private Methods:
        __transform_data: Transform the dataset into a format that can be used by the Azure OpenAI API
        __call_aoai: Call the Azure OpenAI API to generate an answer

    Public Method:
        test: Run the PubMedQA evaluation and output a dataframe

    '''
    def __init__(self, deployment_config: dict, sample_size: float = 1.0, log_level: str = "INFO"):
        super().__init__(deployment_config, sample_size, log_level=log_level)
        

    def __transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Take subset of data
        self.logger.info(f"Sampling data to {self.sample_size*100}% ")
        df=df.sample(frac=self.sample_size,replace=False).reset_index()
        return df

     
    def __call_aoai(self, row: list, version: str = "2024-02-01") -> dict:

        client = AzureOpenAI(
                        azure_endpoint = self._base, 
                        api_key=self._key,  
                        api_version=version
                        )
    
        sys_message = "Complete the given problem to the best of your ability. \
                    Accuracy is very important. \
                    Given a context, answer the research question with either a yes, no, or maybe \
                    THe context will be a list of strings. Each string is some relevant information to inform your decision\
                    Select ONLY ONE answer from the following choices [yes, no, maybe]. \
                    Your answer must be a single word, all lowercase, do not use quotations"

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": f"Question: {row['question']}.  Context: {row['context.contexts']}. Answer:"},
            ]
        )
        output = {"generated": response.choices[0].message.content, "correct": row["final_decision"]}
        output["score"] = self._score(output["generated"], output["correct"])

        return output


    async def test(self) -> pd.DataFrame:
        test_data = self._load_data(dataset="qiaojin/PubMedQA", subset="pqa_labeled", split="train", flatten=True)
        test_data = self.__transform_data(test_data)

        output_list = []
        self.logger.info("Starting evaluation")
        for index, row in test_data.iterrows():
            self.logger.info(f"Evaluating row {index} of {test_data.shape[0]}")
            try: 
                output = self.__call_aoai(row)
                output_list.append(output)
            except Exception as e:
                self.logger.warning(f"Skipping...error in row {index}: {e}")

        self.logger.info("Evaluation complete.")
        results = pd.DataFrame(output_list).reset_index()
        results_dict = {'deployment': self.model, 'test': 'MedPub QA' ,'overall_score': results.loc[:, 'score'].mean()}

        return pd.DataFrame([results_dict])


if __name__ == "__main__":

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    deploy_dict = {
        "model": os.getenv("AOAI_MODEL"),
        "endpoint": os.getenv("AOAI_ENDPOINT"),
        "key": os.getenv("AOAI_KEY"),
    }

    pubmed_eval = PubMedQA(deploy_dict, sample_size=0.01, log_level="INFO")
    result = asyncio.run(pubmed_eval.test())

    print(f"Results: \n{result}")