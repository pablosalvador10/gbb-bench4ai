from src.quality.eval import Eval
from openai import AzureOpenAI
import pandas as pd
import os
import asyncio


class MMLU(Eval):
    '''
    This is a class implementing the MMLU benchmark evaluation.
    Inputs:
        deployment_config:
            key: AOAI API key
            endpoint: AOAI endpoint
            model: AOAI deployment name
        categories: list of subjects to evaluate (optional)
            Must be one of the following: [STEM, Medical, Business, Social Sciences, Humanities, Other]
        sample_size: fraction of data to sample for evaluation (optional)
        log_level: logging level (optional)

    Inhereted Methods:
        _score: Compare the generated answer to the correct answer
        _load_data: Load the dataset from the MMLU benchmark into memory
        
    Private Methods:
        __transform_data: Transform the dataset into a format that can be used by the Azure OpenAI API
        __call_aoai: Call the Azure OpenAI API to generate an answer

    Public Method:
        test: Run the MMLU evaluation and output a dataframe

    '''
    def __init__(self, deployment_config: dict, sample_size: float = 1.0, log_level: str = "INFO", categories: list = None):
        super().__init__(deployment_config, sample_size, log_level=log_level)
        self.categories = categories
        
        global subject2category
        subject2category = {
            "abstract_algebra": "stem",
            "anatomy": "medical",
            "astronomy": "stem",
            "business_ethics": "business",
            "clinical_knowledge": "medical",
            "college_biology": "medical",
            "college_chemistry": "stem",
            "college_computer_science": "stem",
            "college_mathematics": "stem",
            "college_medicine": "medical",
            "college_physics": "stem",
            "computer_security": "stem",
            "conceptual_physics": "stem",
            "econometrics": "social_sciences",
            "electrical_engineering": "stem",
            "elementary_mathematics": "stem",
            "formal_logic": "humanities",
            "global_facts": "humanities",
            "high_school_biology": "stem",
            "high_school_chemistry": "stem",
            "high_school_computer_science": "stem",
            "high_school_european_history": "humanities",
            "high_school_geography": "social_sciences",
            "high_school_government_and_politics": "social_sciences",
            "high_school_macroeconomics": "social_sciences",
            "high_school_mathematics": "stem",
            "high_school_microeconomics": "social_sciences",
            "high_school_physics": "stem",
            "high_school_psychology": "social_sciences",
            "high_school_statistics": "stem",
            "high_school_us_history": "humanities",
            "high_school_world_history": "humanities",
            "human_aging": "medical",
            "human_sexuality": "social_sciences",
            "international_law": "humanities",
            "jurisprudence": "humanities",
            "logical_fallacies": "humanities",
            "machine_learning": "stem",
            "management": "business",
            "marketing": "business",
            "medical_genetics": "medical",
            "miscellaneous": "other",
            "moral_disputes": "humanities",
            "moral_scenarios": "humanities",
            "nutrition": "medical",
            "philosophy": "humanities",
            "prehistory": "humanities",
            "professional_accounting": "business",
            "professional_law": "humanities",
            "professional_medicine": "medical",
            "professional_psychology": "social_sciences",
            "public_relations": "social_sciences",
            "security_studies": "social_sciences",
            "sociology": "social_sciences",
            "us_foreign_policy": "social_sciences",
            "virology": "medical",
            "world_religions": "humanities",
        }


    def __transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter to specified categories of subjects.
        # Broad categories defined in subject2category dict (global) - STEM, Medical, Business, Social Sciences, Humanities, Other
        if self.categories:
            self.categories = [c.lower() for c in self.categories]
            self.categories = [c.replace(" ", "_") for c in self.categories]
            df["category"] = df["subject"].map(subject2category)
            df = df[df["category"].isin(self.categories)]
            self.logger.info(f"Trimmed dataset to specified categories: {df.value_counts('category')}")

        # Subset data based on subject
        self.logger.info(f"Sampling data to {self.sample_size*100}% of each subject")
        df=df.groupby('subject',as_index = False, group_keys=False).apply(lambda s: s.sample(frac=self.sample_size,replace=False)).reset_index()

        self.logger.info(f"Data loaded. {df.shape[0]} rows.")
        return df

     
    def __call_aoai(self, row: list, version: str = "2024-02-01") -> dict:

        client = AzureOpenAI(
                            azure_endpoint = self._base, 
                            api_key=self._key,  
                            api_version=version
                            )
        
        sys_message = "Complete the given problem to the best of your ability. \
                        Accuracy is very important. \
                        Choices are a list of quoted strings with a starting index of 0 \
                        Select ONLY ONE answer from the choices. \
                        Return ONLY the index of the correct answer in the choices list. Your answer must be a single ineteger."

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": f"Question: {row['question']}.  Choices: {row['choices']}. Answer:"},
            ]
        )
        output = {"generated": response.choices[0].message.content, "correct": row["answer"] ,"subject": row["subject"]}
        output["score"] = self._score(output["generated"], output["correct"])

        return output


    async def test(self) -> pd.DataFrame:
        test_data = self._load_data(dataset="cais/mmlu", subset="all", split="test")
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

        self.logger.info("Evaluation complete")

        self.logger.info("Aggregating Results")
        results = pd.DataFrame(output_list).groupby("subject").agg({"score": "mean"}).reset_index()
        results_dict = {'deployment': self.model,'test': 'MMLU','overall_score': results.loc[:, 'score'].mean()}

        return pd.DataFrame([results_dict])


if __name__ == "__main__":

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    deploy_dict = {
        "model": os.getenv("AOAI_MODEL"),
        "endpoint": os.getenv("AOAI_ENDPOINT"),
        "key": os.getenv("AOAI_KEY"),
    }

    mmlu_eval = MMLU(deploy_dict, sample_size=0.01, categories = ['Business'], log_level="INFO")
    result = asyncio.run(mmlu_eval.test())

    print(f"Results: \n{result}")