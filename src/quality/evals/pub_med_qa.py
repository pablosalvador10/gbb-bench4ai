import os
import logging
import pandas as pd
from openai import AzureOpenAI
from datasets import load_dataset


logging.basicConfig(level=logging.INFO)
global logger
logger = logging.getLogger(__name__)

def load_data(sample_size=1):
    logger.info("Loading PubMed QA data")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    dataset = dataset.flatten()
    df = dataset.to_pandas()
    logger.info(f"Sampling data to {sample_size*100}% ")
    df=df.sample(frac=sample_size,replace=False).reset_index()

    logger.info(f"Data loaded. {df.shape[0]} rows.")
    return df

def score(generated, correct):
    try:
        if generated.lower() == correct.lower():
            return 1
        else:
            return 0
        
    except TypeError as t:
        logging.warn(f"TypeError: {t}")
        return 0
    except ValueError as v:
        logging.warn(f"ValueError: {v}")
        return 0
    except Exception as e:
        logging.warn(f"Exception: {e}")
        return 0

def call_aoai(row, endpoint, key, model, version="2024-02-01"):

    client = AzureOpenAI(
                        azure_endpoint = endpoint, 
                        api_key=key,  
                        api_version=version
                        )
    
    sys_message = "Complete the given problem to the best of your ability. \
                Accuracy is very important. \
                Given a context, answer the research question with either a yes, no, or maybe \
                THe context will be a list of strings. Each string is some relevant information to inform your decision\
                Select ONLY ONE answer from the following choices [yes, no, maybe]. \
                Your answer must be a single word, all lowercase, do not use quotations"

    # parse context lists
    # TODO

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": f"Question: {row['question']}.  Context: {row['context.contexts']}. Answer:"},
        ]
    )
    output = {"generated": response.choices[0].message.content, "correct": row["final_decision"]}
    output["score"] = score(output["generated"], output["correct"])

    return output


def test(deploy_dict, sample_size=1):
    test_data = load_data(sample_size)

    output_list = []
    logger.info("Starting Pub Med QA evaluation")
    for index, row in test_data.iterrows():
        logger.info(f"Evaluating row {index} of {test_data.shape[0]}")
        try: 
            output = call_aoai(row, deploy_dict["endpoint"], deploy_dict["key"], deploy_dict["model"])
            output_list.append(output)
        except Exception as e:
            logger.warn(f"Skipping...error in row {index}: {e}")

    logger.info("Evaluation complete.")

    return pd.DataFrame(output_list).reset_index()

if __name__ == "__main__":

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    deploy_dict = {
        "model": os.getenv("AOAI_MODEL"),
        "endpoint": os.getenv("AOAI_ENDPOINT"),
        "key": os.getenv("AOAI_KEY"),
    }

    result_df = test(deploy_dict, sample_size=0.01)
    print(f"Score: {result_df.loc[:, 'score'].mean()}")

