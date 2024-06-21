import logging
import os

import pandas as pd
from datasets import load_dataset
from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO)
global logger
logger = logging.getLogger(__name__)

global subect2category
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


def load_data(sample_size=1, categories=None):
    # Download dataset
    logger.info("Loading MMLU data")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df = dataset.to_pandas()

    # Filter to specified categories of subjects.
    # Broad categories defined in subject2category dict (global) - STEM, Medical, Business, Social Sciences, Humanities, Other
    # TODO
    if categories:
        categories = [c.lower() for c in categories]
        df["category"] = df["subject"].map(subject2category)
        df = df[df["category"].isin(categories)]
        logger.info(
            f"Trimmed dataset to specified categories: {df.value_counts('category')}"
        )

    # Subset data based on subject
    logger.info(f"Sampling data to {sample_size*100}% of each subject")
    df = (
        df.groupby("subject", as_index=False, group_keys=False)
        .apply(lambda s: s.sample(frac=sample_size, replace=False))
        .reset_index()
    )

    logger.info(f"Data loaded. {df.shape[0]} rows.")
    return df


def score(generated, correct):
    try:
        if int(generated) == int(correct):
            return 1
        else:
            return 0

    except TypeError as t:
        logging.warm(f"TypeError: {t}")
        return 0
    except ValueError as v:
        logging.warn(f"ValueError: {v}")
        return 0
    except Exception as e:
        logging.warn(f"Exception: {e}")
        return 0


def call_aoai(row, endpoint, key, model, version="2024-02-01"):
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=version)

    sys_message = "Complete the given problem to the best of your ability. \
                Accuracy is very important. \
                Choices are a list of quoted strings with a starting index of 0 \
                Select ONLY ONE answer from the choices. \
                Return ONLY the index of the correct answer in the choices list. Your answer must be a single ineteger."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_message},
            {
                "role": "user",
                "content": f"Question: {row['question']}.  Choices: {row['choices']}. Answer:",
            },
        ],
    )
    output = {
        "generated": response.choices[0].message.content,
        "correct": row["answer"],
        "subject": row["subject"],
    }
    output["score"] = score(output["generated"], output["correct"])

    return output


def test(deploy_dict, sample_size=1, categories=None):
    test_data = load_data(sample_size, categories=categories)

    output_list = []
    logger.info("Starting MLU evaluation")
    for index, row in test_data.iterrows():
        logger.info(f"Evaluating row {index} of {test_data.shape[0]}")
        try:
            output = call_aoai(
                row, deploy_dict["endpoint"], deploy_dict["key"], deploy_dict["model"]
            )
            output_list.append(output)
        except Exception as e:
            logger.warn(f"Skipping...error in row {index}: {e}")

    logger.info("Evaluation complete")

    logger.info("Aggregating Results")
    result_df = (
        pd.DataFrame(output_list)
        .groupby("subject")
        .agg({"score": "mean"})
        .reset_index()
    )

    return result_df


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    deploy_dict = {
        "model": os.getenv("AOAI_MODEL"),
        "endpoint": os.getenv("AOAI_ENDPOINT"),
        "key": os.getenv("AOAI_KEY"),
    }

    result_df = test(deploy_dict, sample_size=0.01, categories=["other", "business"])
    print(f"Results: \n{result_df}")
    print(f"\nOverall Score: {result_df.loc[:, 'score'].mean()}")

