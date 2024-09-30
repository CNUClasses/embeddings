# pip install openai==1.43.0 instructor==1.4.0 python-dotenv==1.0.1
import instructor
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List
from openai import Client


load_dotenv()
openai_client = Client(api_key=os.getenv("OPENAI_API_KEY"))
client = instructor.from_openai(openai_client)


HN_MINING_SYSTEM_PROMPT = """# Task
You are a hard negative mining system. You will be given a `query` and a list of potential `hard_negatives`.
Your job is to evaluate each hard negative and determine if it is truly a hard negative in comparison to the input query. You will return only the hard negatives that are definitively negative, as these will be used to create a high-quality dataset for finetuning an embedding model.

# Output
Return only the list of hard negatives that meet the criteria above. Do not include any other commentary or explanation.
"""

USER_TEXT_TEMPLATE = """Query: {{query}}

Potential Hard Negatives: {{hns}}"""



class FilteredHardNegatives(BaseModel):
    hard_negatives: List[str]


def format_messages(
        query: str,
        candidate_hard_negatives: List[str]
    ) -> List[Dict]:
    """
    Create messages to send to OpenAI
    """
    user_text = USER_TEXT_TEMPLATE.replace("{{query}}", query)
    user_text = user_text.replace("{{hns}}", ", ".join(candidate_hard_negatives))
    return [
        {
            "role": "system",
            "content": HN_MINING_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_text
        }
    ]


def filter_hard_negatives(
        query: str,
        candidate_hard_negatives: List[str]
    ) -> List[str]:
    """
    Filters any potential positives 
    """
    messages = format_messages(query, candidate_hard_negatives)
    response: FilteredHardNegatives = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=2000,
        response_model=FilteredHardNegatives
    )
    return response.hard_negatives


if __name__ == "__main__":
    print(filter_hard_negatives(
        query="best fuel-efficient cars",
        candidate_hard_negatives=[
            "Top 10 Fuel-Efficient Cars of 2023",
            "Best Gas Mileage Cars for 2023",
            "Most Fuel-Efficient Sedans You Can Buy",
            "Hybrid Cars with the Best Fuel Economy",
            "Affordable Cars with Great Fuel Efficiency",
            "Best Luxury Cars of 2023",
            "Top Sports Cars for Speed Enthusiasts",
            "Most Powerful SUVs and Trucks",
            "Best Off-Road Vehicles for Adventure",
            "Top Electric Cars with Longest Range",
            "Best places to live in Austin, Texas"
        ]
    ))
