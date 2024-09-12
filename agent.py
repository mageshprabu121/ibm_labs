from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_ibm import WatsonxLLM
import os

from dotenv import load_dotenv


load_dotenv()

os.environ["WATSONX_APIKEY"] = os.getenv("WATSONX_APIKEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")


app = FastAPI()


class QueryRequest(BaseModel):
    topic: str


parameters = {"decoding_method": "greedy", "max_new_tokens": 1500}


llm = WatsonxLLM(
    model_id="meta-llama/llama-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters,
    project_id="53812b92-5f8a-4ac0-82a9-ad49dd512222",
)

function_calling_llm = WatsonxLLM(
    model_id="ibm-mistralai/merlinite-7b",
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters,
    project_id="53812b92-5f8a-4ac0-82a9-ad49dd512222",
)


search = SerperDevTool()

@app.post("/generate-response")
async def generate_response(request: QueryRequest):
    s = request.topic

    
    researcher = Agent(
        llm=llm,
        function_calling_llm=function_calling_llm,
        role="Interview panel",
        goal=f"Find promising information about {s} that can be used to test the candidate",
        backstory=f"You are an interviewer, interviewing a candidate on the topic {s} based on the search results only",
        allow_delegation=False,
        tools=[search],
        verbose=1,
    )

    
    task1 = Task(
        description=f"Search the internet and find 5 questions directly related to {s} along with expected elaborate answers",
        expected_output=f"Generate 5 bullet points each with a question on {s} and accurate elaborate answers for the question",
        output_file="task1output.txt",
        agent=researcher,
    )

   
    crew = Crew(agents=[researcher], tasks=[task1], verbose=1)
    
    
    search_query = f"{s} interview questions"
    search_results = search.run({"search_query": search_query})

    
    result = crew.kickoff()

   
    return {"result": result, "search_results": search_results}


