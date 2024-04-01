from typing import Dict, List


from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader

from llama_index.core import Settings # llama-index v0.10 and up, deprecating ServiceContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from langchain.embeddings import OpenAIEmbeddings
from llama_index.core.tools import FunctionTool
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from trulens_eval import TruLlama, Feedback, Select, OpenAI as fOpenAI

llm = OpenAI(model="gpt-4-0125-preview")
import streamlit as st




Settings.llm = llm
# Settings.embed_model =  OpenAIEmbeddings()
# sentence_node_parser = SentenceWindowNodeParser.from_defaults(
#     window_size=2,
#     window_metadata_key="window",
#     original_text_metadata_key="original_text"
# )
# Settings.node_parser = sentence_node_parser


TRULENS_stock_feedback_functions_docs_url = "https://www.trulens.org/trulens_eval/evaluation/feedback_implementations/stock/"

# use simple web page reader from llamahub
loader = SimpleWebPageReader()
docs = loader.load_data(urls=[TRULENS_stock_feedback_functions_docs_url])
# chroma_client = chromadb.PersistentClient()
# chroma_collection = chroma_client.get_or_create_collection("hackathon")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# storage_context = StorageContext.from_defaults(vector_store=vector_store)


trulens_doc_index = VectorStoreIndex.from_documents(docs)
trulens_doc_query_engine = trulens_doc_index.as_query_engine(verbose=True)

ff_prompts_reader = SimpleDirectoryReader(input_files=["/home/ubuntu/hackathon/trulens/trulens_eval/trulens_eval/feedback/v2/feedback.py", "/home/ubuntu/hackathon/trulens/trulens_eval/trulens_eval/feedback/prompts.py"])

# TODO: figure out how to control order of tool executions

def extract_eval_criteria() -> str:
    """Extract evaluation criteria based on conversation history so far."""
    prompt_str = f"""
    Based on the conversation history til this point between the user and assistant that typically involves question answering over documents, understand the interaction between user and assistant: {str(st.session_state.agent_messages)}\n\n 
     try to extract meaningful evaluation criteria from the messages. The evaluation criteria are such that they can be used to evaluate the quality of the responses and how well they
    align with the user's requirement toward the question-answering system. In the conversation history, you should pay attention to user's frustration (typically by repeating the same questions with inpatient tone),
    or paraphrasing the same question in different ways, or asking for clarification on the credibility of the responses, etc. These patterns are indicative of the quality of the responses and can be used to extract evaluation criteria.
    The evaluation criteria should be specific and actionable, such that they can be used to instrument the LLM-based system to generate feedback on the quality of the responses.
    """

    return llm.complete(prompt_str)

extract_eval_criteria_tool = FunctionTool.from_defaults(
    fn=extract_eval_criteria,
    name="eval_criteria_extractor",
    description="Extract evaluation criteria from conversation history . This function should be called when user asks the agent to present with evaluation criteria.",
)


def feedback_functions_retriever(criteria_prompt: str):
    """Fetch definitions of feedback functions in Python based on the criteria. The criteria are often listed in the previous response from the assistant."""
    prompt_str = """
    Given the provided evaluation criteria: {criteria_prompt}, retrieve as accurately as possible a list of feedback functions that match or suit the provided criteria based on the feedback functions' description in the doc
    - make sure to return both the description of the feedback function as well as the Python code example (often below the details class ="usage" or "class="usage-on-rag-contexts" section) from the source doc and ONLY from the source TruLens doc.
        Note context_relevance/context_relevance_with_cot_reasons is equivalent to qs_relevance/qs_relevance_with_cot_reasons, but these two are distinct from relevance.

    If there's no suitable feedback functions found in the doc, just say there is none found and DO NOT make up one that is not in the source doc. 
    """.format(criteria_prompt=criteria_prompt)

    return trulens_doc_query_engine.query(prompt_str)


feedback_functions_retriever_tool = FunctionTool.from_defaults(
    fn=feedback_functions_retriever,
    name="feedback_functions_retriever",
    description="Retrieve feedback functions based on the evaluation criteria provided by the assistant and is approved / confirmed by the user.",
)

REFERENCE_PROMPT = """
You are tasked with evaluating summarization quality. Please follow the instructions below.

INSTRUCTIONS:

1. Identify the key points in the provided source text and assign them high or low importance level.

2. Assess how well the summary captures these key points.

Are the key points from the source text comprehensively included in the summary? More important key points matter more in the evaluation.

Scoring criteria:
0 - Capturing no key points with high importance level
5 - Capturing 70 percent of key points with high importance level
10 - Capturing all key points of high importance level

Answer using the entire template below.

TEMPLATE:
Score: <The score from 0 (capturing none of the important key points) to 10 (captures all key points of high importance).>
Criteria: <Mention key points from the source text that should be included in the summary>
Supporting Evidence: <Which key points are present and which key points are absent in the summary.>
"""
def custom_feedback_function_prompt_builder(criteria_prompt: str): # TODO: note this only works with single response text
    """If no suitable feedback functions are found in the source doc, help the user to build a custom feedback function prompt."""
    
    return llm.complete(f"Based on the criteria: {criteria_prompt}, build a prompt that is apt for the criteria at hand with similar style and utility as {REFERENCE_PROMPT}")
    # return str.format("Based on the criteria: {} \n\n, as a meticulous score, please rate the following text on a scale from 0 to 10, where 0 is not at all meeting the criteria and 10 is almost perfectly matching the provided criteria: \n\n{}", criteria_prompt, response)     




custom_feedback_function_prompt_helper_tool = FunctionTool.from_defaults(
    fn=custom_feedback_function_prompt_builder,
    name="custom_feedback_function_prompt_builder",
    description="Build a custom feedback function prompt based on the evaluation criteria provided by the assistant and is approved / confirmed by the user.",
)


def instrument_custom_feedback_function(feedback_prompt: str, criteria_prompt: str):
    """Instrument the feedback function based on the feedback prompt. The feedback prompt is generated in previous chat interaction"""
    def named_lambda(name, f):
        """Wrap a lambda function with a name for better identifiability."""
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        wrapped.__name__ = name
        return wrapped
    
    ff_name = "_".join(criteria_prompt.split()[:2]).lower()

    # custom_feedback_lambda = named_lambda(f'custom_feedback_function_{ff_name}',
    #                                   lambda response: fOpenAI().generate_score(
    #                                       system_prompt=feedback_prompt + f"\n\n Use the previous rating rubric to rate: {response}"
    #                                   ))
    custom_feedback_lambda = lambda response: fOpenAI().generate_score(
                                          system_prompt=feedback_prompt + f"\n\n Use the previous rating rubric to rate: {response}")
    
    if "agent" in st.session_state.keys():
        f_custom_function = Feedback(custom_feedback_lambda).on_default()

        if "tru_agent" not in st.session_state.keys():
            st.error("TruLens instrumentation not initialized.")
        else:
            prev_version = int(st.session_state.tru_agent.app_id.split()[-1])
            st.session_state.tru_agent = TruLlama(st.session_state.agent,
            app_id=f'Eval Pilot {str(prev_version+1)}',
            tags = "agent prototype",
            feedbacks = [
                f_custom_function
            ]
        )


    
custom_feedback_function_instrument_tool = FunctionTool.from_defaults(
    fn=instrument_custom_feedback_function,
    name="instrument_feedback_function",
    description="Instrument the new custom feedback function",
)
