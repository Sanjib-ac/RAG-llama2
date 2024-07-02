"""
pip install pypdf
pip install -q transformers einops accelerate langchain bitsandbytes
pip install install sentence_transformers  #Embedding
pip install llama-index-llms-huggingface
pip install llama_index
pip install llama-index-embeddings-langchain
pip install accelerate
pip install sentence-transformers

#for image
pip install pdfminer.six
pip install pillow-heif
pip install opencv-contrib-python


# pip install -i https://pypi.org/simple/ bitsandbytes
# pip install transformers==4.30
"""
import time
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader  # ServiceContext, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt

# import llama_index.core.settings as lis

# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor

from langchain_community.document_loaders.image import UnstructuredImageLoader
torch.cuda.current_device()


loader = UnstructuredImageLoader("./data/images/radio.jpg").load()
print(loader)


#
# documents = SimpleDirectoryReader("./data/").load_data()
# print(documents)
# for i, j in enumerate(documents):
#     print(f'{i}: {j}\n')
#     time.sleep(1)
'''
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
# Default format supportable by LLama2
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
print(query_wrapper_prompt)

# from huggingface_hub import login
#
# login(token="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")

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=128,
    generate_kwargs={"temperature": 0.1},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="cuda",
    model_kwargs={"torch_dtype": torch.float16}
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)  # , "load_in_8bit_fp32_cpu_offload": True

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext

# from llama_index.embeddings import LangchainEmbedding

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                    model_kwargs={'device': 'cuda'},
                                    encode_kwargs={'normalize_embeddings': False},
                                    cache_folder='./data',
                                    # multi_process=True,
                                    show_progress=True,
                                    )

service_context = ServiceContext.from_defaults(
    chunk_size=512,
    llm=llm,
    embed_model=embed_model
)

# service_context = lis.Settings(
#     chunk_size=1024,
#     llm=llm,
#     embed_model=embed_model
# )

# print(f'Service Context: {service_context}')
index = VectorStoreIndex.from_documents(documents, service_context=service_context,
                                        show_progress=True,
                                        model_kwargs={'device': 'cuda'})

# print(f'Index: {index}')
query_engine = index.as_query_engine(device_map='CUDA')
# print(f'Query:{query_engine}')
while True:
    qry = str(input("Any query from the database?"))
    # response=query_engine.query("what is attention is all you need?")
    response = query_engine.query(qry)
    print(f'Response: {response}')

'''