import requests

# Notebook funcs
##################

from openai import OpenAI
from functools import reduce
import pandas as pd,numpy as np,json,re,os,httpx,pylab as pl
from json_repair import repair_json
from tqdm.notebook import tqdm

local_client=OpenAI(base_url='http://localhost:11434/v1/',api_key='12345')
groq_client=OpenAI(base_url='https://api.groq.com/openai/v1/',api_key=os.environ['GROQKEY'])
pplx_client=OpenAI(base_url='https://api.perplexity.ai',api_key=os.environ['PPLXKEY'])
gpt4o_client=OpenAI(base_url=os.environ['GPTPROXYURL'],api_key=os.environ['GPTPROXYKEY'],
                   http_client=httpx.Client(verify='/Users/wwoon/projects/hackathon/text-search-hackathon-2024/certificates/GAP-proxy-certificate.crt'))

def topic_extraction_msgs(passage):

    return [{'role':'system',
          'content':'You are a helpful research assistant'},
            {'role':'user',
             'content':f"""
             
The following is some content provided by the user:
{passage}

Your main task is to read this abstract carefully and determine the main research domain(s) from the following list. Consider the core concepts, specific algorithms, and techniques discussed in the abstract, not just general machine learning terms:

[List of domains with brief descriptions]
- Kernel Methods: Algorithms using kernel functions for pattern analysis
- Neural Networks: Computational models inspired by biological neural networks
- Deep Learning Architectures: Advanced neural network structures with multiple layers
- Probabilistic Graphical Models: Representations of probabilistic relationships among variables
- Decision Trees and Random Forests: Tree-based models for classification and regression
- Support Vector Machines: Supervised learning models for classification and regression
- Clustering Algorithms: Unsupervised learning techniques for grouping similar data points
- Evolutionary Algorithms: Optimization techniques inspired by biological evolution
- Bayesian Methods: Probabilistic approaches which model solutions as drawn from some distribution
- Ensemble Methods: Techniques combining multiple learning algorithms
- Dimensionality Reduction Techniques: Methods for reducing the number of random variables
- Hidden Markov Models: Statistical models where the system is a Markov process with hidden states
- Convolutional Neural Networks: Deep learning models particularly effective for image processing
- Recurrent Neural Networks: Neural networks designed to work with sequence data
- Generative Adversarial Networks: Deep learning models for generating new data
- Attention Mechanisms: Techniques allowing models to focus on specific parts of input data
- Transfer Learning Methods: Approaches for applying knowledge from one task to another
- Fuzzy Logic Systems: Reasoning based on "degrees of truth" rather than boolean logic
- Natural Language Processing Models: Techniques for processing and analyzing human language
- Reinforcement Learning: Techniques for learning using indirect "rewards" rather than explicit errors

"""+"""
Additional instructions:
1. Provide the response in the form of a JSON object with the following structure:
   {
     "justification": "Brief explanation of why these topics were chosen",
     "main_topic": {"name": "topic_1", "confidence": 0.0-1.0},
     "secondary_topics": [{"name": "topic_2", "confidence": 0.0-1.0}, {"name": "topic_3", "confidence": 0.0-1.0}]
   }
2. Use ONLY topics provided in the list above. If none fit, use "N/A"
3. Include up to two secondary topics if relevant, with a minimum confidence of 0.3.
4. Consider the overall context of the abstract, focusing on specific algorithms, methods, and problem domains discussed.
5. Be aware that general machine learning terms like "training", "convergence", and "performance" are used across many domains. Focus on the unique aspects of each domain.
6. Provide ONLY the JSON response, with no additional comments or text.
"""
          }]

def extract_topics(passage,
                  the_model='mistral-nemo:12b',
                  client=local_client,
                  retries=3):

  for count in range(retries):

    try:
      resp=client.chat.completions.create(
        model=the_model,
        messages=topic_extraction_msgs(passage)
      )
      
      js=json.loads(repair_json(resp.choices[0].message.content))
      return { 
        'main_topic' : js.get("main_topic","N/A"),
        'secondary_topic' : js.get("secondary_topic","N/A"),
        'model' : the_model,
        'num_retries' : count
      }
    except:
      pass

    # Give up
    return { 'main_topic':'N/A' , 'secondary_topic':'N/A' , 'model':the_model, 'num_retries':-1}
          

# Quick functions

def get_topics_groq(inseries):
  extract_topics_groq=lambda x:extract_topics(x,client=groq_client,the_model='llama3-70b-8192')
  res=[]
  for abs in tqdm(inseries):
    res.append(extract_topics_groq(abs)['main_topic']['name'])
  return res

def get_topics_pplx(inseries,the_client=pplx_client,the_model='llama-3.1-sonar-large-128k-chat'):
  extract_topics_pplx=lambda x:extract_topics(x,client=the_client,the_model=the_model)
  res=[]
  for abs in tqdm(inseries):
    try:
      res.append(extract_topics_pplx(abs)['main_topic']['name'])
    except:
      res.append('N/A')
  return res


model_map={"llama_small":"llama-3.1-8b-instruct",
           "llama_large":"llama-3.1-70b-instruct",
           "sonar_small":"llama-3.1-sonar-small-128k-chat",
           "sonar_large":"llama-3.1-sonar-large-128k-chat"}



# OpenAlex API calls
######################

def get_source_ids(the_name,max_reqs=20):

    base_url = "https://api.openalex.org/sources"
    params = {
              "filter": f"display_name.search:\"{the_name}\"",
              "per-page": 200,
              "cursor": "*"
          }

    all_sources=[]
    
    for count in range(max_reqs):

        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        sources = data.get("results", [])
        all_sources.extend(sources)

        if data.get("meta", {}).get("next_cursor"):
            params["cursor"] = data["meta"]["next_cursor"]
        else:
            break

    return all_sources


def get_publications_by_source(year,
                            source_id='S4210191458', # AAAI
                            max_reqs=20):

    base_url = "https://api.openalex.org/works"

    params = {
             "filter": f"primary_location.source.id:{source_id},publication_year:{year}",
             "per-page": 200,
             "cursor": '*'
         }
    
    all_publications = []
    
    for count in range(max_reqs):
        
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        
        publications = data.get("results", [])
        all_publications.extend(publications)
        
        if data.get("meta", {}).get("next_cursor"):
            params["cursor"] = data["meta"]["next_cursor"]
        else:
            break
    
    return all_publications


def test2010():

    # Retrieve the publications
    aaai_2010_publications = get_publications_by_source(2010)

    # Print the number of publications retrieved
    print(f"Total publications retrieved: {len(aaai_2010_publications)}")

    # Print some details of the first few publications
    for pub in aaai_2010_publications[:5]:
        print(f"Title: {pub['title']}")
        print(f"DOI: {pub['doi']}")
        print(f"Publication Date: {pub['publication_date']}")
        print("---")

    return aaai_2010_publications
