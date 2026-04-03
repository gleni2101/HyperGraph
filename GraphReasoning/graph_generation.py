from GraphReasoning.graph_tools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *
from GraphReasoning.prompt_config import get_prompt

from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import networkx as nx
import os
import re
import importlib

def _get_misc_properties(self):
    if "misc_properties" in self.columns:
        return self["misc_properties"]
    return pd.Series([{} for _ in range(len(self))], index=self.index, dtype=object)


def _set_misc_properties(self, value):
    self["misc_properties"] = value


pd.DataFrame.misc_properties = property(_get_misc_properties, _set_misc_properties)

try:
    RecursiveCharacterTextSplitter = importlib.import_module(
        "langchain.text_splitters"
    ).RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    RecursiveCharacterTextSplitter = importlib.import_module(
        "langchain_text_splitters"
    ).RecursiveCharacterTextSplitter

try:
    import pdfkit
except ImportError:
    pdfkit = None
from pathlib import Path
import random
from pyvis.network import Network
from tqdm.notebook import tqdm

import seaborn as sns

from hashlib import md5


#hypergraph add ons
import json #do we need? 
import re #do we need? 
import hypernetx as hnx
import pickle



palette = "hls"
# Code based on: https://github.com/rahulnyk/knowledge_graph


def _cache_dir() -> Path:
    cache_root = os.getenv("GRAPH_REASONING_CACHE_DIR", "temp")
    cache_path = Path(cache_root)
    if not cache_path.is_absolute():
        cache_path = (Path.cwd() / cache_path).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _parse_json_object_from_text(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected non-empty JSON text.")

    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload

    raise ValueError("Could not parse a JSON object from model output.")


def _coerce_structured_payload(value) -> dict:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        return _parse_json_object_from_text(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, dict):
            return dumped

    fields = {}
    for field_name in ("nodes", "edges", "events"):
        if hasattr(value, field_name):
            fields[field_name] = getattr(value, field_name)
    if fields:
        return fields

    raise ValueError("Unsupported structured output type from generate(...).")


def _item_get(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _to_string_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _to_text(value) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
            "chunk_id": md5(chunk.encode()).hexdigest(),
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)

    return df


def df2Graph(df: pd.DataFrame, generate, generate_figure=None, image_list=None, repeat_refine=0, do_distill=True, do_relabel = False, verbatim=False,
          
            ) -> nx.DiGraph:
    
    subgraph_list = []
    for _, row in df.iterrows():
        subgraph = graphPrompt(
            row.text, 
            generate,
            generate_figure, 
            image_list,
            {"chunk_id": row.chunk_id}, 
            do_distill=do_distill,
            do_relabel=do_relabel,
            repeat_refine=repeat_refine, 
            verbatim=verbatim,
        )
        print(subgraph, type(subgraph))
        subgraph_list.append(subgraph)

        
    G = nx.DiGraph()

    for g in subgraph_list:
        G = nx.compose(G, g)
    
    return G

def df2hypergraph(
    df: pd.DataFrame,
    generate,
    generate_figure=None,
    image_list=None,
    repeat_refine=0,
    do_distill=True,
    do_relabel=False,
    verbatim=False,
) -> hnx.Hypergraph:
    """
    Build one big HyperNetX hypergraph by unioning together
    all of the sub-hypergraphs produced for each row of the DataFrame.
    """
    sub_hgs = []
    sub_dfs = []

    for _, row in df.iterrows():
        try:
            hg, chunk_df = hypergraphPrompt(
                row.text,
                generate,
                generate_figure,
                image_list,
                {"chunk_id": row.chunk_id},
                do_distill=do_distill,
                do_relabel=do_relabel,
                repeat_refine=repeat_refine,
                verbatim=verbatim,
            )
            # Only keep valid subgraphs
            if isinstance(hg, hnx.Hypergraph):
                sub_hgs.append(hg)
                sub_dfs.append(chunk_df)
            else:
                print(f"Skipping chunk {row.chunk_id} – returned non-Hypergraph")
        except Exception as e:
            print(f"Exception while processing chunk {row.chunk_id}: {e}")

    if not sub_hgs:
        print("No valid subgraphs found. Returning None.")
        return None, None

    # Start from an empty hypergraph
    H = hnx.Hypergraph({})

    # Union them all safely
    for hg in sub_hgs:
        H = H.union(hg)

    return H, sub_dfs

import sys
sys.path.append("..")

import json

def graphPrompt(input: str, generate, generate_figure=None, image_list=None, metadata={}, #model="mistral-openorca:latest",
                do_distill=True, repeat_refine=0,verbatim=False,
               ) -> nx.DiGraph:
    cache_graphml = _cache_dir() / f"{metadata['chunk_id']}.graphml"
    
    try:
        return nx.read_graphml(cache_graphml)
    except:
        pass

    SYS_PROMPT_DISTILL = get_prompt("graph", "distill_system")

    USER_PROMPT_DISTILL = get_prompt("graph", "distill_user", input=input)

    SYS_PROMPT_FIGURE = get_prompt("graph", "figure_system")

    USER_PROMPT_FIGURE = get_prompt("graph", "figure_user", input=input)
    input_fig = ''
    if generate_figure: # if image in the chunk
        
        for image_name in image_list:
            _image_name = image_name.split('/')[-1]
            if _image_name.lower() in input.lower():  
                input_fig =  f'Here is the information in the image: {image_name}' + \
                generate_figure( image = image_name, system_prompt=SYS_PROMPT_FIGURE, prompt=USER_PROMPT_FIGURE)
    
    if do_distill:
        #Do not include names, figures, plots or citations in your response, only facts."
        input = _to_text(generate( system_prompt=SYS_PROMPT_DISTILL, prompt=USER_PROMPT_DISTILL))

    if input_fig:
        input += input_fig
    
    SYS_PROMPT_GRAPHMAKER = get_prompt("graph", "graphmaker_system")
     
    USER_PROMPT = get_prompt("graph", "graphmaker_user", input=input)
    # result = [dict(item, **metadata) for item in result]
    
    print ('Generating triples...')
    result_raw  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)

    try:
        result = _coerce_structured_payload(result_raw)
    except Exception as e:
        print(f"Failed to parse graph JSON for chunk {metadata.get('chunk_id')}: {e}")
        return nx.DiGraph()

    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    G = nx.DiGraph()
    for node in nodes:
        node_id = _item_get(node, "id")
        if node_id is None:
            continue
        node_type = _item_get(node, "type", "entity")
        G.add_node(str(node_id), type=str(node_type))
    for edge in edges:
        source = _item_get(edge, "source")
        target = _item_get(edge, "target")
        relation = _item_get(edge, "relation", "related_to")
        if source is None or target is None:
            continue
        G.add_edge(str(source), str(target), relation=str(relation), chunk_id=metadata['chunk_id'])

    nx.write_graphml(G, cache_graphml)
    print(f'Generated graph: {G}')

    return G


def hypergraphPrompt(input: str, generate, generate_figure=None, image_list=None, metadata={}, #model="mistral-openorca:latest",
                do_distill=True, do_relabel=False,repeat_refine=0,verbatim=False,
               ) -> hnx.Hypergraph:
    cache_pkl = _cache_dir() / f"{metadata['chunk_id']}.pkl"
    
    try:
        with open(cache_pkl, "rb") as fin:
            cached = pickle.load(fin)
        if verbatim:
            print(f"Loaded hypergraph from {cache_pkl}")
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached
        return cached, pd.DataFrame()
    except:
        pass
    SYS_PROMPT_DISTILL = get_prompt("hypergraph", "distill_system")

    USER_PROMPT_DISTILL = get_prompt("hypergraph", "distill_user", input=input)

    SYS_PROMPT_FIGURE = get_prompt("hypergraph", "figure_system")

    USER_PROMPT_FIGURE = get_prompt("hypergraph", "figure_user", input=input)
    input_fig = ''
    if generate_figure: # if image in the chunk
        
        for image_name in image_list:
            _image_name = image_name.split('/')[-1]
            if _image_name.lower() in input.lower():  
                input_fig =  f'Here is the information in the image: {image_name}' + \
                generate_figure( image = image_name, system_prompt=SYS_PROMPT_FIGURE, prompt=USER_PROMPT_FIGURE)
    
    if do_distill:
        #Do not include names, figures, plots or citations in your response, only facts."
        input = _to_text(generate( system_prompt=SYS_PROMPT_DISTILL, prompt=USER_PROMPT_DISTILL))

    if input_fig:
        input += input_fig

    SYS_PROMPT_GRAPHMAKER = get_prompt("hypergraph", "graphmaker_system")
 
    #USER_PROMPT = f'Context: ```{input}``` \n\ Extract the hypergraph knowledge graph in structured JSON format: '
    USER_PROMPT = get_prompt("hypergraph", "graphmaker_user", input=input)

    print ('Generating hypergraph...')
    validated_raw  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)

    try:
        validated_result = _coerce_structured_payload(validated_raw)
    except Exception as e:
        print(f"Failed to parse hypergraph JSON for chunk {metadata.get('chunk_id')}: {e}")
        return None, pd.DataFrame()

    raw_events = validated_result.get("events", [])
    events = []
    for event in raw_events:
        source = _to_string_list(_item_get(event, "source"))
        target = _to_string_list(_item_get(event, "target"))
        relation = _item_get(event, "relation", "related_to")
        if not source or not target:
            continue
        events.append({
            "source": source,
            "relation": str(relation),
            "target": target,
        })

    if not events:
        print(f"No valid events found for chunk {metadata.get('chunk_id')}.")
        return None, pd.DataFrame()

    # 1) Build the raw edge→relation mapping
    edge_mapping = {
        f"e{i+1}": event["relation"]
        for i, event in enumerate(events)
    }
    
    # 2) Build the base incidence dict
    base_edge_dict = {
        eid: (
            #set(event.source if isinstance(event.source, list) else [event.source])
            set(event["source"]) | set(event["target"])
            #| {event.target}
        )
        for eid, event in zip(edge_mapping.keys(), events)
    }
    
    # 3) Prepare the source/target/chunk maps
    source_map = {
        eid: event["source"]
        for eid, event in zip(edge_mapping.keys(), events)
    }
    target_map = {
        eid: event["target"]
        for eid, event in zip(edge_mapping.keys(), events)
    }
    chunk_map = {
        eid: metadata["chunk_id"]
        for eid in edge_mapping.keys()
    }
    
    # 4) Choose your edge IDs
    if do_relabel:
        # keep e1, e2, … IDs
        final_incidence = base_edge_dict
        final_source   = source_map
        final_target   = target_map
        final_chunk    = chunk_map

    else:
    # use human-readable relation names with unique suffixes
        renamed_edges = {
        eid: f"{edge_mapping[eid]}_chunk{chunk_map[eid]}_{i}"
        for i, eid in enumerate(edge_mapping.keys())
        }
    
        final_incidence = {
            renamed_edges[eid]: nodes
            for eid, nodes in base_edge_dict.items()
        }
        final_source = {
            renamed_edges[eid]: val
            for eid, val in source_map.items()
        }
        final_target = {
            renamed_edges[eid]: val
            for eid, val in target_map.items()
        }
        final_chunk = {
            renamed_edges[eid]: val
            for eid, val in chunk_map.items()
        }
        
    
    # 5) Create the HyperNetX hypergraph
    H_simple = hnx.Hypergraph(final_incidence)
    
    # 6) Build a combined DataFrame    
    rows = []
    for eid, nodes in final_incidence.items():
        rows.append({
            "edge":   eid,
            "nodes":  nodes,
            "source": final_source[eid],
            "target": final_target[eid],
            "chunk":  final_chunk[eid],
        })
    try:
        chunk_df = pd.DataFrame(rows).set_index("edge")
    except KeyError as e:
        print("Error during KG generation - skipping this chunk!:", e)
        return None, pd.DataFrame()
        
    print(
        f"Generated hypergraph with {len(H_simple.nodes)} nodes, "
        f"{len(H_simple.edges)} edges."
    )

    try:
        with open(cache_pkl, "wb") as fout:
            pickle.dump((H_simple, chunk_df), fout)
    except Exception:
        pass
    
    # 7) Return both graph and table
    return H_simple, chunk_df


def colors2Community(communities) -> pd.DataFrame:
    
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

def make_graph_from_text (txt,generate, generate_figure=None, image_list=None,
                          graph_root='graph_root',
                          chunk_size=2500,chunk_overlap=0,do_distill=True, do_relabel=False,
                          repeat_refine=0,verbatim=False,
                          data_dir='./data_output_KG/',
                          save_HTML=False,
                          save_PDF=False,#TO DO
                         ):    
    
    ## data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)     
    graph_GraphML=  f'{data_dir}/{graph_root}.graphml'  #  f'{data_dir}/result.graphml',

    try:
        G = nx.read_graphml(graph_GraphML)
    except:

        outputdirectory = Path(f"./{data_dir}/") #where graphs are stored from graph2df function
        
    
        splitter = RecursiveCharacterTextSplitter(
            #chunk_size=5000, #1500,
            chunk_size=chunk_size, #1500,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        pages = splitter.split_text(txt)
        print("Number of chunks = ", len(pages))
        if verbatim:
            display(Markdown (pages[0]) )
        
        df = documents2Dataframe(pages)
        df.to_csv(f'{data_dir}/{graph_root}_chunks_clean.csv')

        G = df2Graph(df,generate, generate_figure, image_list, do_distill=do_distill, do_relabel=do_relabel, repeat_refine=repeat_refine,verbatim=verbatim) #model='zephyr:latest' )

        nx.write_graphml(G, graph_GraphML)
        

    graph_HTML = None
    net= None
    output_pdf = None
    if save_HTML:
        net = Network(
                notebook=True,
                cdn_resources="remote",
                height="900px",
                width="100%",
                select_menu=True,
                filter_menu=False,
            )

        net.from_nx(G)
        net.force_atlas_2based(central_gravity=0.015, gravity=-31)

        net.show_buttons()

        graph_HTML= f'{data_dir}/{graph_root}.html'
        net.save_graph(graph_HTML,
                )
        if verbatim:
            net.show(graph_HTML,
                )


        if save_PDF:
            output_pdf=f'{data_dir}/{graph_root}.pdf'
            if pdfkit is None:
                raise ImportError("pdfkit is required for save_PDF=True. Install it with `pip install pdfkit` and ensure wkhtmltopdf is available on PATH.")
            pdfkit.from_file(graph_HTML,  output_pdf)
        
    
    return graph_HTML, graph_GraphML, G, net, output_pdf

def make_hypergraph_from_text(
    txt,
    generate,
    generate_figure=None,
    image_list=None,
    graph_root='graph_root',
    chunk_size=2500,
    chunk_overlap=0,
    do_distill=True,
    do_relabel=False,
    repeat_refine=0,
    verbatim=False,
    data_dir='./data_output_KG/',
):
    """
    Builds or loads a graph stored in a .pkl file.

    - If `{graph_root}.pkl` exists in `data_dir`, loads and returns it.
    - Otherwise, splits `txt` into chunks, generates a graph `G`, 
      pickles `G` to `{graph_root}.pkl`, and returns it.

    Returns:
    pkl_path (str), G 
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    pkl_path = os.path.join(data_dir, f'{graph_root}.pkl')
    sub_dfs_pkl_path = os.path.join(data_dir, f'{graph_root}_sub_dfs.pkl')

    # Load or build the graph
    if os.path.isfile(pkl_path):
        with open(pkl_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Loaded existing graph from {pkl_path}")
    else:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        pages = splitter.split_text(txt)
        print("Number of chunks =", len(pages))
        if verbatim:
            from IPython.display import Markdown, display
            display(Markdown(pages[0]))

        # Convert chunks to DataFrame
        df = documents2Dataframe(pages)
        df.to_csv(os.path.join(data_dir, f'{graph_root}_chunks_clean.csv'), index=False)

        # Generate graph
        G, sub_dfs = df2hypergraph(
            df,
            generate,
            generate_figure,
            image_list,
            do_distill=do_distill,
            do_relabel=do_relabel,
            repeat_refine=repeat_refine,
            verbatim=verbatim
        )

        # Save as pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Saved new graph to {pkl_path}")

        # Save sub_dfs as pickle
        with open(sub_dfs_pkl_path, 'wb') as f:
            pickle.dump(sub_dfs, f)
        print(f"Saved new graph to {sub_dfs_pkl_path}")
      

    return pkl_path, G, sub_dfs_pkl_path, sub_dfs


import time
from copy import deepcopy

def add_new_subgraph_from_text(txt=None,generate=None,generate_figure=None, image_list=None, 
                               node_embeddings=None,tokenizer=None, model=None, original_graph=None,
                               data_dir_output='./data_temp/',graph_root='graph_root',
                               chunk_size=10000,chunk_overlap=2000,
                               do_update_node_embeddings=True, do_distill=True, do_relabel = False, 
                               do_simplify_graph=True,size_threshold=10,
                               repeat_refine=0,similarity_threshold=0.95,
                               do_Louvain_on_new_graph=True, 
                               #whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,
                               graph_GraphML_to_add=None,
                               verbatim=True,):

    display (Markdown(txt[:32]+"..."))
    graph_GraphML=None
    G_new=None
    
    res=None
    # try:
    start_time = time.time() 

    if verbatim:
        print ("Now create or load new graph...")

    if (G_to_add is not None and graph_GraphML_to_add is not None):
        print("G_to_add and graph_GraphML_to_add cannot be used together. Pick one or the other to provide a graph to be added.")
        return
    elif graph_GraphML_to_add==None and G_to_add==None: #make new if no existing one provided
        print ("Make new graph from text...")
        _, graph_GraphML_to_add, G_to_add, _, _ =make_graph_from_text (txt,generate,
                                 data_dir=data_dir_output,
                                 graph_root=f'graph_root',
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                 repeat_refine=repeat_refine, 
                                 verbatim=verbatim,
                                 )
        if verbatim:
            print ("New graph from text provided is generated and saved as: ", graph_GraphML_to_add)
    elif G_to_add is None:
        if verbatim:
            print ("Loading or using provided graph... Any txt data provided will be ignored...:", G_to_add, graph_GraphML_to_add)
            G_to_add = nx.read_graphml(graph_GraphML_to_add)
    # res_newgraph=graph_statistics_and_plots_for_large_graphs(G_to_add, data_dir=data_dir_output,                                      include_centrality=False,make_graph_plot=False,                               root='new_graph')
    print("--- %s seconds ---" % (time.time() - start_time))
    # except:
        # print ("ALERT: Graph generation failed...")
        
    print ("Now grow the existing graph...")
    
    # try:
    #Load original graph
    if type(original_graph) == str:
        G = nx.read_graphml(original_graph)
    else:
        G = deepcopy(original_graph)
    print(G, G_to_add)
    G_new = nx.compose(G, G_to_add)

    if do_update_node_embeddings:
        if verbatim:
            print ("Now update node embeddings")
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if do_simplify_graph:
        if verbatim:
            print ("Now simplify graph.")
        G_new, node_embeddings = simplify_graph (G_new, node_embeddings, tokenizer, model , 
                                                similarity_threshold=similarity_threshold, use_llm=False, data_dir_output=data_dir_output,
                                verbatim=verbatim,)
    if size_threshold >0:
        if verbatim:
            print ("Remove small fragments")            
        G_new=remove_small_fragents (G_new, size_threshold=size_threshold)
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if return_only_giant_component:
        if verbatim:
            print ("Select only giant component...")   
        connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
        G_new = G_new.subgraph(connected_components[0]).copy()
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if do_Louvain_on_new_graph:
        G_new=graph_Louvain (G_new, graph_GraphML=None)
        if verbatim:
            print ("Done Louvain...")

    if verbatim:
        print ("Done update graph")

    graph_GraphML= f'{data_dir_output}/{graph_root}_integrated.graphml'
    if verbatim:
        print ("Save new graph as: ", graph_GraphML)

    nx.write_graphml(G_new, graph_GraphML)
    if verbatim:
        print ("Done saving new graph")
    
    # res=graph_statistics_and_plots_for_large_graphs(G_new, data_dir=data_dir_output,include_centrality=False,make_graph_plot=False,root='assembled')
    # print ("Graph statistics: ", res)

    # except:
        # print ("Error adding new graph.")
    print(G_new, graph_GraphML)
        # print (end="")

    return graph_GraphML, G_new, G_to_add, node_embeddings, res


#START SANITIZE

import pandas as pd, re
import hypernetx as hnx

_CC = re.compile(r'[\x00-\x1F\x7F]')

def _scrub(s) -> str:
    s = "" if s is None else str(s)
    s = _CC.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _flatten_props_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(index=pd.Index([], name="edge"))
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        if "edge" in df.columns:
            df = df.set_index("edge")
        else:
            idx_cols = [c for c in df.columns if str(c).startswith("level_")]
            if not idx_cols:
                idx_cols = list(df.columns[:2])
            df["edge"] = df[idx_cols].astype(str).agg("|".join, axis=1)
            df = df.drop(columns=idx_cols).set_index("edge")
    df.index = pd.Index([_scrub(ix) for ix in df.index], name=df.index.name or "edge")
    return df

def _collect_incidence(G):
    """Return {edge_id: [node,...]} robustly across HNX versions (Hypergraph or HypergraphView)."""
    # 1) Best: incidence_dict (edge -> set(nodes))
    if hasattr(G, "incidence_dict"):
        return {e: [n for n in nodes] for e, nodes in G.incidence_dict.items()}

    # 2) Next: indexable edges (edges[e] -> iterable of nodes)
    try:
        return {e: list(G.edges[e]) for e in G.edges}
    except Exception:
        pass

    # 3) Fallback: rebuild from node→edges (adjacency_dict)
    if hasattr(G, "adjacency_dict"):
        inc = {}
        for n, edges in G.adjacency_dict.items():
            for e in edges:
                inc.setdefault(e, set()).add(n)
        return {e: list(ns) for e, ns in inc.items()}

    # If all else fails, raise with a hint
    raise TypeError("Could not extract incidence from HyperNetX object; version/API mismatch.")

def normalize_hnx_graph_rebuild(G: hnx.Hypergraph) -> hnx.Hypergraph:
    """
    Build a *new* Hypergraph with:
      - clean string edge ids (tuples -> 'a|b', control chars stripped)
      - scrubbed node labels
      - edge_properties single-index DF aligned to edge ids
    """
    # Edge id map (tuple -> joined; scrub)
    edge_map = {}
    for e in G.edges:
        if isinstance(e, tuple):
            edge_map[e] = _scrub("|".join(map(str, e)))
        else:
            edge_map[e] = _scrub(e)

    # Node label map
    node_map = {n: _scrub(n) for n in G.nodes}

    # Incidence via robust collector
    raw_inc = _collect_incidence(G)
    incidence = {edge_map[e]: [node_map[n] for n in raw_inc[e]] for e in raw_inc}

    # Properties
    try:
        props_df = _flatten_props_df(getattr(G._E, "properties", None))
    except Exception:
        props_df = pd.DataFrame(index=pd.Index([], name="edge"))

    if len(props_df.index) > 0:
        props_df.index = pd.Index([_scrub(ix) for ix in props_df.index], name="edge")

    # Align props to incidence keys exactly
    edges = pd.Index(list(incidence.keys()), name="edge")
    props_df = props_df.loc[props_df.index.intersection(edges)]
    missing = edges.difference(props_df.index)
    if len(missing):
        props_df = pd.concat([props_df, pd.DataFrame(index=missing)], axis=0)
    props_df = props_df.reindex(edges)

    return hnx.Hypergraph(incidence, edge_properties=props_df)

def safe_union(H: hnx.Hypergraph, G2: hnx.Hypergraph) -> hnx.Hypergraph:
    Hn  = normalize_hnx_graph_rebuild(H)
    G2n = normalize_hnx_graph_rebuild(G2)
    return Hn.union(G2n)

#END SANITIZE


def add_new_hypersubgraph_from_text(
    txt=None, generate=None, generate_figure=None,
    image_list=None, node_embeddings=None, tokenizer=None,
    model=None, original_graph=None, data_dir_output='./data_temp/',
    graph_root='graph_root', chunk_size=10000, chunk_overlap=2000,
    do_update_node_embeddings=True, do_distill=True, do_relabel=False, do_simplify_graph=True,
    size_threshold=10, repeat_refine=0, similarity_threshold=0.95,
    do_Louvain_on_new_graph=True, return_only_giant_component=False,
    save_common_graph=False, G_to_add=None, graph_pkl_to_add=None, sub_dfs=None,
    verbatim=True,
):
    if verbatim and isinstance(txt, str) and txt:
        preview = txt[:120].replace("\n", " ")
        print(f"Text preview: {preview}...")

    updated_sub_dfs = sub_dfs #if no update is made 

    start_time = time.time()
    if verbatim:
        print("Now create or load new hypergraph...")

    # Determine hypergraph to add
    if G_to_add is not None and graph_pkl_to_add is not None:
        print("Provide only one of G_to_add or graph_pkl_to_add.")
        return
    elif graph_pkl_to_add is None and G_to_add is None:
        # Generate new hypergraph
        print("Make new hypergraph from text...")
        graph_pkl_to_add, G_to_add, _, _ = make_hypergraph_from_text(
            txt, generate, data_dir=data_dir_output,
            graph_root=graph_root, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, repeat_refine=repeat_refine,
            verbatim=verbatim,
        )
        if verbatim:
            print(f"Received new PKL from make_hypergraph_from_text: {graph_pkl_to_add}")
    elif G_to_add is None:
        # Load existing PKL
        if verbatim:
            print(f"Loading hypergraph from PKL; txt ignored: {graph_pkl_to_add}")
        with open(graph_pkl_to_add, 'rb') as f:
            G_to_add = pickle.load(f)

    print(f"--- Load/generate time: {time.time() - start_time:.2f}s ---")
    print("Now grow the existing hypergraph...")

    # Load original hypergraph (PKL or object)
    if isinstance(original_graph, str):
        with open(original_graph, 'rb') as f:
            H = pickle.load(f)
    else:
        H = deepcopy(original_graph)

    # Merge hypergraphs via HyperNetX union
    G_new = H.union(G_to_add)
    #G_new = safe_union(H, G_to_add)

    if do_update_node_embeddings:
        if verbatim:
            print("Updating node embeddings...")
        node_embeddings = update_hypernode_embeddings(
            node_embeddings, G_new, tokenizer, model, verbatim=verbatim
        )
        

    # 2) After simplification (if it itself updates embeddings)
    if do_simplify_graph:
        if verbatim:
            print("Simplifying hypergraph...")
        G_new, node_embeddings, updated_sub_dfs = simplify_hypergraph(
            G_new, sub_dfs, node_embeddings, tokenizer, model,
            similarity_threshold=similarity_threshold,
            use_llm=False, data_dir_output=data_dir_output,
            verbatim=verbatim,
        )
        

    # 3) After pruning small fragments
    if size_threshold > 0:
        if verbatim:
            print("Removing small fragments...")
        G_new, updated_sub_dfs = remove_small_hyperfragments(
            H_new=G_new,
            sub_dfs=updated_sub_dfs,
            size_threshold=size_threshold,
            return_singletons=False
        )
        node_embeddings = update_hypernode_embeddings(
            node_embeddings, G_new, tokenizer, model, verbatim=verbatim
        )


    # Giant component only
    if return_only_giant_component:
        if verbatim:
            print("Selecting giant component...")
        cc = sorted(hnx.connected_components(G_new), key=len, reverse=True)
        G_new = G_new.subhypergraph(cc[0])
        node_embeddings = update_hypernode_embeddings(
            node_embeddings, G_new, tokenizer, model, verbatim=verbatim
        )

    # Louvain clustering
    if do_Louvain_on_new_graph:
        G_new = graph_Louvain(G_new)
        if verbatim:
            print("Completed Louvain clustering.")

    if verbatim:
        print("Hypergraph update complete.")

    def _safe_pickle_write(file_path: str, payload, retries: int = 6, delay_seconds: float = 1.0):
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        temp_path = f"{file_path}.tmp"
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(payload, f)
                os.replace(temp_path, file_path)
                return
            except PermissionError as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                wait_time = delay_seconds * attempt
                print(
                    f"Permission denied writing {file_path} (attempt {attempt}/{retries}). "
                    f"temp_path={temp_path}. Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            finally:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass

        if last_exc is not None:
            raise PermissionError(
                f"Could not write file after {retries} attempts: {file_path} (temp: {temp_path})"
            ) from last_exc

    # Save integrated hypergraph as PKL
    integrated_pkl = f"{data_dir_output}/{graph_root}_integrated.pkl"
    if verbatim:
        print(f"Saving integrated hypergraph to: {integrated_pkl}")
    _safe_pickle_write(integrated_pkl, G_new)
    if verbatim:
        print("Integrated hypergraph saved.")

    #save the updated sub_dfs 
    updated_subdfs_pkl = f"{data_dir_output}/{graph_root}_updated_sub_dfs.pkl"
    if verbatim:
        print(f"Saving updated subdfs to: {updated_subdfs_pkl}")
    _safe_pickle_write(updated_subdfs_pkl, updated_sub_dfs)
    if verbatim:
        print("Updated subdfs are saved.")

    print(G_new, integrated_pkl)
    return integrated_pkl, G_new, G_to_add, node_embeddings, updated_sub_dfs

