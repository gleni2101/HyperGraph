import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS: dict[str, Any] = {
    "graph": {
        "distill_system": "You are provided with a context chunk (delimited by ```). Your task is to respond with a concise scientific heading, summary, and a bulleted list with reasoning. Ignore human names, references, and citations.",
        "distill_user": "In a matter-of-fact voice, rewrite this ```{input}```. The writing must stand on its own, include required background details, and ignore references. Extract relevant table information and organize it. Focus on scientific facts.",
        "figure_system": "You are provided a figure that contains important information. Analyze the figure in detail and report the scientific facts. If it is not an academic figure, return an empty string. Always include the full image location.",
        "figure_user": "In a matter-of-fact voice, rewrite this ```{input}```. Include necessary background details. Extract relevant image information and organize it. Focus on scientific facts.",
        "graphmaker_system": "You are a logistics automation knowledge-graph extractor. You are provided with a context chunk (delimited by ```). Extract an ontology focused on logistics operations, planning, and execution. Prefer domain entities such as shipment, order, SKU, package, pallet, container, lane, route, stop, facility, warehouse, dock, carrier, vehicle, driver, customer, supplier, region, country, event, exception, KPI, SLA, cost, time window, and inventory state. Keep technical terms and abbreviations exactly as written (e.g., ETA, OTIF, ASN, POD, TMS, WMS). If you receive an image location, include it as a node with <id> equal to the location and <type>=\"image\", and connect it only when the relation is explicit in context. Use precise relation labels that are operationally meaningful (e.g., ships_to, departs_from, arrives_at, handled_by, delayed_by, violates, satisfies, depends_on, constrained_by, triggers, updates, measured_by). Do not include author names, citations, or generic filler nodes. Return a JSON with two fields: <nodes> and <edges>. Each node must have <id> and <type>. Each edge must have <source>, <target>, and <relation>.",
        "graphmaker_user": "Context: ```{input}```\nExtract the knowledge graph in structured JSON. Return only a valid JSON object with keys nodes and edges.",
    },
    "hypergraph": {
        "distill_system": "You are provided with a context chunk (delimited by ```). Your task is to respond with a concise scientific heading, summary, and a bulleted list with reasoning. Ignore human names, references, and citations.",
        "distill_user": "In a matter-of-fact voice, rewrite this ```{input}```. The writing must stand on its own, include required background details, and ignore references. Extract relevant table information and organize it. Focus on scientific facts.",
        "figure_system": "You are provided a figure that contains important information. Analyze the figure in detail and report the scientific facts. If it is not an academic figure, return an empty string. Always include the full image location.",
        "figure_user": "In a matter-of-fact voice, rewrite this ```{input}```. Include necessary background details. Extract relevant image information and organize it. Focus on scientific facts.",
        "graphmaker_system": "You are a logistics automation ontology extractor that returns precise Subject–Verb–Object events from a context chunk. You are provided with a context chunk delimited by triple backticks: ```. Extract logistics operations, planning, constraints, exceptions, and KPI relations. Target entities include: shipment, order, line item, SKU, package, pallet, container, lane, route, stop, facility, warehouse, dock, carrier, vehicle, driver, customer, supplier, region, country, event, alert, SLA, KPI, delay, inventory, and capacity. Keep technical abbreviations verbatim (ETA, ETD, ASN, POD, OTIF, TMS, WMS, ERP). Produce two passes: (1) exact grammatical S–V–O extraction from each sentence, and (2) conservative completion for explicit logistics meaning not realized as a clean S–V–O surface form. Rules: work sentence by sentence, preserve source/target text exactly as written, split explicitly coordinated entities into lists, use operational relation phrases, resolve vague mentions only when clear, omit author names/references/filler nodes, and include only fields source/relation/target. Return a JSON object with one field 'events', where each event has source: list[str], relation: str, target: list[str]. Return only this JSON object.",
        "graphmaker_user": "Context: ```{input}```\nExtract the hypergraph knowledge graph in structured JSON format. Return only a valid JSON object with key events.",
    },
    "runtime": {
        "default_system_prompt": "You extract scientific relations from a context chunk. Return strict JSON with one top-level key 'events'. Each event must include: source (list[str]), target (list[str]), relation (str).",
        "figure_system_prompt": "You are an assistant who describes scientific figures.",
        "figure_user_prompt": "Describe this figure in detail.",
    },
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_config_path(config_path: str | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser().resolve()
    env_path = os.getenv("GRAPH_REASONING_PROMPT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "prompt_config.json").resolve()


def load_prompt_config(config_path: str | None = None) -> dict[str, Any]:
    resolved = _resolve_config_path(config_path)
    if not resolved.exists():
        return DEFAULT_PROMPTS

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            return DEFAULT_PROMPTS
        return _deep_merge(DEFAULT_PROMPTS, loaded)
    except Exception:
        return DEFAULT_PROMPTS


def get_prompt(section: str, key: str, config_path: str | None = None, **kwargs) -> str:
    prompts = load_prompt_config(config_path=config_path)
    section_data = prompts.get(section, {}) if isinstance(prompts, dict) else {}
    template = section_data.get(key, "") if isinstance(section_data, dict) else ""
    if not isinstance(template, str):
        return ""
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template
