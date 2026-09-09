"""Workflow assembly — builds the agentic graph with real dependencies.

Graph topology (Part C):

                       ┌──────────────┐
                 ┌────▶│   RETRIEVER  │────┐      (multi-query, per-entity,
                 │     └──────────────┘    ▼       round-robin merged when the
          ┌──────────┐            ┌────────┐        question was decomposed)
 question │  ROUTER  │            │ GRADER │   relevant    ┌───────────┐
 ────────▶│          │            └────────┘───────┐──────▶│ GENERATOR │
          └─┬─┬─┬────┘                 ▲           │       └─────┬─────┘
   chitchat │ │ │ out_of_scope         │ not_rel.  │             ▼
            │ │ ▼                ┌─────┴────┐      │       ┌───────────┐
  comparative│ ┌──────────┐      │ REWRITER │◀─────┘       │ VERIFIER  │
  multi-entity┴▶DECOMPOSER│      └──────────┘ (retries<N)  └─────┬─────┘
            ┌────┴────────┘                                    │
            ▼          ┌──────────┐        not_supported       │
       ┌─────────┐     │ FALLBACK │◀──────────────────────────┘
       │ CHITCHAT│     └────▲─────┘
       └─────────┘          └────────────────────────────────────
                        supported → END (with citations)
"""

from __future__ import annotations

from app.agents.graph import AgentGraph, END
from app.agents.nodes import (
    grader_conditional,
    make_chitchat_node,
    make_decomposer_node,
    make_fallback_node,
    make_generator_node,
    make_grader_node,
    make_retriever_node,
    make_rewriter_node,
    make_router_node,
    make_verifier_node,
)
from app.llm.service import ModelService
from app.rag.retriever import HybridRetriever


def build_graph(models: ModelService, retriever: HybridRetriever, mode: str) -> AgentGraph:
    """Wire the full agentic workflow. `mode` snapshots the serving mode at
    build time; the graph is rebuilt by the app when the mode flips."""
    g = AgentGraph()
    g.add_node("router", make_router_node(models.brain))
    # decomposer gets the brain for LLM-driven decomposition in vLLM mode
    # (validated, regex path stays the guarantee — see decomposer.py)
    g.add_node("decomposer", make_decomposer_node(models.brain))
    g.add_node("retriever", make_retriever_node(retriever))
    g.add_node("grader", make_grader_node(models.brain))
    g.add_node("rewriter", make_rewriter_node(models.brain))
    g.add_node("generator", make_generator_node(models.brain))
    g.add_node("verifier", make_verifier_node(models.brain))
    g.add_node("chitchat", make_chitchat_node(mode, models.backend))
    g.add_node("fallback", make_fallback_node())

    g.set_entry_point("router")
    # Unconditional simple edges:
    g.add_edge("decomposer", "retriever")
    g.add_edge("retriever", "grader")
    g.add_edge("rewriter", "retriever")
    g.add_edge("generator", "verifier")
    # Conditional hops:
    g.add_conditional_edges("grader", grader_conditional)   # relevant | rewrite | fallback
    # router / verifier / chitchat / fallback return explicit next steps
    # (END or a node name) from inside the node function itself.
    return g
