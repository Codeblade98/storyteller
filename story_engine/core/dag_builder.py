"""Scene DAG construction and ordering utilities."""

from collections import defaultdict, deque
from typing import Iterable
import logging

from story_engine.core.edge_context_designer import EdgeContextDesigner
from story_engine.models.edge import EdgeContext
from story_engine.models.node import SceneNode
from story_engine.models.scene import ScenePlan

logger = logging.getLogger("story_engine.engine")


class SceneDAG:
    """Directed acyclic graph of scene nodes and their dependencies.

    The DAG stores the scene nodes and edges used to determine execution order.
    """

    def __init__(self, nodes: dict[str, SceneNode], edges: list[EdgeContext]) -> None:
        """Initialize the DAG with its nodes and edges."""
        self.nodes = nodes
        self.edges = edges

    def topological_order(self) -> list[SceneNode]:
        """Return scene nodes in dependency-safe topological order.

        Uses NetworkX when available and falls back to an in-module
        implementation if NetworkX cannot be imported.
        """
        try:
            return self._networkx_order()
        except ImportError:
            return self._fallback_order()

    def _networkx_order(self) -> list[SceneNode]:
        """Compute a topological ordering using NetworkX.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        import networkx as nx

        graph = nx.DiGraph()
        for scene_id in self.nodes:
            graph.add_node(scene_id)
        for edge in self.edges:
            graph.add_edge(edge.source, edge.target)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Scene dependency graph contains a cycle")
        return [self.nodes[scene_id] for scene_id in nx.topological_sort(graph)]

    def _fallback_order(self) -> list[SceneNode]:
        """Compute a topological ordering without external graph libraries.

        This uses Kahn's algorithm to produce the same ordering guarantee as
        the NetworkX-based implementation.
        """
        indegree = {scene_id: 0 for scene_id in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            children[edge.source].append(edge.target)
            indegree[edge.target] += 1

        queue = deque(scene_id for scene_id, degree in indegree.items() if degree == 0)
        ordered: list[str] = []
        while queue:
            scene_id = queue.popleft()
            ordered.append(scene_id)
            for child in children[scene_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(self.nodes):
            raise ValueError("Scene dependency graph contains a cycle")
        return [self.nodes[scene_id] for scene_id in ordered]


class SceneDAGBuilder:
    """Build a scene dependency graph from scene plans."""

    def __init__(self, edge_context_designer: EdgeContextDesigner | None = None) -> None:
        self.edge_context_designer = edge_context_designer or EdgeContextDesigner()

    def build(self, plans: Iterable[ScenePlan]) -> SceneDAG:
        """Build a DAG from the provided scene plans.

        Raises:
            ValueError: If scene IDs are duplicated or dependencies are invalid.
        """
        plan_list = list(plans)
        logger.info("dag_build_started scene_count=%s", len(plan_list))
        plan_by_id = {plan.scene_id: plan for plan in plan_list}
        if len(plan_by_id) != len(plan_list):
            raise ValueError("Scene IDs must be unique")

        nodes = {scene_id: SceneNode(scene_id=scene_id, plan=plan) for scene_id, plan in plan_by_id.items()}
        edges: list[EdgeContext] = []

        for plan in plan_list:
            for source in plan.hard_dependencies or plan.depends_on:
                self._validate_dependency(source, plan.scene_id, plan_by_id)
                edge = self.edge_context_designer.design(plan_by_id[source], plan, "hard")
                edges.append(edge)
                nodes[source].outgoing_edges.append(edge)
                nodes[plan.scene_id].incoming_edges.append(edge)

            for source in plan.soft_dependencies:
                self._validate_dependency(source, plan.scene_id, plan_by_id)
                edge = self.edge_context_designer.design(plan_by_id[source], plan, "soft")
                edges.append(edge)
                nodes[source].outgoing_edges.append(edge)
                nodes[plan.scene_id].incoming_edges.append(edge)

        dag = SceneDAG(nodes=nodes, edges=edges)
        dag.topological_order()
        logger.info("dag_build_completed nodes=%s edges=%s", len(nodes), len(edges))
        return dag

    def _validate_dependency(self, source: str, target: str, plan_by_id: dict[str, ScenePlan]) -> None:
        """Validate that a dependency refers to an existing, distinct scene."""
        if source not in plan_by_id:
            raise ValueError(f"{target} depends on unknown scene {source}")
        if source == target:
            raise ValueError(f"{target} cannot depend on itself")
