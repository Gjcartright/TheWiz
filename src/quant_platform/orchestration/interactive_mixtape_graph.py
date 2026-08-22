from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NotRequired, TypedDict

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CommandResult:
    paths: dict[str, Path]
    summary: dict[str, object]


class MixtapeAgentState(TypedDict):
    brief: str
    brand_name: str
    target_platform: str
    agents: dict[str, dict[str, object]]
    risks: list[str]
    solution: NotRequired[dict[str, object]]
    markdown: NotRequired[str]


AGENT_SEQUENCE = [
    "brand_strategist",
    "experience_architect",
    "commerce_architect",
    "content_lore_designer",
    "technical_planner",
    "qa_critic",
    "synthesizer",
]


def build_interactive_mixtape_solution(
    *,
    brief: str,
    brand_name: str = "The Ave",
    target_platform: str = "Shopify",
    root: Path = ROOT,
    output_dir: Path | None = None,
) -> CommandResult:
    output_base = output_dir or root / "reports" / "brand_workflows"
    output_base.mkdir(parents=True, exist_ok=True)

    graph = build_interactive_mixtape_graph()
    state = graph.invoke(
        {
            "brief": brief,
            "brand_name": brand_name,
            "target_platform": target_platform,
            "agents": {},
            "risks": [],
        }
    )

    slug = _slug(brand_name)
    md_path = output_base / f"{slug}_langgraph_solution.md"
    json_path = output_base / f"{slug}_langgraph_state.json"
    atomic_write_text(md_path, str(state["markdown"]), encoding="utf-8")
    atomic_write_text(json_path, json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    agents = state.get("agents", {})
    solution = state.get("solution", {})
    return CommandResult(
        paths={"mixtape_solution_md": md_path, "mixtape_solution_state": json_path},
        summary={
            "agents": len(agents),
            "workflow": " -> ".join(AGENT_SEQUENCE),
            "phases": len(solution.get("implementation_phases", [])) if isinstance(solution, dict) else 0,
            "zones": len(solution.get("core_zones", [])) if isinstance(solution, dict) else 0,
            "risks": len(state.get("risks", [])),
            "langgraph_runtime": _langgraph_available(),
        },
    )


def build_interactive_mixtape_graph() -> Any:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _FallbackGraph(AGENT_SEQUENCE)

    graph = StateGraph(MixtapeAgentState)
    for node_name in AGENT_SEQUENCE:
        graph.add_node(node_name, _NODE_FUNCTIONS[node_name])

    graph.add_edge(START, "brand_strategist")
    graph.add_edge("brand_strategist", "experience_architect")
    graph.add_edge("experience_architect", "commerce_architect")
    graph.add_edge("commerce_architect", "content_lore_designer")
    graph.add_edge("content_lore_designer", "technical_planner")
    graph.add_edge("technical_planner", "qa_critic")
    graph.add_edge("qa_critic", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


class _FallbackGraph:
    def __init__(self, node_names: list[str]) -> None:
        self.node_names = node_names

    def invoke(self, state: MixtapeAgentState) -> MixtapeAgentState:
        current = dict(state)
        for node_name in self.node_names:
            patch = _NODE_FUNCTIONS[node_name](current)  # type: ignore[arg-type]
            current.update(patch)
        return current  # type: ignore[return-value]


def _brand_strategist(state: MixtapeAgentState) -> dict[str, object]:
    brief = state["brief"]
    insights = [
        "Do not frame the store as a catalog; make product discovery feel like pressing play on a cultural artifact.",
        "Use neighborhoods as collection gateways so sneakers, apparel, collaborations, media, and events each have a memorable place.",
        "Treat every collection as a mixtape with track-style sequencing, drops, hidden bonuses, and story-led product copy.",
    ]
    if "easter" in brief.lower() or "hidden" in brief.lower():
        insights.append("Build a repeat-visit loop around hidden drops, unlockable art, and limited-access collaborations.")

    return _agent_patch(
        state,
        "brand_strategist",
        {
            "mission": f"Turn {state['brand_name']} into a shoppable hip hop culture world.",
            "tagline": "Press Play. Discover the Culture.",
            "positioning": (
                "A culture-first Shopify storefront led by a 35-year hip hop clothing and footwear "
                "executive: part sneaker boutique, part mixtape archive, part interactive drop room."
            ),
            "audience": [
                "sneaker collectors",
                "streetwear shoppers",
                "hip hop nostalgia fans",
                "drop hunters",
                "artist-collab buyers",
            ],
            "insights": insights,
            "success_metric": "Visitors understand the culture-first concept before seeing a normal product grid.",
        },
    )


def _experience_architect(state: MixtapeAgentState) -> dict[str, object]:
    return _agent_patch(
        state,
        "experience_architect",
        {
            "entry_scene": "Cassette lands, PLAY button pulses, tape spins, CRT texture flickers, then the city opens.",
            "world_map": [
                {"zone": "Sneaker District", "objects": ["basketball court", "sneaker shop", "subway stop"], "commerce": "shoe drops"},
                {"zone": "Fashion Avenue", "objects": ["vintage racks", "hoodie wall", "hat kiosk"], "commerce": "apparel"},
                {"zone": "Producer Alley", "objects": ["MPC", "turntables", "vinyl bins"], "commerce": "accessories and bundles"},
                {"zone": "Graffiti Tunnel", "objects": ["murals", "spray cans", "artist tags"], "commerce": "collabs and hidden drops"},
                {"zone": "Block Party", "objects": ["stage", "video wall", "crowd circle"], "commerce": "new releases and events"},
            ],
            "navigation_model": "Explorable city first, conventional collection/product fallbacks always available.",
            "interaction_rules": [
                "Pressing PLAY starts the visual journey, not audio autoplay.",
                "Hover, tap, keyboard focus, and search all reveal the same products.",
                "Each zone has one hero drop, one evergreen collection, and one hidden track.",
            ],
            "accessibility_note": "Every animated hotspot also appears in keyboard-accessible navigation and search.",
        },
    )


def _commerce_architect(state: MixtapeAgentState) -> dict[str, object]:
    return _agent_patch(
        state,
        "commerce_architect",
        {
            "shopify_structure": {
                "collections": "Mixtapes",
                "products": "Tracks",
                "tags": ["zone", "era", "drop", "artist", "rarity", "sound"],
                "metafields": ["lore", "sound_cue", "scene_object", "unlock_condition", "era_reference"],
            },
            "merchitecture": [
                {"lane": "Lead Singles", "role": "new sneaker and apparel drops with countdown energy"},
                {"lane": "Deep Cuts", "role": "evergreen essentials that keep conversion steady"},
                {"lane": "Features", "role": "artist, designer, DJ, and local-culture collaborations"},
                {"lane": "Hidden Tracks", "role": "limited products revealed by codes, murals, and email/SMS clues"},
            ],
            "purchase_flow": [
                "hotspot click opens quick-look track card",
                "track card shows lore, sizes, scarcity, and add-to-cart",
                "full product page gives record-player animation and deeper story",
                "cart keeps the mixtape theme while preserving standard checkout trust",
            ],
            "retention_loops": [
                "weekly hidden track",
                "artist collab mural unlock",
                "drop countdown in Block Party",
                "collector badges for customers who complete a mixtape bundle",
            ],
        },
    )


def _content_lore_designer(state: MixtapeAgentState) -> dict[str, object]:
    return _agent_patch(
        state,
        "content_lore_designer",
        {
            "voice": "cinematic, street-level, specific, never generic product-spec filler",
            "product_lore_template": {
                "origin": "Where in the city this piece was born",
                "era": "The music, sneaker, or streetwear memory it nods to",
                "use_case": "When the customer wears it",
                "detail": "One real material or construction fact woven into the story",
            },
            "sample_track": {
                "title": "Track 01: Late Train Low",
                "copy": "Built for the ride home after midnight battles: clean leather, city miles, and a sole that keeps moving.",
            },
            "launch_campaign": [
                "Side A: tease the cassette and first five zone names across social and email.",
                "Side B: reveal the Sneaker District lead single with a playable product card.",
                "Bonus Track: release a password-protected hidden product through mural clues.",
            ],
            "media": ["short loop per zone", "ambient city sound bed", "artist interview snippets", "drop teaser clips"],
        },
    )


def _technical_planner(state: MixtapeAgentState) -> dict[str, object]:
    return _agent_patch(
        state,
        "technical_planner",
        {
            "recommended_stack": [
                "Shopify Hydrogen for the immersive storefront",
                "Shopify Storefront API for products, carts, and collections",
                "Metaobjects/metafields for lore, hotspots, zones, and unlock rules",
                "Canvas or lightweight WebGL for the city map",
                "CSS motion with reduced-motion fallbacks for product moments",
            ],
            "data_model": [
                "Zone metaobject: name, art, ambient sound, collection handle",
                "Hotspot metaobject: coordinates, object type, linked product or mixtape, unlock condition",
                "Mixtape collection: title, era, cover art, track order",
                "Track product: lore, sound cue, scene object, rarity, Shopify product reference",
            ],
            "analytics_events": [
                "play_pressed",
                "zone_entered",
                "hotspot_opened",
                "track_quick_add",
                "hidden_track_unlocked",
                "mixtape_bundle_started",
                "checkout_started",
            ],
            "implementation_phases": [
                "Define the brand bible, zone map, mixtape taxonomy, and Shopify metafield schema.",
                "Prototype the city map with 5 zones, sample products, and keyboard-accessible hotspots.",
                "Connect Storefront API carts, product variants, collection filters, and checkout.",
                "Add the lore/metafield publishing workflow for drops, collaborations, and hidden tracks.",
                "Layer animation, explicit sound controls, reduced-motion mode, and unlockable Easter eggs.",
                "Run conversion QA, mobile QA, accessibility QA, analytics tagging, and launch rehearsal.",
            ],
        },
    )


def _qa_critic(state: MixtapeAgentState) -> dict[str, object]:
    risks = [
        "Animation can slow shopping if product access is buried.",
        "Audio can feel intrusive unless it starts muted with clear controls.",
        "Hidden drops can frustrate shoppers if core inventory is hard to find.",
        "Shopify content operations can get messy without strict metafield templates.",
    ]
    return _agent_patch(
        state,
        "qa_critic",
        {
            "risks": risks,
            "guardrails": [
                "Persistent search, cart, and zone menu on every screen.",
                "No autoplay audio; use explicit PLAY and visible mute.",
                "Every hotspot has a normal link equivalent.",
                "Measure add-to-cart rate by zone, product, and hotspot.",
            ],
        },
        risks=risks,
    )


def _synthesizer(state: MixtapeAgentState) -> dict[str, object]:
    agents = dict(state["agents"])
    technical = agents["technical_planner"]
    commerce = agents["commerce_architect"]
    solution = {
        "concept": f"{state['brand_name']} is a shoppable hip hop city where every collection is a mixtape and every product is a track.",
        "positioning": agents["brand_strategist"]["positioning"],
        "primary_experience": agents["experience_architect"]["entry_scene"],
        "core_zones": agents["experience_architect"]["world_map"],
        "commerce_model": commerce["shopify_structure"],
        "merchitecture": commerce["merchitecture"],
        "launch_campaign": agents["content_lore_designer"]["launch_campaign"],
        "analytics_events": technical["analytics_events"],
        "implementation_phases": technical["implementation_phases"],
        "must_have_controls": agents["qa_critic"]["guardrails"],
    }
    agents["synthesizer"] = {
        "role": "Merge brand, experience, commerce, content, technical, and QA recommendations into the launch-ready answer.",
        "decision": "Ship an immersive Shopify storefront with conventional commerce fallbacks and measured drop mechanics.",
    }
    markdown = _solution_markdown(state, solution)
    return {"agents": agents, "solution": solution, "markdown": markdown}


_NODE_FUNCTIONS = {
    "brand_strategist": _brand_strategist,
    "experience_architect": _experience_architect,
    "commerce_architect": _commerce_architect,
    "content_lore_designer": _content_lore_designer,
    "technical_planner": _technical_planner,
    "qa_critic": _qa_critic,
    "synthesizer": _synthesizer,
}


def _agent_patch(
    state: MixtapeAgentState,
    agent_name: str,
    payload: dict[str, object],
    risks: list[str] | None = None,
) -> dict[str, object]:
    agents = dict(state.get("agents", {}))
    agents[agent_name] = payload
    patch: dict[str, object] = {"agents": agents}
    if risks:
        patch["risks"] = [*state.get("risks", []), *risks]
    return patch


def _solution_markdown(state: MixtapeAgentState, solution: dict[str, object]) -> str:
    agents = state["agents"]
    zones = agents["experience_architect"]["world_map"]
    phases = solution["implementation_phases"]
    guardrails = solution["must_have_controls"]

    lines = [
        f"# {state['brand_name']} LangGraph Agent Workflow",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Platform target: {state['target_platform']}",
        "",
        "## Workflow",
        "",
        "brief -> brand strategist -> experience architect -> commerce architect -> content/lore designer -> technical planner -> QA critic -> synthesizer",
        "",
        "## Solution",
        "",
        str(solution["concept"]),
        "",
        "## Positioning",
        "",
        str(solution["positioning"]),
        "",
        "## First Screen",
        "",
        str(solution["primary_experience"]),
        "",
        "## Interactive City Zones",
        "",
    ]
    for zone in zones:
        if isinstance(zone, dict):
            lines.append(f"- {zone['zone']}: {zone['commerce']} through {', '.join(zone['objects'])}.")

    lines.extend(["", "## Commerce Build Model", ""])
    commerce_model = solution["commerce_model"]
    if isinstance(commerce_model, dict):
        for key, value in commerce_model.items():
            lines.append(f"- {key}: {value}")

    lines.extend(["", "## Merchitecture", ""])
    merchitecture = solution["merchitecture"]
    for lane in merchitecture:
        if isinstance(lane, dict):
            lines.append(f"- {lane['lane']}: {lane['role']}.")

    lines.extend(["", "## Launch Campaign", ""])
    launch_campaign = solution["launch_campaign"]
    for step in launch_campaign:
        lines.append(f"- {step}")

    lines.extend(["", "## Analytics Events", ""])
    analytics_events = solution["analytics_events"]
    for event_name in analytics_events:
        lines.append(f"- {event_name}")

    lines.extend(["", "## Implementation Phases", ""])
    for index, phase in enumerate(phases, start=1):
        lines.append(f"{index}. {phase}")

    lines.extend(["", "## Guardrails", ""])
    for guardrail in guardrails:
        lines.append(f"- {guardrail}")

    lines.extend(["", "## Sample Product Lore", ""])
    sample = agents["content_lore_designer"]["sample_track"]
    if isinstance(sample, dict):
        lines.append(f"**{sample['title']}**")
        lines.append("")
        lines.append(str(sample["copy"]))

    return "\n".join(lines) + "\n"


def _langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value.strip()]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "the-ave"
