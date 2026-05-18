from dataclasses import dataclass, field

from story_engine.llm.client import LLMClient, GroqClient
from story_engine.models.llm_contracts import LLMRole


# Groq model mapping based on LLMRole complexity
# planner: Requires complex reasoning and scene sequencing - uses most capable model
# generator: Creative scene generation - balanced model
# extractor: State diff extraction - fast, smaller model
# verifier: Verification checks - balanced model
GROQ_MODEL_FOR_ROLE = {
    "planner": "llama-3.1-70b-versatile",    # Complex reasoning
    "generator": "qwen-2.5-32b",              # Creative generation
    "extractor": "gemma-2-9b-it",             # State extraction
    "verifier": "qwen-2.5-32b",               # Verification logic
    "default": "qwen-2.5-32b",                # Default model
}


@dataclass
class ModelRouter:
    clients: dict[LLMRole, LLMClient] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default Groq clients for each role if not provided."""
        if not self.clients:
            # Initialize Groq clients for all roles
            for role in GROQ_MODEL_FOR_ROLE.keys():
                role_typed: LLMRole = role  # type: ignore
                try:
                    model = GROQ_MODEL_FOR_ROLE.get(role)
                    self.clients[role_typed] = GroqClient(model="openai/gpt-oss-20b")
                except ValueError:
                    # API key not provided, skip initialization
                    pass

    def client_for(self, role: LLMRole) -> LLMClient | None:
        role = role if role in self.clients else "default"
        return self.clients.get(role)

    def with_client(self, role: LLMRole, client: LLMClient) -> "ModelRouter":
        next_clients = dict(self.clients)
        next_clients[role] = client
        return ModelRouter(next_clients)
