from tools.email_tool import EmailTool
from tools.translation_tool import TranslationTool
from tools.retrieval_tool import KnowledgeBaseRetriever

search_fn = KnowledgeBaseRetriever("Qwen/Qwen3-Embedding-0.6B")

tool_list = {
    "email": EmailTool(),
    "retrieve": search_fn,
}


# === Other Tools ===
#"translation": TranslationTool(),
# PDF/HTML reports from data or summary inputs
# customer_insight
# policy_lookup