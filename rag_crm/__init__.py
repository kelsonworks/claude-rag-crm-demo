"""RAG assistant with CRM write-back — portfolio demo.

Pipeline: markdown docs -> section chunks -> TF-IDF retrieval -> Claude answer
with inline citations -> CRM record (HubSpot in real mode, local JSON in mock
mode).
"""

__version__ = "1.0.0"
