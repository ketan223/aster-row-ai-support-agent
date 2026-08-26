import uuid

# Memory dictionary mapping session_id strings to list of messages.
_sessions = {}
# Tracks if a session has triggered human handoff
_handoffs = {}

# Retrieves the conversation history for a given session ID.
# If the session does not exist, it initializes an empty conversation history.
def get_session_history(session_id: str) -> list[dict]:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]

# Appends a user or assistant message to the given session's history.
def add_message_to_session(session_id: str, role: str, content: str):
    if not session_id:
        return
    history = get_session_history(session_id)
    history.append({"role": role, "content": content})

# Clears all conversation history associated with the given session ID.
def clear_session(session_id: str):
    print(f"[MEMORY DEBUG] clear_session({session_id})")
    if session_id in _sessions:
        _sessions[session_id] = []
    _handoffs[session_id] = False

# Sets the handoff status for a session.
def set_session_handoff(session_id: str, handoff: bool):
    print(f"[MEMORY DEBUG] set_session_handoff({session_id}, {handoff})")
    _handoffs[session_id] = handoff

# Returns True if handoff was triggered for the session.
def get_session_handoff(session_id: str) -> bool:
    val = _handoffs.get(session_id, False)
    print(f"[MEMORY DEBUG] get_session_handoff({session_id}) -> {val}")
    return val
