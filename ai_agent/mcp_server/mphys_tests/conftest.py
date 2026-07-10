"""Make `import openmdao_server` resolve to THIS repo's copy (the sibling
ai_agent/mcp_server/openmdao_server.py), not any other copy on the machine."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
