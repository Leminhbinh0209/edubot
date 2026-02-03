import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import tempfile 
from mybot.session.manager import SessionManager

def test_session_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(Path(tmpdir))
        session = manager.get_or_create("test:123")
        session.add_message("user", "Hello")
        manager.save(session)
        
        # Reload
        session2 = manager.get_or_create("test:123")
        assert len(session2.messages) == 1
if __name__ == "__main__":
    print("Testing session manager...")
    test_session_manager()
    print("✓ Session manager test passed")