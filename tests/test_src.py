from src.generateS import Generate
from src.memory import ChatbotMemory

def test_memory():
    mem = ChatbotMemory()
    mem.update_memory('test', 'test')
    assert mem.get_memory() == ["'user': test, 'bot': test"]

def test_generate():
    from src.generateS import Generate
    mod = Generate(model='test')
    assert mod.model == 'test'