
from abc import ABC, abstractmethod

"""
    MMAgent下所有ComputerUseAgent的基类
"""
class ComputerUseBaseAgent(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass
    