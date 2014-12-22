from unittest import TestCase
from memory.memoryd import MemoryD

__author__ = 'deepmind'


class TestMemoryD(TestCase):

    def setUp(self):
        self.memory = MemoryD(10)

    def test_memory_size_at_initialization(self):
        self.assertTrue(self.memory.size == 10)
