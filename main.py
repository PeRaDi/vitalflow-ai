import asyncio
import sys
import torch
from dotenv import load_dotenv
from db.database import Database
from rabbit_queue import RabbitQueue

class Node:
    def __init__(self, node_type: str):
        self.type = node_type
        self.db = Database()
        self.device = self._select_device()
        self.queue = RabbitQueue(self.db, self.type, self.device)
    
    def _select_device(self) -> torch.device:
        """Selects GPU if available, otherwise exits the program."""
        print("<!> Selecting GPU as main device <!>")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"<!> Using: {torch.cuda.get_device_name(0)} <!>")
            return torch.device("cuda")
        
        print("<!> No GPU available <!>")
        sys.exit(1)
    
    def __str__(self) -> str:
        return f'Node-{self.type}'

    def __repr__(self) -> str:
        return str(self)
    
    async def run(self):
        """Initializes the node, setting up the database and starting the queue."""
        print(f'[!] Setting up node as a {self.type}.')
        await self.db.setup_database()
        self.queue.start()

if __name__ == '__main__':
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python node.py <node_type>")
        sys.exit(1)
    
    node_type = sys.argv[1]
    node = Node(node_type)
    asyncio.run(node.run())