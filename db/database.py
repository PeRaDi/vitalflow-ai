import os
import psycopg2
from psycopg2.extensions import connection
from typing import Dict, Optional

class Database:
    def __init__(self):
        self.conn: Optional[connection] = None

    def _load_config(self) -> Dict[str, str]:
        """Loads database configuration from environment variables."""
        return {
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT'),
            'database': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USERNAME'),
            'password': os.getenv('DATABASE_PASSWORD')
        }

    async def setup_database(self):
        """Establishes a connection to the database."""
        try:
            config = self._load_config()
            self.conn = psycopg2.connect(**config)
            print("<!> DB: Connection successfully established.")
        except (psycopg2.DatabaseError, Exception) as error:
            print(f"<!> DB: Failed to connect to the database: {error}")
            raise

    async def close_connection(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("<!> DB: Database connection closed.")

    def get_connection(self) -> connection:
        """Returns the active database connection or raises an error if not connected."""
        if not self.conn:
            raise RuntimeError("<!> DB: Database connection has not been established.")
        return self.conn
    
    def get_item_info(self, item_id: int) -> Dict:
        """Fetches item information from the database."""
        cursor = self.conn.cursor()
        cursor.execute(f"""SELECT id, lead_time, frequent_order FROM items WHERE id = {item_id};""")
        item_info = cursor.fetchone()
        
        cursor.close()
        if not item_info:
            return None
        return {
            'id': item_info[0],
            'lead_time': int(item_info[1]),
            'frequent_order': bool(item_info[2])
        }
    
    def get_item_data(self, item_id: int) -> Dict:
        """Fetches item data from the database."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT
                DATE(created_at) AS date,
                SUM(quantity) AS daily_quantity_out
            FROM stock_transactions
            WHERE item_id = {item_id} 
                AND transaction_type_id = 2
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at);     
        """)
        data = cursor.fetchall()
        
        cursor.close()
        if not data:
            return None

        return data