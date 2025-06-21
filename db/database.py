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
            print("[DB] Connection successfully established.")
        except (psycopg2.DatabaseError, Exception) as error:
            print(f"[DB] Failed to connect to the database: {error}")
            raise

    async def close_connection(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("[DB] Database connection closed.")

    def get_connection(self) -> connection:
        """Returns the active database connection or raises an error if not connected."""
        if not self.conn:
            raise RuntimeError("[DB] Database connection has not been established.")
        return self.conn
    
    def get_item_data(self, item_id: int) -> Dict:
        """Fetches item data from the database."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT
                DATE(st.created_at) AS date,
                SUM(st.quantity) AS daily_quantity_out
            FROM stock_transactions st
            JOIN item_stock ist ON st.item_stock_id = ist.id
            WHERE ist.item_id = {item_id} 
                AND st.transaction_type_id = 2
            GROUP BY DATE(st.created_at)
            ORDER BY DATE(st.created_at);     
        """)
        data = cursor.fetchall()
        
        cursor.close()
        if not data:
            return None

        return data