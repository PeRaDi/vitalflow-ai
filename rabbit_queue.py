from tasks.forecaster.forecaster import Forecaster
from tasks.trainer import Trainer
import pika
import json
import os

class RabbitQueue:
    def __init__(self, db, type, device):
        self.type = type
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host = os.getenv('RABBITMQ_HOST'),
            port = os.getenv('RABBITMQ_PORT'),
            credentials = pika.PlainCredentials(
                os.getenv('RABBITMQ_USERNAME'),
                os.getenv('RABBITMQ_PASSWORD')
            ),
            heartbeat=0, 
            blocked_connection_timeout=None,  
            socket_timeout=None, 
            connection_attempts=1, 
            retry_delay=0  
        ))
        self.channel = self.connection.channel()
        
        self.queue_name = f'queue_{type}'
        self.backend_queue_name = os.getenv('RABBITMQ_BACKEND_QUEUE')

        if type == 'trainer':
            self.op = Trainer(db, device)
        elif type == 'forecaster':
            self.op = Forecaster(db, device)
            
        self.channel.queue_declare(queue=self.queue_name)
        self.channel.queue_declare(queue=self.backend_queue_name, durable=True)
        
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.on_request)

    def on_request(self, ch, method, props, body):
        data = json.loads(body)["data"]
        job_id = data["job_id"]
        
        ch.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            properties=pika.BasicProperties(
                correlation_id=props.correlation_id
            ),
            body=(json.dumps({"status": "PROCESSING"}))
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

        result = self.exec(data)
        self.post_to_backend_queue(job_id, result)

    def exec(self, payload):
        try:
            result = self.op.exec(payload)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def post_to_backend_queue(self, job_id, message):
        try:
            message = {
                "pattern": self.type,
                "data": {
                    "job_id": job_id,
                    "result": message
                }
            }

            self.channel.basic_publish(
                exchange='', 
                routing_key=os.getenv('RABBITMQ_BACKEND_QUEUE'),
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                )
            )
        except Exception as e:
            print(f"Error posting to backend queue: {str(e)}")
    
    def start(self):
        print(f"<!> Node listening on {self.queue_name}")
        self.channel.start_consuming()