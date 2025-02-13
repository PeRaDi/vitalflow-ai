from tasks.forecaster import Forecaster
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
            )
        ))
        self.channel = self.connection.channel()
        
        self.queue_name = f'queue_{type}'

        if type == 'trainer':
            self.op = Trainer(db, device)
        elif type == 'forecaster':
            self.op = Forecaster(db)
            
        self.channel.queue_declare(queue=self.queue_name)
        
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.on_request)

    def on_request(self, ch, method, props, body):
        payload = json.loads(body)

        result = self.exec(payload["payload"])
        
        ch.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            properties=pika.BasicProperties(
                correlation_id=props.correlation_id
            ),
            body=json.dumps(result))
        
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def exec(self, payload):
        print(f" [.] Executing task with payload {payload}")
        result = self.op.exec(payload)
        return {"result": result}
    
    # def forecast(self, payload):
    #     print(f" [.] Forecasting with payload {payload}")
    #     result = self.forecaster.exec(payload)
    #     return {"result": result}

    def start(self):
        print(f"[!] Node listening on {self.queue_name}")
        self.channel.start_consuming()