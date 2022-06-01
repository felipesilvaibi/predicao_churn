from app.instance import server
from app.routes.predict import *

app = server.app

if __name__ == "__main__":
    server.run()